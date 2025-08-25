use crate::bumpqueue::BumpQueue;
use crate::math::*;
use crate::rendercamera::RenderCamera;
use crate::renderer::RasterPacket;
use crate::scene::{Material, Scene};
use glam::{BVec4, BVec4A, IVec2, UVec4, Vec3, Vec4};

// Helper functions for attribute interpolation

fn interpolate_attribute(
    a: f32,
    b: f32,
    c: f32,
    bary0: Vec4,
    bary1: Vec4,
    bary2: Vec4,
) -> Vec4 {
    a * bary0 + b * bary1 + c * bary2
}

fn interpolate_attribute_vec3x4(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    bary0: Vec4,
    bary1: Vec4,
    bary2: Vec4,
) -> Vec3x4 {
    let x = interpolate_attribute(a.x, b.x, c.x, bary0, bary1, bary2);
    let y = interpolate_attribute(a.y, b.y, c.y, bary0, bary1, bary2);
    let z = interpolate_attribute(a.z, b.z, c.z, bary0, bary1, bary2);
    Vec3x4::new(x, y, z)
}

fn bvec4_to_bvec4a(b: BVec4) -> BVec4A {
    let arr: [bool; 4] = b.into();
    BVec4A::from(arr)
}

fn bvec4a_to_bvec4(b: BVec4A) -> BVec4 {
    let arr: [bool; 4] = b.into();
    BVec4::from(arr)
}

struct ShadingParams<'a> {
    p: IVec2,
    bary0: Vec4,
    bary1: Vec4,
    bary2: Vec4,
    mask: BVec4A,
    light_dir: Vec3x4,
    packet: &'a RasterPacket,
    material: &'a Material,
    camera: &'a RenderCamera,
    scene: &'a Scene,
}

// The tile which the rasterizer uses to consume work
// NOTE: The color and depth values are stored using SIMD vectors
pub struct TileRasterizer {
    pub screen_min: IVec2,
    pub screen_max: IVec2,
    pub packets_opaque: BumpQueue<RasterPacket>,
    pub packets_translucent: BumpQueue<RasterPacket>,
    pub color: Vec<UVec4>,
    pub depth: Vec<Vec4>,
    // Internal "visibility-buffer"
    pub packet_index: Vec<UVec4>,
    pub bary0: Vec<Vec4>,
    pub bary1: Vec<Vec4>,
}

impl TileRasterizer {
    pub fn render_tile(&mut self, scene: &Scene, camera: &RenderCamera) {
        // Clear depth
        self.depth.fill(Vec4::splat(f32::INFINITY));

        // Render opaque packets
        for packet_index in 0..self.packets_opaque.len() {
            self.rasterize_packet::<false>(scene, camera, packet_index);
        }

        // Shade opaque pixels
        self.shade_opaque_pixels(scene, camera);

        // Fill unwritten pixels with skybox
        self.fill_with_skybox(scene, camera);

        // Sort translucent packets by z, back to front
        self.packets_translucent
            .sort_by(|a, b| b.avg_z.cmp(&a.avg_z));

        // Render translucent packets
        for packet_index in 0..self.packets_translucent.len() {
            self.rasterize_packet::<true>(scene, camera, packet_index);
        }

        // Reset the queues
        self.packets_opaque.reset();
        self.packets_translucent.reset();
    }

    /// Fill depth=infinity pixels with skybox color
    pub fn fill_with_skybox(&mut self, scene: &Scene, camera: &RenderCamera) {
        let tile_width = (self.screen_max.x - self.screen_min.x) as usize;
        let tile_height = (self.screen_max.y - self.screen_min.y) as usize;
        let width_quads = tile_width / 2;

        let inv_viewproj = camera.inverse_view_project_matrix;
        let inv_viewproj_transposed = inv_viewproj.transpose();

        let rcp_camera_width = Vec4::splat(1.0 / camera.width as f32);
        let rcp_camera_height = Vec4::splat(1.0 / camera.height as f32);

        // Loop over rows of quads
        for y in 0..tile_height / 2 {
            let screen_y = self.screen_min.y + (y as i32) * 2;

            // Loop over Vec4s in the row
            for quad_x in 0..width_quads {
                let screen_x = self.screen_min.x + (quad_x as i32 * 2);
                let index = self.index_from_xy(screen_x, screen_y);

                let depth = self.depth[index];
                let depth_mask = depth.cmpeq(Vec4::INFINITY);
                if !depth_mask.any() {
                    continue;
                }

                let pixel_x = Vec4::new(
                    (screen_x as f32) + 0.5,
                    (screen_x as f32) + 1.5,
                    (screen_x as f32) + 0.5,
                    (screen_x as f32) + 1.5,
                );
                let pixel_y = Vec4::new(
                    (screen_y as f32) + 0.5,
                    (screen_y as f32) + 0.5,
                    (screen_y as f32) + 1.5,
                    (screen_y as f32) + 1.5,
                );

                let ndc_x = pixel_x * rcp_camera_width * 2.0 - Vec4::ONE;
                let ndc_y = (Vec4::ONE - pixel_y  * rcp_camera_height) * 2.0 - Vec4::ONE;

                let clip_pos = Vec3x4::new(ndc_x, ndc_y, Vec4::ONE);

                // Transform by the transposed inverse view-projection matrix
                let world_dir = Vec3x4::transform_direction_transposed(inv_viewproj_transposed, clip_pos);
                let world_normal = world_dir.normalize();

                // Sample the cubemap
                let cubemap_sample = scene.cubemap.sample_cubemap(world_normal);
                let cubemap_color = Vec3x4::new(cubemap_sample.col(0), cubemap_sample.col(1), cubemap_sample.col(2));
                
                // Pack color
                let out_colors = pack_colors(cubemap_color);

                // Use depth test to selectively write color
                let booleans: [bool; 4] = depth_mask.into();
                let mask_bvec4 = BVec4::from(booleans);
                let current_color = self.color[index];
                let color = UVec4::select(mask_bvec4, out_colors, current_color);
                self.color[index] = color;
            }
        }
    }

    // Main rasterizer code
    fn rasterize_packet<const FORWARD_SHADING: bool>(
        &mut self,
        scene: &Scene,
        camera: &RenderCamera,
        packet_index: usize,
    ) {
        let packet = if FORWARD_SHADING {
            debug_assert!(packet_index < self.packets_translucent.len());
            self.packets_translucent.get(packet_index)
        } else {
            debug_assert!(packet_index < self.packets_opaque.len());
            self.packets_opaque.get(packet_index)
        };
        let packet_index_vec = UVec4::splat(packet_index as u32);

        let x_start = packet.screen_min.x & !1; // Align to quads
        let y_start = packet.screen_min.y & !1;
        let mut p = IVec2::new(x_start, y_start);

        // Fetch the light
        let light = &scene.light;
        let light_dir = Vec3x4::new(
            Vec4::splat(light.direction.x),
            Vec4::splat(light.direction.y),
            Vec4::splat(light.direction.z),
        );

        // Fetch the material
        let mesh_index = packet.mesh_index as usize;
        let primitive_index = packet.primitive_index as usize;
        let mesh = &scene.meshes[mesh_index];
        let material_index = mesh.primitives[primitive_index].material_index;
        let material = &scene.materials[material_index];

        // Get triangle vertices in screen space
        let p0 = packet.pos_screen[0];
        let p1 = packet.pos_screen[1];
        let p2 = packet.pos_screen[2];

        // Invert one over area to take the winding switch into account
        let one_over_area_vec = Vec4::splat(-packet.one_over_area);

        // Set up edge function coefficients for edges v0->v1, v1->v2, v2->v0

        // Edge v0->v1: (y1-y0)x + (x0-x1)y + (x1*y0 - x0*y1) = 0
        let a01 = p1.y - p0.y;
        let b01 = p0.x - p1.x;
        let c01 = p1.x * p0.y - p0.x * p1.y;

        // Edge v1->v2: (y2-y1)x + (x1-x2)y + (x2*y1 - x1*y2) = 0
        let a12 = p2.y - p1.y;
        let b12 = p1.x - p2.x;
        let c12 = p2.x * p1.y - p1.x * p2.y;

        // Edge v2->v0: (y0-y2)x + (x2-x0)y + (x0*y2 - x2*y0) = 0
        let a20 = p0.y - p2.y;
        let b20 = p2.x - p0.x;
        let c20 = p0.x * p2.y - p2.x * p0.y;

        // Step deltas for 4-pixel SIMD stepping
        const STEP_X_SIZE: i32 = 2;
        const STEP_Y_SIZE: i32 = 2;

        let step_x01 = Vec4::splat(a01 * STEP_X_SIZE as f32);
        let step_x12 = Vec4::splat(a12 * STEP_X_SIZE as f32);
        let step_x20 = Vec4::splat(a20 * STEP_X_SIZE as f32);

        let step_y01 = Vec4::splat(b01 * STEP_Y_SIZE as f32);
        let step_y12 = Vec4::splat(b12 * STEP_Y_SIZE as f32);
        let step_y20 = Vec4::splat(b20 * STEP_Y_SIZE as f32);

        // Initial edge function values at the starting pixel block with half-pixel offset
        let x = Vec4::new(
            p.x as f32 + 0.5,
            p.x as f32 + 1.5,
            p.x as f32 + 0.5,
            p.x as f32 + 1.5,
        );
        let y = Vec4::new(
            p.y as f32 + 0.5,
            p.y as f32 + 0.5,
            p.y as f32 + 1.5,
            p.y as f32 + 1.5,
        );

        // Edge function values: w0 = edge v1->v2, w1 = edge v2->v0, w2 = edge v0->v1
        let mut w0_row = Vec4::splat(a12) * x + Vec4::splat(b12) * y + Vec4::splat(c12);
        let mut w1_row = Vec4::splat(a20) * x + Vec4::splat(b20) * y + Vec4::splat(c20);
        let mut w2_row = Vec4::splat(a01) * x + Vec4::splat(b01) * y + Vec4::splat(c01);

        // Main loop for rasterizing the triangle
        while p.y < packet.screen_max.y {
            let mut w0 = w0_row;
            let mut w1 = w1_row;
            let mut w2 = w2_row;

            p.x = x_start;
            while p.x < packet.screen_max.x {
                // Test if any pixels are inside the triangle
                let mask = w0.cmpge(Vec4::ZERO) & w1.cmpge(Vec4::ZERO) & w2.cmpge(Vec4::ZERO);
                if mask.any() {
                    // Compute the barycentrics
                    let bary0: Vec4 = w0 * one_over_area_vec;
                    let bary1: Vec4 = w1 * one_over_area_vec;
                    let bary2: Vec4 = w2 * one_over_area_vec;

                    // Interpolate w and z
                    let w = 1.0
                    / interpolate_attribute(
                        packet.one_over_w[0],
                        packet.one_over_w[1],
                        packet.one_over_w[2],
                        bary0,
                        bary1,
                        bary2,
                    );
        
                    let z = interpolate_attribute(
                        packet.z_over_w[0],
                        packet.z_over_w[1],
                        packet.z_over_w[2],
                        bary0,
                        bary1,
                        bary2,
                    ) * w;

                    // Perform depth test
                    let (out_depth, mut out_mask) = self.depth_test(p.x, p.y, z, mask);

                    let shading_params = ShadingParams {
                        p,
                        bary0,
                        bary1,
                        bary2,
                        mask: out_mask,
                        light_dir,
                        packet: &packet,
                        material,
                        camera,
                        scene,
                    };
                    
                    if FORWARD_SHADING {
                        // Forward shade translucent pixels
                        self.shade_pixels::<true>(shading_params);
                    }
                    else {
                        // Store "visibility" buffer for opaque pixels
                        if out_mask.any() {
                            if material.is_alpha_tested {
                                let alpha_test_mask = self.get_alpha_test_mask(shading_params, w);
                                out_mask &= alpha_test_mask;
                            }
                            // Store depth, packet index and barycentrics
                            let index = self.index_from_xy(p.x, p.y);
                            let mask_bvec4 = bvec4a_to_bvec4(out_mask);
                            self.packet_index[index] = UVec4::select(mask_bvec4, packet_index_vec, self.packet_index[index]);
                            self.bary0[index] = Vec4::select(out_mask, bary0, self.bary0[index]);
                            self.bary1[index] = Vec4::select(out_mask, bary1, self.bary1[index]);
                            self.depth[index] = Vec4::select(out_mask, out_depth, self.depth[index]);
                        }
                    }
                }

                // Step in X
                w0 += step_x12;
                w1 += step_x20;
                w2 += step_x01;
                p.x += STEP_X_SIZE;
            }

            // Step in Y
            w0_row += step_y12;
            w1_row += step_y20;
            w2_row += step_y01;
            p.y += STEP_Y_SIZE;
        }
    }

    fn shade_opaque_pixels(&mut self, scene: &Scene, camera: &RenderCamera) {
        let tile_width = (self.screen_max.x - self.screen_min.x) as usize;
        let tile_height = (self.screen_max.y - self.screen_min.y) as usize;
        let width_quads = tile_width / 2;
        
        // Loop over rows of quads
        for y in 0..tile_height / 2 {
            let screen_y = self.screen_min.y + (y as i32) * 2;

            // Loop over Vec4s in the row
            for quad_x in 0..width_quads {
                let screen_x = self.screen_min.x + (quad_x as i32 * 2);
                let index = self.index_from_xy(screen_x, screen_y);

                let packet_index_vec = self.packet_index[index];

                let depth = self.depth[index];
                let depth_mask = depth.cmpne(Vec4::INFINITY);

                // TODO: On depth=INF pixels just compute the skybox here

                // Consume all possible packets in packet_index uniformly
                let mut remaining = depth_mask;
                while remaining.any() {
                    // Find the first active lane (smallest i where remaining[i] == true)
                    let lane = remaining.bitmask().trailing_zeros() as usize;

                    // Extract that lane's value
                    let packet_index = packet_index_vec[lane];

                    // Build mask for all lanes equal to this value
                    let mask = bvec4_to_bvec4a(packet_index_vec.cmpeq(UVec4::splat(packet_index)));

                    let bary0 = self.bary0[index];
                    let bary1 = self.bary1[index];
                    let bary2 = Vec4::ONE - bary0 - bary1;
                    let packet = self.packets_opaque.get(packet_index as usize);

                    // Fetch the material
                    let mesh_index = packet.mesh_index as usize;
                    let primitive_index = packet.primitive_index as usize;
                    let mesh = &scene.meshes[mesh_index];
                    let material_index = mesh.primitives[primitive_index].material_index;
                    let material = &scene.materials[material_index];
    
                    let p = IVec2::new(screen_x, screen_y);
                    let light_dir = Vec3x4::new(
                        Vec4::splat(scene.light.direction.x),
                        Vec4::splat(scene.light.direction.y),
                        Vec4::splat(scene.light.direction.z),
                    );
                    let packet = &packet;
    
                    let shading_params = ShadingParams {
                        p,
                        bary0,
                        bary1,
                        bary2,
                        mask,
                        light_dir,
                        packet: &packet,
                        material,
                        camera,
                        scene,
                    };
    
                    self.shade_pixels::<false>(shading_params);

                    // Clear out the lanes we just handled
                    remaining &= !mask;
                }
            }
        }
    }

    fn get_alpha_test_mask(&self, shading_params: ShadingParams, w: Vec4) -> BVec4A {
        let bary0 = shading_params.bary0;
        let bary1 = shading_params.bary1;
        let bary2 = shading_params.bary2;
        let packet = shading_params.packet;
        let material = shading_params.material;
        let mask = shading_params.mask;
        
        let uv_x = interpolate_attribute(
            packet.uv_over_w[0].x,
            packet.uv_over_w[1].x,
            packet.uv_over_w[2].x,
            bary0,
            bary1,
            bary2,
        ) * w;
        let uv_y = interpolate_attribute(
            packet.uv_over_w[0].y,
            packet.uv_over_w[1].y,
            packet.uv_over_w[2].y,
            bary0,
            bary1,
            bary2,
        ) * w;

        let mut out_mask = mask;
        if let Some(diffuse_texture) = &material.base_color_texture {
            let diffuse_mat = diffuse_texture.sample4(uv_x, uv_y);
            let alpha = diffuse_mat.col(3);
            out_mask &= alpha.cmpge(material.alpha_cutoff_vec);
        }

        out_mask
    }

    // "Pixel shader" for the rasterizer. Shades 4 pixels simultaneously.
    #[inline]
    fn shade_pixels<const FORWARD_SHADING: bool>(
        &mut self,
        shading_params: ShadingParams,
    ) {
        let p = shading_params.p;
        let bary0 = shading_params.bary0;
        let bary1 = shading_params.bary1;
        let bary2 = shading_params.bary2;
        let mask = shading_params.mask;
        let light_dir = shading_params.light_dir;
        let packet = shading_params.packet;
        let material = shading_params.material;
        let camera = shading_params.camera;
        let scene = shading_params.scene;

        // Begin attribute interpolation

        let w = 1.0
            / interpolate_attribute(
                packet.one_over_w[0],
                packet.one_over_w[1],
                packet.one_over_w[2],
                bary0,
                bary1,
                bary2,
            );

        let input_normal = interpolate_attribute_vec3x4(
            packet.normals[0],
            packet.normals[1],
            packet.normals[2],
            bary0,
            bary1,
            bary2,
        )
        .normalize();

        let input_tangent = interpolate_attribute_vec3x4(
            packet.tangents[0],
            packet.tangents[1],
            packet.tangents[2],
            bary0,
            bary1,
            bary2,
        )
        .normalize();

        let pos_world = interpolate_attribute_vec3x4(
            packet.pos_world_over_w[0],
            packet.pos_world_over_w[1],
            packet.pos_world_over_w[2],
            bary0,
            bary1,
            bary2,
        ) * w;

        let uv_x = interpolate_attribute(
            packet.uv_over_w[0].x,
            packet.uv_over_w[1].x,
            packet.uv_over_w[2].x,
            bary0,
            bary1,
            bary2,
        ) * w;
        let uv_y = interpolate_attribute(
            packet.uv_over_w[0].y,
            packet.uv_over_w[1].y,
            packet.uv_over_w[2].y,
            bary0,
            bary1,
            bary2,
        ) * w;

        let mut normal_world = input_normal;

        // Apply normal mapping if we have a normal map
        if let Some(normal_map) = &material.normal_texture {
            let normal_map_mat = normal_map.sample4(uv_x, uv_y);
            let mut tangent_space_normal = (Vec3x4::new(
                normal_map_mat.col(0),
                normal_map_mat.col(1),
                normal_map_mat.col(2),
            )) * 2.0
                - 1.0;
            tangent_space_normal = tangent_space_normal.normalize();
            let binormal = input_normal.cross(input_tangent);
            normal_world = input_tangent * tangent_space_normal.x
                + binormal * tangent_space_normal.y
                + input_normal * tangent_space_normal.z;
        }

        // Compute N.L diffuse lighting
        let n_dot_l = normal_world.dot(light_dir);
        let diffuse = n_dot_l.clamp(Vec4::ZERO, Vec4::ONE);

        // Compute the view direction for each pixel
        let view_dir = Vec3x4::from_vec3(camera.position) - pos_world;
        let view_normal = view_dir.normalize();
        // Compute half vector
        let half_vector_add = light_dir + view_normal;
        // Normalize half vector
        let half_vector = half_vector_add.normalize();
        // H.N
        let n_dot_h = normal_world.dot(half_vector);
        // Exponentiate
        let n_dot_h_2 = n_dot_h * n_dot_h;
        let n_dot_h_4 = n_dot_h_2 * n_dot_h_2;
        let n_dot_h_8 = n_dot_h_4 * n_dot_h_4;
        let n_dot_h_16 = n_dot_h_8 * n_dot_h_8;
        let n_dot_h_32 = n_dot_h_16 * n_dot_h_16;

        // More ambient light coming from the top, peak intensity of 0.15
        let ambient = (normal_world.y + Vec4::splat(1.5)) * Vec4::splat((0.5 / 1.5) * 0.2);

        // Get voxel grid lighting if available, for shadows
        let voxel_light_intensity = if let Some(voxel_grid) = &scene.voxel_grid {
            voxel_grid.get_filtered_light_intensity_vec(pos_world)
        } else {
            Vec4::ONE
        };

        let light_intensity = diffuse * voxel_light_intensity + ambient;

        // Diffuse starts with the base color factor
        let mut diffuse = Vec3x4::from_f32(
            material.base_color_factor.x,
            material.base_color_factor.y,
            material.base_color_factor.z,
        );

        let mut out_mask = mask;

        // If we have a base texture, sample it
        if let Some(diffuse_texture) = &material.base_color_texture {
            let diffuse_mat = diffuse_texture.sample4(uv_x, uv_y);

            // Alpha test
            if material.is_alpha_tested {
                let alpha = diffuse_mat.col(3);
                out_mask &= alpha.cmpge(material.alpha_cutoff_vec);
            }

            // Multiply diffuse color
            // Simple gamma 2.0 instead of 2.2 for perf
            diffuse.x *= diffuse_mat.col(0) * diffuse_mat.col(0);
            diffuse.y *= diffuse_mat.col(1) * diffuse_mat.col(1);
            diffuse.z *= diffuse_mat.col(2) * diffuse_mat.col(2);
        }

        let mut roughness = Vec4::splat(material.roughness_factor);
        let mut metallic = Vec4::splat(material.metallic_factor);

        // If we have a metallic/roughness texture, sample it
        if let Some(spec_texture) = &material.metallic_roughness_texture {
            let metallic_roughness_mat = spec_texture.sample4(uv_x, uv_y);
            roughness *= metallic_roughness_mat.col(1);
            metallic *= metallic_roughness_mat.col(2);
        }
        let dielectric = Vec4::ONE - metallic;

        // Compute specular
        let shininess = Vec4::ONE - roughness;
        let specular_shiny = n_dot_h_32;
        let specular_rough = n_dot_h_2 * Vec4::splat(0.01);
        let n_dot_h_blend = specular_shiny * shininess + specular_rough * roughness;
        let specular = n_dot_h_blend * voxel_light_intensity;

        // Now compute the color, starting with irradiance
        // TODO: This is wrong for metallic, but without filtered cubemaps it's tricky to fix
        let mut color = Vec3x4::from_vec4(light_intensity);

        // For translucent materials, blend in the current framebuffer color
        if FORWARD_SHADING {
            let mut transmission = Vec4::splat(material.transmission);

            // Sample transmission texture if provided
            if let Some(transmission_texture) = &material.transmission_texture {
                let transmission_mat = transmission_texture.sample4(uv_x, uv_y);
                transmission *= transmission_mat.col(0);
            }

            // Multiply 1-transmission to diffuse lighting
            let inv_transmission = Vec4::ONE - transmission;
            color *= inv_transmission;

            // Add the transmitted color multiplied by the transmission factor
            let current_color_packed = self.color[self.index_from_xy(p.x, p.y)];
            let current_color = unpack_colors(current_color_packed);
            color += current_color * transmission;
        }

        // Multiply by diffuse color
        color *= diffuse;

        // Add dielectric specular
        color += specular * dielectric;

        // Add metallic specular
        color += diffuse * specular * metallic;

        // Sample cubemap (with some totally arbitrary weighting)
        // TODO: Currently fades out cubemap with roughness because we lack cubemap mipmaps
        let reflect = view_normal.reflect(normal_world);
        let cubemap_mat = scene.cubemap.sample_cubemap(reflect * -1.0);
        let dielectric_strength =
            ((shininess - Vec4::splat(0.6)) * Vec4::splat(0.22)).clamp(Vec4::ZERO, Vec4::ONE);
        let metallic_strength = metallic * shininess;
        let cubemap_color = Vec3x4::new(
            cubemap_mat.col(0) * cubemap_mat.col(0),
            cubemap_mat.col(1) * cubemap_mat.col(1),
            cubemap_mat.col(2) * cubemap_mat.col(2),
        );
        // Add dielectric cubemap contribution
        color += cubemap_color * dielectric_strength;
        // Add metallic cubemap contribution
        color += cubemap_color * diffuse * metallic_strength;

        // Debug: Show world space position
        //color = pos_world;

        // Debug: Show normal
        // color = input_normal;

        // Debug: Show voxel lighting
        // color.x = voxel_light_intensity;
        // color.y = voxel_light_intensity;
        // color.z = voxel_light_intensity;

        // Debug: Show cubemap
        // color.x = cubemap_mat.col(0);
        // color.y = cubemap_mat.col(1);
        // color.z = cubemap_mat.col(2);

        // Clamp final colors before packing
        color = color.clamp(Vec4::ZERO, Vec4::ONE);

        let packed_colors = pack_colors(color);
        self.write_pixels(p.x, p.y, packed_colors, out_mask);
    }

    // Perform depth test and return final depth and a mask of the pixels that passed depth testing
    #[inline]
    fn depth_test(&mut self, x: i32, y: i32, depth: Vec4, mask: BVec4A) -> (Vec4, BVec4A) {
        let index = self.index_from_xy(x, y);
        // Load current depth values
        let current_depth = self.depth[index];
        // Perform depth test
        let mask_depth = depth.cmple(current_depth);
        // Create final mask which determines which pixels to shade and write
        let final_mask = mask & mask_depth;
        // Do the depth select here because we already have everything we need
        let final_depth = Vec4::select(final_mask, depth, current_depth);

        (final_depth, final_mask)
    }

    // Perform depth test and write color and depth
    #[inline]
    fn write_pixels(
        &mut self,
        x: i32,
        y: i32,
        color: UVec4,
        final_mask: BVec4A,
    ) {
        let index = self.index_from_xy(x, y);
        let booleans: [bool; 4] = final_mask.into();
        let mask_bvec4 = BVec4::from(booleans);

        // Select final color
        let current_color = self.color[index];
        let final_color = UVec4::select(mask_bvec4, color, current_color);

        // Write color and depth
        self.color[index] = final_color;
    }

    fn index_from_xy(&self, x: i32, y: i32) -> usize {
        debug_assert!((x - self.screen_min.x) % 2 == 0 || (x - self.screen_min.x) % 2 == 1, "x within bounds");
        debug_assert!((y - self.screen_min.y) % 2 == 0 || (y - self.screen_min.y) % 2 == 1, "y within bounds");
    
        let local_x = x - self.screen_min.x;
        let local_y = y - self.screen_min.y;
    
        let width = self.screen_max.x - self.screen_min.x;
        let width_quads = width / 2;
    
        let quad_x = local_x / 2;
        let quad_y = local_y / 2;
    
        let index = quad_y * width_quads + quad_x;
        index as usize
    }
}

fn pack_colors(color: Vec3x4) -> UVec4 {
    // Gamma correct the final colors before packing
    // Apply gamma 2.0 instead of 2.2 for perf
    let color_r = sqrt_vec(color.x);
    let color_g = sqrt_vec(color.y);
    let color_b = sqrt_vec(color.z);

    UVec4::new(
        ((color_r.x * 255.0) as u32) << 16
            | ((color_g.x * 255.0) as u32) << 8
            | ((color_b.x * 255.0) as u32),
        ((color_r.y * 255.0) as u32) << 16
            | ((color_g.y * 255.0) as u32) << 8
            | ((color_b.y * 255.0) as u32),
        ((color_r.z * 255.0) as u32) << 16
            | ((color_g.z * 255.0) as u32) << 8
            | ((color_b.z * 255.0) as u32),
        ((color_r.w * 255.0) as u32) << 16
            | ((color_g.w * 255.0) as u32) << 8
            | ((color_b.w * 255.0) as u32),
    )
}

fn unpack_colors(color: UVec4) -> Vec3x4 {
    // Extract red (bits 16-23)
    let color_r = Vec4::new(
        ((color.x >> 16) & 0xFF) as f32 / 255.0,
        ((color.y >> 16) & 0xFF) as f32 / 255.0,
        ((color.z >> 16) & 0xFF) as f32 / 255.0,
        ((color.w >> 16) & 0xFF) as f32 / 255.0,
    );

    // Extract green (bits 8-15)
    let color_g = Vec4::new(
        ((color.x >> 8) & 0xFF) as f32 / 255.0,
        ((color.y >> 8) & 0xFF) as f32 / 255.0,
        ((color.z >> 8) & 0xFF) as f32 / 255.0,
        ((color.w >> 8) & 0xFF) as f32 / 255.0,
    );

    // Extract blue (bits 0-7)
    let color_b = Vec4::new(
        (color.x & 0xFF) as f32 / 255.0,
        (color.y & 0xFF) as f32 / 255.0,
        (color.z & 0xFF) as f32 / 255.0,
        (color.w & 0xFF) as f32 / 255.0,
    );

    Vec3x4::new(color_r * color_r, color_g * color_g, color_b * color_b)
}
