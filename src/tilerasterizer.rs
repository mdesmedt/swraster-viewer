use crate::bumpqueue::BumpQueue;
use crate::math::*;
use crate::rendercamera::RenderCamera;
use crate::renderer::RasterPacket;
use crate::scene::{Material, Scene};
use glam::{BVec4, BVec4A, IVec2, UVec4, Vec3, Vec4};

// The tile which the rasterizer uses to consume work
// NOTE: The color and depth values are stored using SIMD vectors
pub struct TileRasterizer {
    pub screen_min: IVec2,
    pub screen_max: IVec2,
    pub packets_opaque: BumpQueue<RasterPacket>,
    pub packets_translucent: BumpQueue<RasterPacket>,
    pub color: Vec<UVec4>,
    pub depth: Vec<Vec4>,
}

impl TileRasterizer {
    pub fn rasterize_packets(&mut self, scene: &Scene, camera: &RenderCamera) {
        // Fill the tile with the skybox
        self.fill_with_skybox(scene, camera);
        // Clear depth
        self.depth.fill(Vec4::splat(f32::INFINITY));
        // Opaque packets
        while let Some(packet) = self.packets_opaque.pop() {
            self.rasterize_packet::<true>(scene, camera, packet);
        }
        // Sort translucent packets by z, back to front
        self.packets_translucent
            .sort_by(|a, b| b.avg_z.cmp(&a.avg_z));
        // Translucent packets first get collected and then rendered
        while let Some(packet) = self.packets_translucent.pop() {
            self.rasterize_packet::<false>(scene, camera, packet);
        }
        // Reset the queues
        self.packets_opaque.reset();
        self.packets_translucent.reset();
    }

    pub fn fill_with_skybox(&mut self, scene: &Scene, camera: &RenderCamera) {
        let inv_viewproj = camera.inverse_view_project_matrix;
        let tile_width = (self.screen_max.x - self.screen_min.x) as usize;
        let tile_height = (self.screen_max.y - self.screen_min.y) as usize;
        let width_vec4 = (tile_width + 3) / 4; // SIMD alignment

        // Loop over rows
        for y in 0..tile_height {
            let screen_y = self.screen_min.y + y as i32;

            // Loop over Vec4s in the row
            for x_vec4 in 0..width_vec4 {
                let screen_x = self.screen_min.x + (x_vec4 as i32 * 4);
                let index = self.index_from_xy(screen_x, screen_y);
                let mut out_colors = UVec4::ZERO;

                // Compute one pixel at a time for now
                // TODO: Optimize with SIMD and with simpler math
                for i in 0..4 {
                    let pixel_x = screen_x + i as i32;

                    let ndc_x = (pixel_x as f32 + 0.5) / camera.width as f32 * 2.0 - 1.0;
                    let ndc_y = (1.0 - (screen_y as f32 + 0.5) / camera.height as f32) * 2.0 - 1.0;

                    let clip_pos = Vec4::new(ndc_x, ndc_y, 1.0, 1.0); // Use +1.0 for far plane
                    let world_dir = (inv_viewproj * clip_pos).truncate().normalize();

                    // Sample the cubemap
                    let cubemap_color = scene.cubemap.sample_cubemap(
                        Vec4::splat(world_dir.x),
                        Vec4::splat(world_dir.y),
                        Vec4::splat(world_dir.z),
                    );

                    let r = cubemap_color.col(0).x;
                    let g = cubemap_color.col(1).x;
                    let b = cubemap_color.col(2).x;

                    // Pack color
                    let packed_color =
                        ((r * 255.0) as u32) << 16 | ((g * 255.0) as u32) << 8 | (b * 255.0) as u32;
                    match i {
                        0 => out_colors.x = packed_color,
                        1 => out_colors.y = packed_color,
                        2 => out_colors.z = packed_color,
                        3 => out_colors.w = packed_color,
                        _ => unreachable!(),
                    }
                }
                self.color[index] = out_colors;
            }
        }
    }

    // Main rasterizer code
    fn rasterize_packet<const MODE_OPAQUE: bool>(
        &mut self,
        scene: &Scene,
        camera: &RenderCamera,
        packet: RasterPacket,
    ) {
        let x_start = packet.screen_min.x & !3; // Align left to 4-pixels for SIMD alignment
        let y_start = packet.screen_min.y;
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
        const STEP_X_SIZE: i32 = 4;
        const STEP_Y_SIZE: i32 = 1;

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
            p.x as f32 + 2.5,
            p.x as f32 + 3.5,
        );
        let y = Vec4::splat(p.y as f32 + 0.5);

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
                    // Convert unaligned mask to aligned mask
                    // TODO: Figure out why glam needs two BVec4 types
                    let booleans: [bool; 4] = mask.into();
                    let mask_aligned = BVec4A::from(booleans);
                    self.shade_pixels::<MODE_OPAQUE>(
                        p,
                        w0,
                        w1,
                        w2,
                        mask_aligned,
                        light_dir,
                        one_over_area_vec,
                        &packet,
                        material,
                        camera,
                        scene,
                    );
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

    // "Pixel shader" for the rasterizer. Shades 4 pixels simultaneously.
    #[inline]
    fn shade_pixels<const MODE_OPAQUE: bool>(
        &mut self,
        p: IVec2,
        w0: Vec4,
        w1: Vec4,
        w2: Vec4,
        mask: BVec4A,
        light_dir: Vec3x4,
        one_over_area_vec: Vec4,
        packet: &RasterPacket,
        material: &Material,
        camera: &RenderCamera,
        scene: &Scene,
    ) {
        // Compute the barycentrics
        let bary0: Vec4 = w0 * one_over_area_vec;
        let bary1: Vec4 = w1 * one_over_area_vec;
        let bary2: Vec4 = w2 * one_over_area_vec;

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

        let z = interpolate_attribute(
            packet.z_over_w[0],
            packet.z_over_w[1],
            packet.z_over_w[2],
            bary0,
            bary1,
            bary2,
        ) * w;

        let mut out_depth = Vec4::splat(f32::INFINITY);
        let mut out_mask = mask;

        // Early Z test when not alpha testing
        if !material.is_alpha_tested {
            (out_depth, out_mask) = self.depth_test(p.x, p.y, z, mask);
            if !out_mask.any() {
                return;
            }
        }

        let input_normal = interpolate_attribute_vec3x4(
            packet.normals[0],
            packet.normals[1],
            packet.normals[2],
            bary0,
            bary1,
            bary2,
        ).normalize();

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

        // Compute N.L diffuse lighting
        let n_dot_l = input_normal.dot(light_dir);
        let diffuse = n_dot_l.clamp(Vec4::ZERO, Vec4::ONE);

        // Compute the view direction for each pixel
        let view_dir = Vec3x4::from_vec3(camera.position) - pos_world;
        let view_normal = view_dir.normalize();
        // Compute half vector
        let half_vector_add = light_dir + view_normal;
        // Normalize half vector
        let half_vector = half_vector_add.normalize();
        // H.N
        let n_dot_h = input_normal.dot(half_vector);
        // Exponentiate
        let n_dot_h_2 = n_dot_h * n_dot_h;
        let n_dot_h_4 = n_dot_h_2 * n_dot_h_2;
        let n_dot_h_8 = n_dot_h_4 * n_dot_h_4;
        let n_dot_h_16 = n_dot_h_8 * n_dot_h_8;
        let n_dot_h_32 = n_dot_h_16 * n_dot_h_16;

        // More ambient light coming from the top, peak intensity of 0.15
        let ambient = (input_normal.y + Vec4::splat(1.5)) * Vec4::splat((0.5 / 1.5) * 0.2);

        // Get voxel grid lighting if available, for shadows
        let voxel_light_intensity = if let Some(voxel_grid) = &scene.voxel_grid {
            voxel_grid.get_filtered_light_intensity_vec(pos_world)
        } else {
            Vec4::splat(1.0)
        };

        let light_intensity = diffuse * voxel_light_intensity + ambient;

        // Initialize color vectors with diffuse lighting
        let mut color = Vec3x4::new(light_intensity, light_intensity, light_intensity);

        if !MODE_OPAQUE {
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

        // Apply base color factor
        color.x *= material.base_color_factor.x;
        color.y *= material.base_color_factor.y;
        color.z *= material.base_color_factor.z;

        // If we have a base texture, sample it
        if let Some(diffuse_texture) = &material.base_color_texture {
            let diffuse_mat = diffuse_texture.sample4(uv_x, uv_y);

            // Alpha test
            if material.is_alpha_tested {
                let alpha = diffuse_mat.col(3);
                out_mask &= alpha.cmpge(material.alpha_cutoff_vec);

                // Now do late Z
                (out_depth, out_mask) = self.depth_test(p.x, p.y, z, out_mask);
                if !out_mask.any() {
                    return;
                }
            }

            // Multiply diffuse color
            // Simple gamma 2.0 instead of 2.2 for perf
            color.x *= diffuse_mat.col(0) * diffuse_mat.col(0);
            color.y *= diffuse_mat.col(1) * diffuse_mat.col(1);
            color.z *= diffuse_mat.col(2) * diffuse_mat.col(2);
        }

        let mut roughness = Vec4::splat(material.roughness_factor);

        // If we have a metallic/roughness texture, sample it
        if let Some(spec_texture) = &material.metallic_roughness_texture {
            let metallic_roughness_mat = spec_texture.sample4(uv_x, uv_y);
            roughness *= metallic_roughness_mat.col(1);
        }

        // Compute specular
        let shininess = Vec4::splat(1.0) - roughness;
        let specular_shiny = n_dot_h_32;
        let specular_rough = n_dot_h_2 * Vec4::splat(0.01);
        let n_dot_h_blend = specular_shiny * shininess + specular_rough * roughness;
        let specular = n_dot_h_blend * voxel_light_intensity;

        color += specular;

        // Reflection vector
        let reflect = view_normal.reflect(input_normal);

        // Sample cubemap (totally arbitrarily)
        let cubemap_mat = scene
            .cubemap
            .sample_cubemap(-reflect.x, -reflect.y, -reflect.z);
        let cubemap_strength =
            ((shininess - Vec4::splat(0.6)) * Vec4::splat(0.22)).clamp(Vec4::ZERO, Vec4::ONE);
        color.x += cubemap_mat.col(0) * cubemap_mat.col(0) * cubemap_strength;
        color.y += cubemap_mat.col(1) * cubemap_mat.col(1) * cubemap_strength;
        color.z += cubemap_mat.col(2) * cubemap_mat.col(2) * cubemap_strength;

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

        if MODE_OPAQUE {
            self.write_pixels_opaque(p.x, p.y, packed_colors, out_depth, out_mask);
        } else {
            self.write_pixels_translucent(p.x, p.y, packed_colors, out_mask);
        }
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
    fn write_pixels_opaque(
        &mut self,
        x: i32,
        y: i32,
        color: UVec4,
        final_depth: Vec4,
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
        self.depth[index] = final_depth;
    }

    fn write_pixels_translucent(&mut self, x: i32, y: i32, color: UVec4, final_mask: BVec4A) {
        let index = self.index_from_xy(x, y);
        let booleans: [bool; 4] = final_mask.into();
        let mask_bvec4 = BVec4::from(booleans);

        let current_color = self.color[index];

        // Select final color
        let final_color = UVec4::select(mask_bvec4, color, current_color);

        // Write color and depth
        self.color[index] = final_color;
    }

    fn index_from_xy(&self, x: i32, y: i32) -> usize {
        debug_assert_eq!(x % 4, 0, "x must be a multiple of 4 for SIMD alignment");
        let local_y = y - self.screen_min.y;
        let width = (self.screen_max.x - self.screen_min.x) as usize;
        let width_vec4 = (width + 3) / 4;

        // x is the leftmost pixel of the 4-pixel SIMD block
        let local_x = x - self.screen_min.x;
        let vec4_index = local_x as usize / 4;

        let index = local_y as usize * width_vec4 + vec4_index;
        index
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
