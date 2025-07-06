use crate::bumpqueue::BumpQueue;
use crate::math::*;
use crate::rendercamera::RenderCamera;
use crate::renderer::RasterPacket;
use crate::scene::{Material, Scene};
use glam::{BVec4, BVec4A, IVec2, UVec4, Vec4};

// The tile which the rasterizer uses to consume work
// NOTE: The color and depth values are stored using SIMD vectors
pub struct TileRasterizer {
    pub screen_min: IVec2,
    pub screen_max: IVec2,
    pub packets: BumpQueue<RasterPacket>,
    pub color: Vec<UVec4>,
    pub depth: Vec<Vec4>,
}

impl TileRasterizer {
    fn clear(&mut self) {
        self.color.fill(UVec4::ZERO);
        self.depth.fill(Vec4::splat(f32::INFINITY));
    }

    pub fn rasterize_packets(&mut self, scene: &Scene, camera: &RenderCamera) {
        // Clear the tile first
        self.clear();
        // Then consume all packets in the queue
        while let Some(packet) = self.packets.pop() {
            self.rasterize_packet(scene, camera, packet);
        }
        // Reset the queue
        self.packets.reset();
    }

    // Main rasterizer code
    fn rasterize_packet(&mut self, scene: &Scene, camera: &RenderCamera, packet: RasterPacket) {
        let x_start = packet.screen_min.x & !3; // Align left to 4-pixels for SIMD alignment
        let y_start = packet.screen_min.y;
        let mut p = IVec2::new(x_start, y_start);

        // Fetch the light
        let light = &scene.light;
        let light_x = Vec4::splat(light.direction.x);
        let light_y = Vec4::splat(light.direction.y);
        let light_z = Vec4::splat(light.direction.z);

        // Fetch the material
        let mesh_index = packet.mesh_index as usize;
        let primitive_index = packet.primitive_index as usize;
        let mesh = &scene.meshes[mesh_index];
        let material_index = mesh.primitives[primitive_index].material_index;
        let material: Option<&Material> =
            material_index.and_then(|index| scene.materials.get(index));

        // Determine whether we can do early z testing
        let is_alpha_tested = if let Some(material) = material {
            material.is_alpha_tested
        } else {
            false
        };

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
                    self.shade_pixels(
                        p,
                        w0,
                        w1,
                        w2,
                        mask_aligned,
                        light_x,
                        light_y,
                        light_z,
                        one_over_area_vec,
                        is_alpha_tested,
                        &packet,
                        material,
                        camera,
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
    fn shade_pixels(
        &mut self,
        p: IVec2,
        w0: Vec4,
        w1: Vec4,
        w2: Vec4,
        mask: BVec4A,
        light_x: Vec4,
        light_y: Vec4,
        light_z: Vec4,
        one_over_area_vec: Vec4,
        is_alpha_tested: bool,
        packet: &RasterPacket,
        material: Option<&Material>,
        camera: &RenderCamera,
    ) {
        // Compute the barycentrics
        let bary0: Vec4 = w0 * one_over_area_vec;
        let bary1: Vec4 = w1 * one_over_area_vec;
        let bary2: Vec4 = w2 * one_over_area_vec;

        // Helper function for attribute interpolation
        #[inline]
        fn interpolate_vertex_attribute(
            a: f32,
            b: f32,
            c: f32,
            bary0: Vec4,
            bary1: Vec4,
            bary2: Vec4,
        ) -> Vec4 {
            a * bary0 + b * bary1 + c * bary2
        }

        // Begin attribute interpolation

        let w = 1.0
            / interpolate_vertex_attribute(
                packet.one_over_w[0],
                packet.one_over_w[1],
                packet.one_over_w[2],
                bary0,
                bary1,
                bary2,
            );

        let z = interpolate_vertex_attribute(
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
        if !is_alpha_tested {
            (out_depth, out_mask) = self.depth_test(p.x, p.y, z, mask);
            if !out_mask.any() {
                return;
            }
        }

        let input_normal_x = interpolate_vertex_attribute(
            packet.normals[0].x,
            packet.normals[1].x,
            packet.normals[2].x,
            bary0,
            bary1,
            bary2,
        );
        let input_normal_y = interpolate_vertex_attribute(
            packet.normals[0].y,
            packet.normals[1].y,
            packet.normals[2].y,
            bary0,
            bary1,
            bary2,
        );
        let input_normal_z = interpolate_vertex_attribute(
            packet.normals[0].z,
            packet.normals[1].z,
            packet.normals[2].z,
            bary0,
            bary1,
            bary2,
        );

        // "Pixel shader" which computes color values for the 4 pixels

        // Compute normal
        let length_squared = input_normal_x * input_normal_x
            + input_normal_y * input_normal_y
            + input_normal_z * input_normal_z;
        let one_over_length = rsqrt_vec(length_squared);
        let normal_x = input_normal_x * one_over_length;
        let normal_y = input_normal_y * one_over_length;
        let normal_z = input_normal_z * one_over_length;

        // Apply N.L lighting
        let n_dot_l_x = normal_x * light_x;
        let n_dot_l_y = normal_y * light_y;
        let n_dot_l_z = normal_z * light_z;
        let diffuse = (n_dot_l_x + n_dot_l_y + n_dot_l_z).clamp(Vec4::ZERO, Vec4::ONE);

        let pos_world_x = interpolate_vertex_attribute(
            packet.pos_world_over_w[0].x,
            packet.pos_world_over_w[1].x,
            packet.pos_world_over_w[2].x,
            bary0,
            bary1,
            bary2,
        ) * w;
        let pos_world_y = interpolate_vertex_attribute(
            packet.pos_world_over_w[0].y,
            packet.pos_world_over_w[1].y,
            packet.pos_world_over_w[2].y,
            bary0,
            bary1,
            bary2,
        ) * w;
        let pos_world_z = interpolate_vertex_attribute(
            packet.pos_world_over_w[0].z,
            packet.pos_world_over_w[1].z,
            packet.pos_world_over_w[2].z,
            bary0,
            bary1,
            bary2,
        ) * w;

        // Compute the view direction for each pixel
        let view_dir_x = camera.position.x - pos_world_x;
        let view_dir_y = camera.position.y - pos_world_y;
        let view_dir_z = camera.position.z - pos_world_z;
        let view_dir_length_squared =
            view_dir_x * view_dir_x + view_dir_y * view_dir_y + view_dir_z * view_dir_z;
        let one_over_view_dir_length = rsqrt_vec(view_dir_length_squared);
        let view_normal_x = view_dir_x * one_over_view_dir_length;
        let view_normal_y = view_dir_y * one_over_view_dir_length;
        let view_normal_z = view_dir_z * one_over_view_dir_length;

        // Compute half vector
        let half_vector_add_x = light_x + view_normal_x;
        let half_vector_add_y = light_y + view_normal_y;
        let half_vector_add_z = light_z + view_normal_z;
        // Normalize half vector
        let half_vector_length_squared = half_vector_add_x * half_vector_add_x
            + half_vector_add_y * half_vector_add_y
            + half_vector_add_z * half_vector_add_z;
        let one_over_half_vector_length = rsqrt_vec(half_vector_length_squared);
        let half_vector_x = half_vector_add_x * one_over_half_vector_length;
        let half_vector_y = half_vector_add_y * one_over_half_vector_length;
        let half_vector_z = half_vector_add_z * one_over_half_vector_length;
        // H.N
        let n_dot_h_x = normal_x * half_vector_x;
        let n_dot_h_y = normal_y * half_vector_y;
        let n_dot_h_z = normal_z * half_vector_z;
        let n_dot_h = (n_dot_h_x + n_dot_h_y + n_dot_h_z).clamp(Vec4::ZERO, Vec4::ONE);
        // Exponentiate
        let n_dot_h_2 = n_dot_h * n_dot_h;
        let n_dot_h_4 = n_dot_h_2 * n_dot_h_2;
        let n_dot_h_8 = n_dot_h_4 * n_dot_h_4;
        let n_dot_h_16 = n_dot_h_8 * n_dot_h_8;
        let n_dot_h_32 = n_dot_h_16 * n_dot_h_16;

        // More ambient light coming from the top, peak intensity of 0.1
        let ambient = (normal_y + Vec4::splat(1.5)) * Vec4::splat((0.5 / 1.5) * 0.1);

        let light_intensity = diffuse + ambient;

        let mut color_r = light_intensity;
        let mut color_g = light_intensity;
        let mut color_b = light_intensity;

        if let Some(material) = material {
            // Apply base color factor
            color_r *= material.base_color_factor.x;
            color_g *= material.base_color_factor.y;
            color_b *= material.base_color_factor.z;

            // Interpolate UVs
            let uv_x = interpolate_vertex_attribute(
                packet.uv_over_w[0].x,
                packet.uv_over_w[1].x,
                packet.uv_over_w[2].x,
                bary0,
                bary1,
                bary2,
            ) * w;
            let uv_y = interpolate_vertex_attribute(
                packet.uv_over_w[0].y,
                packet.uv_over_w[1].y,
                packet.uv_over_w[2].y,
                bary0,
                bary1,
                bary2,
            ) * w;

            // If we have a base texture, sample it
            if let Some(diffuse_texture) = &material.base_color_texture {
                let diffuse_vec = diffuse_texture.sample_vec4(uv_x, uv_y);

                // Alpha test
                if is_alpha_tested {
                    let alpha = diffuse_vec[3];
                    out_mask &= alpha.cmpge(material.alpha_cutoff_vec);

                    // Now do late Z
                    (out_depth, out_mask) = self.depth_test(p.x, p.y, z, out_mask);
                    if !out_mask.any() {
                        return;
                    }
                }

                // Multiply diffuse color
                // Simple gamma 2.0 instead of 2.2 for perf
                color_r *= diffuse_vec[0] * diffuse_vec[0];
                color_g *= diffuse_vec[1] * diffuse_vec[1];
                color_b *= diffuse_vec[2] * diffuse_vec[2];
            }

            let mut metallic = Vec4::splat(material.metallic_factor);
            let mut roughness = Vec4::splat(material.roughness_factor);

            // If we have a metallic/roughness texture, sample it
            if let Some(spec_texture) = &material.metallic_roughness_texture {
                let metallic_roughness_vec = spec_texture.sample_vec4(uv_x, uv_y);
                metallic *= metallic_roughness_vec[0];
                roughness *= metallic_roughness_vec[1];
            }

            // HACK: Roughness just reduces specular intensity
            let roughness_hack = 1.0 - roughness;
            let specular = metallic * roughness_hack * roughness_hack * n_dot_h_32;
            color_r += specular;
            color_g += specular;
            color_b += specular;
        } else {
            // Missing material
            // TODO: Guarantee a default material instead?

            // Add default most shiny specular
            color_r += n_dot_h_32;
            color_g += n_dot_h_32;
            color_b += n_dot_h_32;
        }

        // Debug: Show world space position
        // color_r = pos_world_x;
        // color_g = pos_world_y;
        // color_b = pos_world_z;

        // Debug: Show normal
        // color_r = normal_x;
        // color_g = normal_y;
        // color_b = normal_z;

        // Debug: Show ambient
        // color_r = ambient;
        // color_g = ambient;
        // color_b = ambient;

        // Clamp final colors before packing
        color_r = color_r.clamp(Vec4::ZERO, Vec4::ONE);
        color_g = color_g.clamp(Vec4::ZERO, Vec4::ONE);
        color_b = color_b.clamp(Vec4::ZERO, Vec4::ONE);

        // Gamma correct the final colors before packing
        // Apply gamma 2.0 instead of 2.2 for perf
        color_r = sqrt_vec(color_r);
        color_g = sqrt_vec(color_g);
        color_b = sqrt_vec(color_b);

        // Pack colors into integers
        let packed_colors = UVec4::new(
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
        );

        self.write_pixels(p.x, p.y, packed_colors, out_depth, out_mask);
    }

    // Perform depth test and return final depth and a mask of the pixels that passed depth testing
    #[inline]
    fn depth_test(&mut self, x: i32, y: i32, depth: Vec4, mask: BVec4A) -> (Vec4, BVec4A) {
        debug_assert_eq!(x % 4, 0, "x must be a multiple of 4 for SIMD alignment");
        let local_y = y - self.screen_min.y;
        let width = (self.screen_max.x - self.screen_min.x) as usize;
        let width_vec4 = (width + 3) / 4;

        // x is the leftmost pixel of the 4-pixel SIMD block
        let local_x = x - self.screen_min.x;
        let vec4_index = local_x as usize / 4;

        let index = local_y as usize * width_vec4 + vec4_index;
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
        final_depth: Vec4,
        final_mask: BVec4A,
    ) {
        debug_assert_eq!(x % 4, 0, "x must be a multiple of 4 for SIMD alignment");
        let local_y = y - self.screen_min.y;
        let width = (self.screen_max.x - self.screen_min.x) as usize;
        let width_vec4 = (width + 3) / 4;

        // x is the leftmost pixel of the 4-pixel SIMD block
        let local_x = x - self.screen_min.x;
        let vec4_index = local_x as usize / 4;
        let index = local_y as usize * width_vec4 + vec4_index;
        let booleans: [bool; 4] = final_mask.into();
        let mask_bvec4 = BVec4::from(booleans);

        // Select final color
        let current_color = self.color[index];
        let final_color = UVec4::select(mask_bvec4, color, current_color);

        // Write color and depth
        self.color[index] = final_color;
        self.depth[index] = final_depth;
    }
}
