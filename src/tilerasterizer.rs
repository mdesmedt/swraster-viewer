use crate::renderer::{RasterMessage, RasterPacket, RenderBuffer};
use crate::scene::{Material, Scene};
use crossbeam_channel::Receiver;
use glam::{BVec4, BVec4A, IVec2, UVec4, Vec4};

// The tile which the rasterizer uses to consume work
// NOTE: The color and depth values are stored using SIMD vectors
pub struct TileRasterizer {
    pub screen_min: IVec2,
    pub screen_max: IVec2,
    pub channel: Receiver<RasterMessage>,
    pub color: Vec<UVec4>,
    pub depth: Vec<Vec4>,
}

impl TileRasterizer {
    // Copy the tile to the render buffer
    pub fn copy_to_buffer(&self, buffer: &mut RenderBuffer) {
        let width = (self.screen_max.x - self.screen_min.x) as usize;
        let height = (self.screen_max.y - self.screen_min.y) as usize;
        let width_vec4 = (width + 3) / 4;

        for y in 0..height {
            for x_vec4 in 0..width_vec4 {
                let src_index = y * width_vec4 + x_vec4;
                let dst_y = self.screen_min.y + y as i32;

                let colors = self.color[src_index].to_array();
                let dst_base_x = self.screen_min.x + x_vec4 as i32 * 4;

                // Copy each component of the Vec4 to individual pixels
                for i in 0..4 {
                    let pixel_x = dst_base_x + i as i32;
                    let dst_index = dst_y as usize * buffer.width + pixel_x as usize;
                    // Need to check here because the buffer is not guaranteed to match our vec4 alignment
                    if dst_index < buffer.pixels.len() {
                        buffer.pixels[dst_index] = colors[i as usize];
                    }
                }
            }
        }
    }

    fn clear(&mut self) {
        self.color.fill(UVec4::ZERO);
        self.depth.fill(Vec4::splat(f32::INFINITY));
    }

    // Begin rasterization and start listening to the channel
    pub fn begin_rasterization(&mut self, scene: &Scene) {
        // Clear the tile first
        self.clear();
        // Then start listening to the channel
        loop {
            match self.channel.recv() {
                Ok(RasterMessage::Packet(packet)) => self.rasterize_packet(scene, packet),
                Ok(RasterMessage::Terminate) => break,
                Err(_) => break, // Channel was closed
            }
        }
    }

    // Main rasterizer code
    fn rasterize_packet(&mut self, scene: &Scene, packet: RasterPacket) {
        let x_start = packet.screen_min.x & !3; // Align left to 4-pixels for SIMD alignment
        let y_start = packet.screen_min.y;
        let mut p = IVec2::new(x_start, y_start);

        // Fetch the light
        let mesh_index = packet.mesh_index;
        let mesh = &scene.meshes[mesh_index];
        let light = &mesh.light;
        let light_x = Vec4::splat(light.normal.x);
        let light_y = Vec4::splat(light.normal.y);
        let light_z = Vec4::splat(light.normal.z);

        // Fetch the material
        let primitive_index = packet.primitive_index;
        let mesh_index = packet.mesh_index;
        let mesh = &scene.meshes[mesh_index];
        let material_index = mesh.primitives[primitive_index].material_index;
        let material: Option<&Material> = scene.materials.get(material_index.unwrap());

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
                        is_alpha_tested,
                        &packet,
                        material,
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
        is_alpha_tested: bool,
        packet: &RasterPacket,
        material: Option<&Material>,
    ) {
        // Compute the barycentrics
        let bary0: Vec4 = w0 * packet.one_over_area_vec;
        let bary1: Vec4 = w1 * packet.one_over_area_vec;
        let bary2: Vec4 = w2 * packet.one_over_area_vec;

        // Helper function for attribute interpolation
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

        let normal_x = interpolate_vertex_attribute(
            packet.normal_over_w[0].x,
            packet.normal_over_w[1].x,
            packet.normal_over_w[2].x,
            bary0,
            bary1,
            bary2,
        ) * w;
        let normal_y = interpolate_vertex_attribute(
            packet.normal_over_w[0].y,
            packet.normal_over_w[1].y,
            packet.normal_over_w[2].y,
            bary0,
            bary1,
            bary2,
        ) * w;
        let normal_z = interpolate_vertex_attribute(
            packet.normal_over_w[0].z,
            packet.normal_over_w[1].z,
            packet.normal_over_w[2].z,
            bary0,
            bary1,
            bary2,
        ) * w;

        // "Pixel shader" which computes color values for the 4 pixels

        // Compute normal
        let length_squared = normal_x * normal_x + normal_y * normal_y + normal_z * normal_z;
        let one_over_length = 1.0
            / Vec4::new(
                length_squared.x.sqrt(),
                length_squared.y.sqrt(),
                length_squared.z.sqrt(),
                length_squared.w.sqrt(),
            );
        let normalized_x = normal_x * one_over_length;
        let normalized_y = normal_y * one_over_length;
        let normalized_z = normal_z * one_over_length;

        // Apply N.L lighting
        let dot_x = normalized_x * light_x;
        let dot_y = normalized_y * light_y;
        let dot_z = normalized_z * light_z;
        let mut light_intensity = (dot_x + dot_y + dot_z).clamp(Vec4::ZERO, Vec4::ONE);

        // TODO: Add better ambient light somehow
        light_intensity += Vec4::splat(0.2);

        // Start the color with irradiance, currently monochrome
        let mut color_r = light_intensity;
        let mut color_g = light_intensity;
        let mut color_b = light_intensity;

        if let Some(material) = material {
            // Apply base color factor
            color_r *= material.base_color_factor.x;
            color_g *= material.base_color_factor.y;
            color_b *= material.base_color_factor.z;
            // If we have a base texture, sample it
            if let Some(diffuse_texture) = &material.base_color_texture {
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

                // Sample base texture for each pixel
                let diffuse_vec = diffuse_texture.sample_vec4(uv_x, uv_y);
                // Simple gamma 2.0 instead of 2.2 for perf
                color_r *= diffuse_vec[0] * diffuse_vec[0];
                color_g *= diffuse_vec[1] * diffuse_vec[1];
                color_b *= diffuse_vec[2] * diffuse_vec[2];

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
            }
        } else {
            // Missing material
            // TODO: Guarantee a default material instead?
        }

        // Clamp final colors before packing
        color_r = color_r.clamp(Vec4::ZERO, Vec4::ONE);
        color_g = color_g.clamp(Vec4::ZERO, Vec4::ONE);
        color_b = color_b.clamp(Vec4::ZERO, Vec4::ONE);

        fn linear_to_srgb(color: Vec4) -> Vec4 {
            // Apply gamma 2.0 instead of 2.2 for perf
            Vec4::new(
                color.x.sqrt(),
                color.y.sqrt(),
                color.z.sqrt(),
                color.w.sqrt(),
            )
        }

        // Gamma correct the final colors before packing
        color_r = linear_to_srgb(color_r);
        color_g = linear_to_srgb(color_g);
        color_b = linear_to_srgb(color_b);

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
