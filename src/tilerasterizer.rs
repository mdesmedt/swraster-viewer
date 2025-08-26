use crate::bumpqueue::BumpQueue;
use crate::math::*;
use crate::rendercamera::RenderCamera;
use crate::renderer::RasterPacket;
use crate::scene::Scene;
use crate::shader::{
    pbr_shader, PbrShaderParams, RasterizerShader, RasterizerShaderParams,
    TranslucentForwardShader, VBufferOpaqueShader,
};
use glam::{BVec4, BVec4A, IVec2, UVec4, Vec4};

// The tile which the rasterizer uses to consume work
// NOTE: The color and depth values are stored using SIMD vectors
pub struct TileRasterizer {
    pub screen_min: IVec2,
    pub screen_max: IVec2,
    pub packets_opaque: BumpQueue<RasterPacket>,
    pub packets_translucent: BumpQueue<RasterPacket>,
    // Packed color and depth
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

        // Render opaque packets into the vbuffer
        let opaque_shader = VBufferOpaqueShader;
        for packet_index in 0..self.packets_opaque.len() {
            let packet = self.packets_opaque.get(packet_index);
            self.rasterize_packet(scene, camera, packet_index, &packet, &opaque_shader);
        }

        // Shade opaque pixels
        self.shade_vbuffer(scene, camera);

        // Fill unwritten pixels with skybox
        self.fill_with_skybox(scene, camera);

        // Sort translucent packets by z, back to front
        self.packets_translucent
            .sort_by(|a, b| b.avg_z.cmp(&a.avg_z));

        // Render translucent packets
        let translucent_shader = TranslucentForwardShader;
        for packet_index in 0..self.packets_translucent.len() {
            let packet = self.packets_translucent.get(packet_index);
            self.rasterize_packet(scene, camera, packet_index, &packet, &translucent_shader);
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
                let ndc_y = (Vec4::ONE - pixel_y * rcp_camera_height) * 2.0 - Vec4::ONE;

                let clip_pos = Vec3x4::new(ndc_x, ndc_y, Vec4::ONE);

                // Transform by the transposed inverse view-projection matrix
                let world_dir =
                    Vec3x4::transform_direction_transposed(inv_viewproj_transposed, clip_pos);
                let world_normal = world_dir.normalize();

                // Sample the cubemap
                let cubemap_sample = scene.cubemap.sample_cubemap(world_normal);
                let cubemap_color = Vec3x4::new(
                    cubemap_sample.col(0),
                    cubemap_sample.col(1),
                    cubemap_sample.col(2),
                );

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
    fn rasterize_packet<S: RasterizerShader>(
        &mut self,
        scene: &Scene,
        camera: &RenderCamera,
        packet_index: usize,
        packet: &RasterPacket,
        shader: &S,
    ) {
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
                    let (depth, mask) = self.depth_test(p.x, p.y, z, mask);

                    let shading_params = RasterizerShaderParams {
                        index_in_tile: self.index_from_xy(p.x, p.y),
                        packet_index: packet_index,
                        z,
                        w,
                        bary0,
                        bary1,
                        bary2,
                        mask,
                        depth_from_depth_test: depth,
                        light_dir,
                        packet: packet,
                        material,
                        camera,
                        scene,
                    };

                    // Execute shader
                    shader.shade(shading_params, self);
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

    // Executes shading on the vbuffer
    fn shade_vbuffer(&mut self, scene: &Scene, camera: &RenderCamera) {
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

                    let light_dir = Vec3x4::new(
                        Vec4::splat(scene.light.direction.x),
                        Vec4::splat(scene.light.direction.y),
                        Vec4::splat(scene.light.direction.z),
                    );
                    let packet = &packet;

                    let shading_params = PbrShaderParams {
                        current_color: UVec4::ZERO,
                        bary0,
                        bary1,
                        bary2,
                        mask,
                        light_dir,
                        packet: packet,
                        material,
                        camera,
                        scene,
                    };

                    // Execute shader
                    let color = pbr_shader::<true>(shading_params);

                    // Write color
                    self.write_pixels(screen_x, screen_y, color, mask);

                    // Clear out the lanes we just handled
                    remaining &= !mask;
                }
            }
        }
    }

    // Perform depth test and return final depth and a mask of the pixels that passed depth testing
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

    fn write_pixels(&mut self, x: i32, y: i32, color: UVec4, final_mask: BVec4A) {
        let index = self.index_from_xy(x, y);
        let booleans: [bool; 4] = final_mask.into();
        let mask_bvec4 = BVec4::from(booleans);

        // Select final color
        let current_color = self.color[index];
        let final_color = UVec4::select(mask_bvec4, color, current_color);

        // Write color
        self.color[index] = final_color;
    }

    pub fn index_from_xy(&self, x: i32, y: i32) -> usize {
        debug_assert!(
            (x - self.screen_min.x) % 2 == 0 || (x - self.screen_min.x) % 2 == 1,
            "x within bounds"
        );
        debug_assert!(
            (y - self.screen_min.y) % 2 == 0 || (y - self.screen_min.y) % 2 == 1,
            "y within bounds"
        );

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
