use crate::bumpqueue::BumpQueue;
use crate::math::*;
use crate::rendercamera::RenderCamera;
use crate::renderer::RasterPacket;
use crate::scene::{Material, Scene};
use crate::shader::*;
use crate::util::*;
use glam::{BVec4A, IVec2, UVec4, Vec4};

pub const SUBPIXEL_SCALE: i32 = 16;
pub const SUBPIXEL_SCALE_F: f32 = 16.0;
pub const SUBPIXEL_SHIFT: i32 = 4;

// Step deltas for 4-pixel SIMD stepping
const STEP_X_SIZE: i32 = 2 * SUBPIXEL_SCALE;
const STEP_Y_SIZE: i32 = 2 * SUBPIXEL_SCALE;

const HALF_PIXEL: i32 = SUBPIXEL_SCALE / 2;
const ONE_HALF_PIXEL: i32 = SUBPIXEL_SCALE + HALF_PIXEL;

const COARSE_BLOCK_SIZE_PIXELS: i32 = 16;
const COARSE_BLOCK_SIZE_SUBPIXELS: i32 = COARSE_BLOCK_SIZE_PIXELS * SUBPIXEL_SCALE;

// The tile which the rasterizer uses to consume work
pub struct TileRasterizer {
    pub screen_min: IVec2,
    pub screen_max: IVec2,
    pub packets_opaque: BumpQueue<RasterPacket>,
    pub packets_translucent: BumpQueue<RasterPacket>,
    // Color and depth stored as 2x2 pixel quads
    pub color: Vec<Vec3x4>,
    pub depth: Vec<Vec4>,
    // Internal "visibility-buffer" as 2x2 pixel quads
    pub packet_index: Vec<UVec4>,
    pub bary1: Vec<Vec4>,
    pub bary2: Vec<Vec4>,
    pub center_luminance: f32,
}

struct RasterParams<'a> {
    packet: &'a RasterPacket,
    packet_index: usize,
    scene: &'a Scene,
    camera: &'a RenderCamera,
    material: &'a Material,
    a01: Vec4,
    b01: Vec4,
    c01: Vec4,
    a12: Vec4,
    b12: Vec4,
    c12: Vec4,
    a20: Vec4,
    b20: Vec4,
    c20: Vec4,
    step_x01: Vec4,
    step_x12: Vec4,
    step_x20: Vec4,
    step_y01: Vec4,
    step_y12: Vec4,
    step_y20: Vec4,
    one_over_area_vec: Vec4,
}

struct FineRasterParams {
    x_start_subpixels: i32,
    y_start_subpixels: i32,
    x_end_subpixels: i32,
    y_end_subpixels: i32,
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
        if self.packets_opaque.len() == 0 {
            // Fast path for skybox only tiles
            self.shade_vbuffer::<true>(scene, camera);
        } else {
            // Normal v-buffer shading
            self.shade_vbuffer::<false>(scene, camera);
        }

        // Sort translucent packets by z, back to front
        self.packets_translucent
            .sort_by(|a, b| b.avg_z.cmp(&a.avg_z));

        // Render translucent packets
        let translucent_shader = TranslucentForwardShader;
        for packet_index in 0..self.packets_translucent.len() {
            let packet = self.packets_translucent.get(packet_index);
            self.rasterize_packet(scene, camera, packet_index, &packet, &translucent_shader);
        }

        // Compute center luminance for auto exposure metering
        let center_quad = self.color[self.color.len() / 2];
        let luminance_x = center_quad.x * 0.2126 + center_quad.y * 0.7152 + center_quad.z * 0.0722;
        self.center_luminance = (luminance_x.x + luminance_x.y + luminance_x.z + luminance_x.w) * 0.25;

        // Reset the queues
        self.packets_opaque.reset();
        self.packets_translucent.reset();
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
        // Get triangle vertices in screen space
        let p0 = packet.pos_screen_subpixels[0];
        let p1 = packet.pos_screen_subpixels[1];
        let p2 = packet.pos_screen_subpixels[2];

        let one_over_area_vec = Vec4::splat(packet.one_over_area);

        // Fetch the material
        let mesh_index = packet.mesh_index as usize;
        let primitive_index = packet.primitive_index as usize;
        let mesh = &scene.meshes[mesh_index];
        let material_index = mesh.primitives[primitive_index].material_index;
        let material = &scene.materials[material_index];

        // Set up edge function coefficients for edges v0->v1, v1->v2, v2->v0

        // Helper function to apply top-left bias
        fn top_left_bias(a: i32, b: i32) -> i32 {
            if a < 0 || (a == 0 && b > 0) { 0 } else { -1 }
        }

        // Edge v0->v1: (y1-y0)x + (x0-x1)y + (x1*y0 - x0*y1) = 0
        let a01 = p1.y - p0.y;
        let b01 = p0.x - p1.x;
        let c01 = p1.x * p0.y - p0.x * p1.y + top_left_bias(a01, b01);

        // Edge v1->v2: (y2-y1)x + (x1-x2)y + (x2*y1 - x1*y2) = 0
        let a12 = p2.y - p1.y;
        let b12 = p1.x - p2.x;
        let c12 = p2.x * p1.y - p1.x * p2.y + top_left_bias(a12, b12);

        // Edge v2->v0: (y0-y2)x + (x2-x0)y + (x0*y2 - x2*y0) = 0
        let a20 = p0.y - p2.y;
        let b20 = p2.x - p0.x;
        let c20 = p0.x * p2.y - p2.x * p0.y + top_left_bias(a20, b20);

        let step_x01 = Vec4::splat((a01 * STEP_X_SIZE) as f32);
        let step_x12 = Vec4::splat((a12 * STEP_X_SIZE) as f32);
        let step_x20 = Vec4::splat((a20 * STEP_X_SIZE) as f32);

        let step_y01 = Vec4::splat((b01 * STEP_Y_SIZE) as f32);
        let step_y12 = Vec4::splat((b12 * STEP_Y_SIZE) as f32);
        let step_y20 = Vec4::splat((b20 * STEP_Y_SIZE) as f32);

        // Create raster params shared between coarse and fine raster
        let raster_params = RasterParams {
            packet,
            packet_index,
            scene,
            camera,
            material,
            a01: Vec4::splat(a01 as f32),
            b01: Vec4::splat(b01 as f32),
            c01: Vec4::splat(c01 as f32),
            a12: Vec4::splat(a12 as f32),
            b12: Vec4::splat(b12 as f32),
            c12: Vec4::splat(c12 as f32),
            a20: Vec4::splat(a20 as f32),
            b20: Vec4::splat(b20 as f32),
            c20: Vec4::splat(c20 as f32),
            step_x01,
            step_x12,
            step_x20,
            step_y01,
            step_y12,
            step_y20,
            one_over_area_vec,
        };

        // Adaptively check whether to coarse raster
        let width_pixels = packet.screen_max_pixels.x - packet.screen_min_pixels.x;
        let height_pixels = packet.screen_max_pixels.y - packet.screen_min_pixels.y;
        if width_pixels > COARSE_BLOCK_SIZE_PIXELS || height_pixels > COARSE_BLOCK_SIZE_PIXELS {
            // Coarse raster first then fine raster
            self.coarse_raster(raster_params, shader);
        } else {
            // Fine raster smaller triangles immediately

            // Rasterize position aligned to quads
            let x_start_pixels = packet.screen_min_pixels.x & !1;
            let y_start_pixels = packet.screen_min_pixels.y & !1;

            // Compute subpixel positions
            let x_start_subpixels = x_start_pixels * SUBPIXEL_SCALE;
            let y_start_subpixels = y_start_pixels * SUBPIXEL_SCALE;
            let x_end_subpixels = packet.screen_max_pixels.x * SUBPIXEL_SCALE;
            let y_end_subpixels = packet.screen_max_pixels.y * SUBPIXEL_SCALE;

            let fine_raster_params = FineRasterParams {
                x_start_subpixels: x_start_subpixels,
                y_start_subpixels: y_start_subpixels,
                x_end_subpixels: x_end_subpixels,
                y_end_subpixels: y_end_subpixels,
            };
            self.fine_raster(&raster_params, fine_raster_params, shader);
        }
    }

    fn coarse_raster<S: RasterizerShader>(&mut self, raster_params: RasterParams, shader: &S) {
        let packet = raster_params.packet;

        // Rasterize position aligned to quads
        let x_start_pixels = packet.screen_min_pixels.x & !1;
        let y_start_pixels = packet.screen_min_pixels.y & !1;

        // Compute subpixel positions
        let x_start_subpixels = x_start_pixels * SUBPIXEL_SCALE;
        let y_start_subpixels = y_start_pixels * SUBPIXEL_SCALE;
        let x_end_subpixels = packet.screen_max_pixels.x * SUBPIXEL_SCALE;
        let y_end_subpixels = packet.screen_max_pixels.y * SUBPIXEL_SCALE;

        // Coarse raster the triangle in blocks
        let mut block_y = y_start_subpixels;
        while block_y < y_end_subpixels {
            let mut block_x = x_start_subpixels;
            while block_x < x_end_subpixels {
                // Build block corner positions
                let cx = Vec4::new(
                    (block_x + HALF_PIXEL) as f32,
                    (block_x + COARSE_BLOCK_SIZE_SUBPIXELS - HALF_PIXEL) as f32,
                    (block_x + HALF_PIXEL) as f32,
                    (block_x + COARSE_BLOCK_SIZE_SUBPIXELS - HALF_PIXEL) as f32,
                );
                let cy = Vec4::new(
                    (block_y + HALF_PIXEL) as f32,
                    (block_y + HALF_PIXEL) as f32,
                    (block_y + COARSE_BLOCK_SIZE_SUBPIXELS - HALF_PIXEL) as f32,
                    (block_y + COARSE_BLOCK_SIZE_SUBPIXELS - HALF_PIXEL) as f32,
                );

                // Evaluate edge functions at block corners
                let w0_corners =
                    raster_params.a12 * cx + raster_params.b12 * cy + raster_params.c12;
                let w1_corners =
                    raster_params.a20 * cx + raster_params.b20 * cy + raster_params.c20;
                let w2_corners =
                    raster_params.a01 * cx + raster_params.b01 * cy + raster_params.c01;

                // If any edge max < 0 the whole block lies outside the triangle
                let w0_max = w0_corners.max_element();
                let w1_max = w1_corners.max_element();
                let w2_max = w2_corners.max_element();
                let fully_outside = (w0_max < 0.0) || (w1_max < 0.0) || (w2_max < 0.0);
                if fully_outside {
                    // Skip this block
                    block_x += COARSE_BLOCK_SIZE_SUBPIXELS;
                    continue;
                }

                // Fine raster the block
                let fine_raster_params = FineRasterParams {
                    x_start_subpixels: block_x,
                    y_start_subpixels: block_y,
                    x_end_subpixels: i32::min(
                        block_x + COARSE_BLOCK_SIZE_SUBPIXELS,
                        x_end_subpixels,
                    ),
                    y_end_subpixels: i32::min(
                        block_y + COARSE_BLOCK_SIZE_SUBPIXELS,
                        y_end_subpixels,
                    ),
                };

                self.fine_raster(&raster_params, fine_raster_params, shader);

                block_x += COARSE_BLOCK_SIZE_SUBPIXELS;
            }
            block_y += COARSE_BLOCK_SIZE_SUBPIXELS;
        }
    }

    fn fine_raster<S: RasterizerShader>(
        &mut self,
        raster_params: &RasterParams,
        fine_raster_params: FineRasterParams,
        shader: &S,
    ) {
        let x_start_subpixels = fine_raster_params.x_start_subpixels;
        let y_start_subpixels = fine_raster_params.y_start_subpixels;
        let x_end_subpixels = fine_raster_params.x_end_subpixels;
        let y_end_subpixels = fine_raster_params.y_end_subpixels;

        let mut p = IVec2::new(x_start_subpixels, y_start_subpixels);

        // Initial edge function values for the quad at the starting pixel with half-pixel offset
        let x = Vec4::new(
            (p.x + HALF_PIXEL) as f32,
            (p.x + ONE_HALF_PIXEL) as f32,
            (p.x + HALF_PIXEL) as f32,
            (p.x + ONE_HALF_PIXEL) as f32,
        );
        let y = Vec4::new(
            (p.y + HALF_PIXEL) as f32,
            (p.y + HALF_PIXEL) as f32,
            (p.y + ONE_HALF_PIXEL) as f32,
            (p.y + ONE_HALF_PIXEL) as f32,
        );

        // Edge function values: w0 = edge v1->v2, w1 = edge v2->v0, w2 = edge v0->v1
        let mut w0_row = raster_params.a12 * x + raster_params.b12 * y + raster_params.c12;
        let mut w1_row = raster_params.a20 * x + raster_params.b20 * y + raster_params.c20;
        let mut w2_row = raster_params.a01 * x + raster_params.b01 * y + raster_params.c01;

        // Main fine raster loop
        while p.y < y_end_subpixels {
            let mut w0 = w0_row;
            let mut w1 = w1_row;
            let mut w2 = w2_row;

            p.x = x_start_subpixels;
            while p.x < x_end_subpixels {
                // Test if any pixels are inside the triangle
                let mask = w0.cmpge(Vec4::ZERO) & w1.cmpge(Vec4::ZERO) & w2.cmpge(Vec4::ZERO);
                if mask.any() {
                    // Compute the barycentrics
                    let bary1: Vec4 = w1 * raster_params.one_over_area_vec;
                    let bary2: Vec4 = w2 * raster_params.one_over_area_vec;

                    // Interpolate w and z
                    let w = 1.0 / raster_params.packet.one_over_w.interpolate(bary1, bary2);
                    let z = raster_params.packet.z_over_w.interpolate(bary1, bary2) * w;

                    // Compute pixel coordinate from subpixel
                    let pixel = p >> SUBPIXEL_SHIFT;

                    // Perform depth test
                    let (depth, mask) = self.depth_test(pixel, z, mask);
                    if mask.any() {
                        let shading_params = RasterizerShaderParams {
                            index_in_tile: self.index_from_xy(pixel),
                            packet_index: raster_params.packet_index,
                            z,
                            w,
                            bary1,
                            bary2,
                            mask,
                            depth_from_depth_test: depth,
                            packet: raster_params.packet,
                            material: raster_params.material,
                            camera: raster_params.camera,
                            scene: raster_params.scene,
                        };

                        // Execute shader
                        shader.shade(shading_params, self);
                    }
                }

                // Step in X
                w0 += raster_params.step_x12;
                w1 += raster_params.step_x20;
                w2 += raster_params.step_x01;
                p.x += STEP_X_SIZE;
            }

            // Step in Y
            w0_row += raster_params.step_y12;
            w1_row += raster_params.step_y20;
            w2_row += raster_params.step_y01;
            p.y += STEP_Y_SIZE;
        }
    }

    // Executes shading on the vbuffer
    fn shade_vbuffer<const SKYBOX_ONLY: bool>(&mut self, scene: &Scene, camera: &RenderCamera) {
        let tile_width = (self.screen_max.x - self.screen_min.x) as usize;
        let tile_height = (self.screen_max.y - self.screen_min.y) as usize;
        let width_quads = tile_width / 2;

        // Loop over rows of quads
        for y in 0..tile_height / 2 {
            let screen_y = self.screen_min.y + (y as i32) * 2;

            // Loop over each quad in the row
            for quad_x in 0..width_quads {
                let screen_x = self.screen_min.x + (quad_x as i32 * 2);
                let pixel = IVec2::new(screen_x, screen_y);
                let index = self.index_from_xy(pixel);

                // Fast path for skybox only tiles

                if SKYBOX_ONLY {
                    self.color[index] = self.compute_skybox(scene, camera, pixel);
                    continue;
                }

                // Normal v-buffer shading

                let packet_index_vec = self.packet_index[index];

                let depth = self.depth[index];
                let depth_mask = depth.cmpne(Vec4::INFINITY);

                let mut out_color = Vec3x4::ZERO;

                // Check if there are any pixels in this quad
                if depth_mask.any() {
                    // Consume all possible packets in packet_index uniformly
                    let mut remaining = depth_mask;
                    while remaining.any() {
                        // Find the first active lane (smallest i where remaining[i] == true)
                        let lane = remaining.bitmask().trailing_zeros() as usize;

                        // Extract that lane's value
                        let packet_index = packet_index_vec[lane];

                        // Build mask for all lanes equal to this value
                        let mask_bvec4 = packet_index_vec.cmpeq(UVec4::splat(packet_index));
                        let mask = bvec4_to_bvec4a(mask_bvec4);

                        let bary1 = self.bary1[index];
                        let bary2 = self.bary2[index];
                        let packet = self.packets_opaque.get(packet_index as usize);

                        // Fetch the material
                        let mesh_index = packet.mesh_index as usize;
                        let primitive_index = packet.primitive_index as usize;
                        let mesh = &scene.meshes[mesh_index];
                        let material_index = mesh.primitives[primitive_index].material_index;
                        let material = &scene.materials[material_index];

                        let packet = &packet;

                        let shading_params = PbrShaderParams {
                            current_color: Vec3x4::ZERO,
                            bary1,
                            bary2,
                            packet,
                            material,
                            camera,
                            scene,
                        };

                        // Execute shader
                        let color = pbr_shader::<false>(shading_params);

                        // Store fragments
                        out_color = Vec3x4::select(mask, color, out_color);

                        // Clear out the lanes we just handled
                        remaining &= !mask;
                    }
                }

                // Check if there are any skybox pixels in this quad
                if !depth_mask.all() {
                    let skybox_color = self.compute_skybox(scene, camera, pixel);
                    out_color = Vec3x4::select(depth_mask, out_color, skybox_color);
                }

                // Write color to the tile's color buffer
                self.color[index] = out_color
            }
        }
    }

    fn compute_skybox(&self, scene: &Scene, camera: &RenderCamera, pixel: IVec2) -> Vec3x4 {
        let screen_x = pixel.x;
        let screen_y = pixel.y;

        // Render skybox
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

        let ndc_x = pixel_x * camera.one_over_width * 2.0 - Vec4::ONE;
        let ndc_y = (Vec4::ONE - pixel_y * camera.one_over_height) * 2.0 - Vec4::ONE;

        let clip_pos = Vec3x4::new(ndc_x, ndc_y, Vec4::ONE);

        // Transform by the transposed inverse view-projection matrix
        let world_dir =
            Vec3x4::transform_direction_transposed(camera.skybox_matrix_transposed, clip_pos);
        let world_normal = world_dir.normalize();

        // Sample the cubemap
        srgb_to_linear_fast(scene.cubemap.sample_cubemap_rgb(world_normal, UVec4::ZERO)) * 0.75
    }

    // Perform depth test and return final depth and a mask of the pixels that passed depth testing
    fn depth_test(&mut self, pixel: IVec2, depth: Vec4, mask: BVec4A) -> (Vec4, BVec4A) {
        let index = self.index_from_xy(pixel);
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

    pub fn index_from_xy(&self, pixel: IVec2) -> usize {
        debug_assert!(
            pixel.x >= 0 && pixel.x < self.screen_max.x,
            "x within bounds"
        );
        debug_assert!(
            pixel.y >= 0 && pixel.y < self.screen_max.y,
            "y within bounds"
        );

        let local = pixel - self.screen_min;
        let quad = local / 2;

        let width = self.screen_max.x - self.screen_min.x;
        let width_quads = width / 2;

        let index = quad.y * width_quads + quad.x;
        index as usize
    }
}
