use crate::rendercamera::RenderCamera;
use crate::scene::{Material, Node, Scene};
use crossbeam_channel::{unbounded, Receiver, Sender};
use glam::{BVec4, BVec4A, IVec2, Mat3A, Mat4, UVec4, Vec2, Vec3, Vec3A, Vec4};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/******************************************************************************
 * This is the main code of this project. A simple software rasterizer.
 * It renders a Scene with a RenderCamera to a RenderBuffer.
 *
 * Overview of the render path:
 *   1. Traverse nodes/meshes/primitives
 *   2. Clip triangles against the camera frustum
 *   3. Bin triangles into tiles
 *   4. The tile rasterizer rasterizes the triangles within its tile and updates its buffer
 *   5. At the end of rendering every tile gets copied to the final buffer
 *
 * The renderer is multithreaded and uses SIMD to process 4 pixels (in X) at a time.
 * Materials and textures are not supported yet.
******************************************************************************/

pub struct RenderBuffer {
    width: usize,
    height: usize,
    pixels: Vec<u32>,
}

#[derive(Copy, Clone)]
struct Vertex {
    position: Vec4,
    normal: Vec3,
    texcoords: Vec2,
}

#[derive(Copy, Clone)]
struct Triangle {
    v0: Vertex,
    v1: Vertex,
    v2: Vertex,
    mesh_index: usize,
    primitive_index: usize,
}

// Helper struct for the clipper
#[derive(Copy, Clone)]
struct ClipVertex {
    position: Vec4,
    barycentric: Vec3A,
}

impl ClipVertex {
    fn new_empty() -> Self {
        Self {
            position: Vec4::ZERO,
            barycentric: Vec3A::ZERO,
        }
    }
}

// A packet containing a triangle sent from the clipper to the rasterizer for a tile
struct RasterPacket {
    screen_min: IVec2,
    screen_max: IVec2,
    pos_screen: [Vec2; 3],
    z_over_w: [f32; 3],
    one_over_w: [f32; 3],
    one_over_area_vec: Vec4,
    normal_over_w: [Vec3; 3],
    uv_over_w: [Vec2; 3],
    primitive_index: usize,
    mesh_index: usize,
}

enum RasterMessage {
    Packet(RasterPacket),
    Terminate,
}

impl RenderBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; width * height],
        }
    }

    pub fn pixels(&self) -> &[u32] {
        &self.pixels
    }

    pub fn clear(&mut self) {
        self.pixels.fill(0); // Clear to black
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, color: u32) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = color;
        }
    }
}

// The tile which the binner uses to send work
// This is a separate struct from the TileRaster to make ownership easier
struct TileBinner {
    screen_min: IVec2,
    screen_max: IVec2,
    channel: Sender<RasterMessage>,
}

// The tile which the rasterizer uses to consume work
// NOTE: The color and depth values are stored using SIMD vectors
struct TileRasterizer {
    screen_min: IVec2,
    screen_max: IVec2,
    channel: Receiver<RasterMessage>,
    color: Vec<UVec4>,
    depth: Vec<Vec4>,
}

// Creates both the binner and the rasterizer tiles for specified screen bounds
// The binner contains the sender and the rasterizer contains the receiver for the channel
fn create_screen_tile(screen_min: IVec2, screen_max: IVec2) -> (TileBinner, TileRasterizer) {
    let (triangle_sender, triangle_receiver) = unbounded();
    let width = (screen_max.x - screen_min.x) as usize;
    let height = (screen_max.y - screen_min.y) as usize;
    let width_vec4 = (width + 3) / 4;
    (
        TileBinner {
            screen_min,
            screen_max,
            channel: triangle_sender,
        },
        TileRasterizer {
            screen_min,
            screen_max,
            channel: triangle_receiver,
            color: vec![UVec4::ZERO; width_vec4 * height],
            depth: vec![Vec4::splat(f32::INFINITY); width_vec4 * height],
        },
    )
}

impl TileRasterizer {
    // Copy the tile to the render buffer
    fn copy_to_buffer(&self, buffer: &mut RenderBuffer) {
        let width = (self.screen_max.x - self.screen_min.x) as usize;
        let height = (self.screen_max.y - self.screen_min.y) as usize;
        let width_vec4 = (width + 3) / 4;

        for y in 0..height {
            for x_vec4 in 0..width_vec4 {
                let src_index = y * width_vec4 + x_vec4;
                let dst_y = self.screen_min.y + y as i32;

                if dst_y >= 0 && dst_y < buffer.height as i32 {
                    let colors = self.color[src_index].to_array();
                    let dst_base_x = self.screen_min.x + x_vec4 as i32 * 4;

                    // Copy each component of the Vec4 to individual pixels
                    for i in 0..4 {
                        let pixel_x = dst_base_x + i as i32;
                        if pixel_x >= 0 && pixel_x < buffer.width as i32 {
                            let dst_index = dst_y as usize * buffer.width + pixel_x as usize;
                            buffer.pixels[dst_index] = colors[i as usize];
                        }
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
    fn begin_rasterization(&mut self, scene: &Scene) {
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
        let length = Vec4::new(
            length_squared.x.sqrt(),
            length_squared.y.sqrt(),
            length_squared.z.sqrt(),
            length_squared.w.sqrt(),
        );
        let normalized_x = normal_x / length;
        let normalized_y = normal_y / length;
        let normalized_z = normal_z / length;

        // Apply N.L lighting
        let dot_x = normalized_x * light_x;
        let dot_y = normalized_y * light_y;
        let dot_z = normalized_z * light_z;
        let mut light_intensity = dot_x + dot_y + dot_z;

        // TODO: Add ambient light
        light_intensity += Vec4::splat(0.1);

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
                color_r *= diffuse_vec[0];
                color_g *= diffuse_vec[1];
                color_b *= diffuse_vec[2];
            }
        }
        else {
            // TODO: Handle missing material
        }

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

        self.write_pixels(p.x, p.y, packed_colors, z, mask);
    }

    // Perform depth test and write color and depth
    fn write_pixels(&mut self, x: i32, y: i32, color: UVec4, depth: Vec4, mask: BVec4A) {
        debug_assert_eq!(x % 4, 0, "x must be a multiple of 4 for SIMD alignment");
        let local_y = y - self.screen_min.y;
        let width = (self.screen_max.x - self.screen_min.x) as usize;
        let width_vec4 = (width + 3) / 4;

        if local_y >= 0 && local_y < (self.screen_max.y - self.screen_min.y) as i32 {
            // x is the leftmost pixel of the 4-pixel SIMD block
            let local_x = x - self.screen_min.x;
            let vec4_index = local_x as usize / 4;

            if vec4_index < width_vec4 {
                let index = local_y as usize * width_vec4 + vec4_index;
                // Load current depth values
                let current_depth = self.depth[index];
                // Perform depth test
                let mask_depth = depth.cmple(current_depth);
                let mask_combined = mask & mask_depth;
                // TODO: This is a little annoying, now we have to convert to a BVec4 to use the select function for UVec4
                let booleans: [bool; 4] = mask_combined.into();
                let mask_combined_bvec4 = BVec4::from(booleans);
                if mask_combined.any() {
                    let current_color = self.color[index];
                    let new_color = UVec4::select(mask_combined_bvec4, color, current_color);
                    let new_depth = Vec4::select(mask_combined, depth, current_depth);
                    self.color[index] = new_color;
                    self.depth[index] = new_depth;
                }
            }
        }
    }
}

// Renderer which manages clipping, culling, binning and rasterization
pub struct Renderer {
    width: i32,
    height: i32,
    tile_width: i32,
    tile_height: i32,
    tiles_binner: Vec<TileBinner>,
    tiles_rasterizer: Vec<Arc<Mutex<TileRasterizer>>>,
}

impl Renderer {
    pub fn new(width: i32, height: i32) -> Self {
        // Compute the number of tiles
        let tile_width = 128;
        let tile_height = 128;
        let tiles_x = (width + tile_width - 1) / tile_width;
        let tiles_y = (height + tile_height - 1) / tile_height;
        // Create the tiles
        let mut tiles_binner = vec![];
        let mut tiles_rasterizer = vec![];
        for y in 0..tiles_y {
            for x in 0..tiles_x {
                let screen_min = IVec2::new(x * tile_width, y * tile_height);
                let screen_max = IVec2::new((x + 1) * tile_width, (y + 1) * tile_height);
                let (binner, rasterizer) = create_screen_tile(screen_min, screen_max);
                tiles_binner.push(binner);
                tiles_rasterizer.push(Arc::new(Mutex::new(rasterizer)));
            }
        }
        Self {
            width,
            height,
            tile_width,
            tile_height,
            tiles_binner: tiles_binner,
            tiles_rasterizer,
        }
    }

    // Main render function for the scene
    pub fn render_scene(
        &mut self,
        scene: &Scene,
        camera: &RenderCamera,
        buffer: &mut RenderBuffer,
    ) {
        // Set up matrices
        let view_matrix = camera.view_matrix();
        let projection_matrix = camera.projection_matrix();
        let view_project_matrix = projection_matrix * view_matrix;

        // Using a thread scope here to ensure that the rasterizer threads are joined
        std::thread::scope(|scope| {
            // Kick off the rasterizer threads which start listening to their channels
            for rasterizer in &self.tiles_rasterizer {
                let rasterizer_clone = rasterizer.clone();
                scope.spawn(move || {
                    let mut rasterizer_mutable = rasterizer_clone.lock().unwrap();
                    rasterizer_mutable.begin_rasterization(scene);
                });
            }

            // Clip and bin primitives in each node in the scene in parallel
            scene.nodes.par_iter().for_each(|node| {
                self.render_node(scene, node, view_project_matrix);
            });

            // Send the termination message to all channels
            for binner in &self.tiles_binner {
                binner.channel.send(RasterMessage::Terminate).ok();
            }
        });

        // Copy all tile pixels to the main buffer serially
        // TODO: Multithread this by slicing the buffer somehow?
        for tile in &self.tiles_rasterizer {
            let rasterizer_guard = tile.lock().unwrap();
            rasterizer_guard.copy_to_buffer(buffer);
        }
    }

    fn render_node(&self, scene: &Scene, node: &Node, view_project_matrix: Mat4) {
        // Compute projection matrix
        let model_matrix = node.transform;
        let mvp_matrix = view_project_matrix * model_matrix;

        // Extract the rotation matrix from the model matrix for normal rotation to world space
        let rotation_matrix = Mat3A::from_mat4(model_matrix);

        // Render the node's mesh if it has one
        if let Some(mesh_index) = node.mesh_index {
            self.render_mesh(scene, mesh_index, mvp_matrix, rotation_matrix);
        }

        // Recursively render child nodes in parallel
        node.children.par_iter().for_each(|&child_index| {
            let child = &scene.nodes[child_index];
            self.render_node(scene, child, mvp_matrix);
        });
    }

    fn render_mesh(
        &self,
        scene: &Scene,
        mesh_index: usize,
        mvp_matrix: Mat4,
        rotation_matrix: Mat3A,
    ) {
        // Process primitives in parallel
        let mesh = &scene.meshes[mesh_index];
        mesh.primitives
            .par_iter()
            .enumerate()
            .for_each(|(primitive_index, _)| {
                self.assemble_primitive(
                    scene,
                    mesh_index,
                    primitive_index,
                    mvp_matrix,
                    rotation_matrix,
                );
            });
    }

    // Assembles a primitive and sends it to the clipper
    fn assemble_primitive(
        &self,
        scene: &Scene,
        mesh_index: usize,
        primitive_index: usize,
        mvp_matrix: Mat4,
        model_matrix: Mat3A,
    ) {
        let mesh = &scene.meshes[mesh_index];
        let primitive = &mesh.primitives[primitive_index];

        let positions = &primitive.positions;
        let normals = &primitive.normals;
        let texcoords = &primitive.texcoords;
        let indices = &primitive.indices;

        // Process triangles in batches of 128 triangles
        const TRIANGLES_PER_BATCH: usize = 128;
        const INDICES_PER_BATCH: usize = TRIANGLES_PER_BATCH * 3;
        indices.par_chunks(INDICES_PER_BATCH).for_each(|batch| {
            for chunk in batch.chunks(3) {
                if chunk.len() != 3 {
                    continue;
                }

                // Read indices
                let i0 = chunk[0] as usize;
                let i1 = chunk[1] as usize;
                let i2 = chunk[2] as usize;

                // Read vertices
                let world0 = positions[i0];
                let world1 = positions[i1];
                let world2 = positions[i2];

                // Project vertices to clip space
                let clip0 = mvp_matrix * Vec4::new(world0[0], world0[1], world0[2], 1.0);
                let clip1 = mvp_matrix * Vec4::new(world1[0], world1[1], world1[2], 1.0);
                let clip2 = mvp_matrix * Vec4::new(world2[0], world2[1], world2[2], 1.0);

                // Read normals
                let normal0 = normals[i0];
                let normal1 = normals[i1];
                let normal2 = normals[i2];

                // Rotate normals to world space
                let normal_world0 = model_matrix * normal0;
                let normal_world1 = model_matrix * normal1;
                let normal_world2 = model_matrix * normal2;

                let triangle = Triangle {
                    v0: Vertex {
                        position: clip0,
                        normal: normal_world0,
                        texcoords: texcoords[i0],
                    },
                    v1: Vertex {
                        position: clip1,
                        normal: normal_world1,
                        texcoords: texcoords[i1],
                    },
                    v2: Vertex {
                        position: clip2,
                        normal: normal_world2,
                        texcoords: texcoords[i2],
                    },
                    mesh_index,
                    primitive_index,
                };

                // Send the triangle to the clipper
                self.clip_against_frustum(triangle);
            }
        });
    }

    // Clips a triangle against all 6 frustum planes in clip space
    // TODO: Can we get away with only clipping against the near and far planes?
    // TODO: Not robust because it does not do vertex deduplication/welding, but it is fast-ish.
    fn clip_against_frustum(&self, triangle: Triangle) {
        // Frustum planes in clip space
        const PLANES: [Vec4; 6] = [
            Vec4::new(0.0, 0.0, 1.0, 1.0),  // Near:  z + w >= 0
            Vec4::new(0.0, 0.0, -1.0, 1.0), // Far:  -z + w >= 0
            Vec4::new(1.0, 0.0, 0.0, 1.0),  // Right: x + w >= 0
            Vec4::new(-1.0, 0.0, 0.0, 1.0), // Left: -x + w >= 0
            Vec4::new(0.0, 1.0, 0.0, 1.0),  // Top:  y + w >= 0
            Vec4::new(0.0, -1.0, 0.0, 1.0), // Bottom: -y + w >= 0
        ];

        // Test if a vertex is inside a plane
        fn is_inside(v: Vec4, plane: Vec4) -> bool {
            plane.dot(v) >= 0.0
        }

        // Intersect edge with plane
        fn intersect(v0: ClipVertex, v1: ClipVertex, plane: Vec4) -> ClipVertex {
            let d0 = plane.dot(v0.position);
            let d1 = plane.dot(v1.position);
            let t = d0 / (d0 - d1);
            ClipVertex {
                position: v0.position + (v1.position - v0.position) * t,
                barycentric: v0.barycentric + (v1.barycentric - v0.barycentric) * t,
            }
        }

        // Use a fixed-size buffer (assumed bound: 16 vertices) for the polygon.
        let mut poly: [ClipVertex; 16] = [ClipVertex::new_empty(); 16];
        let mut poly_len: usize = 3; // Start with the original triangle (3 vertices)
        poly[0] = ClipVertex {
            position: triangle.v0.position,
            barycentric: Vec3A::new(1.0, 0.0, 0.0),
        };
        poly[1] = ClipVertex {
            position: triangle.v1.position,
            barycentric: Vec3A::new(0.0, 1.0, 0.0),
        };
        poly[2] = ClipVertex {
            position: triangle.v2.position,
            barycentric: Vec3A::new(0.0, 0.0, 1.0),
        };

        // Use a second fixed buffer (new_poly) for the "new" polygon.
        let mut new_poly: [ClipVertex; 16] = [ClipVertex::new_empty(); 16];

        for &plane in &PLANES {
            if poly_len == 0 {
                return;
            }
            let mut new_len: usize = 0;
            for i in 0..poly_len {
                let curr = poly[i];
                let prev = poly[(i + poly_len - 1) % poly_len];
                let curr_in = is_inside(curr.position, plane);
                let prev_in = is_inside(prev.position, plane);
                if curr_in {
                    if !prev_in {
                        new_poly[new_len] = intersect(prev, curr, plane);
                        new_len += 1;
                    }
                    new_poly[new_len] = curr;
                    new_len += 1;
                } else if prev_in {
                    new_poly[new_len] = intersect(prev, curr, plane);
                    new_len += 1;
                }
            }
            // Copy only the valid elements from new_poly to poly
            for i in 0..new_len {
                poly[i] = new_poly[i];
            }
            poly_len = new_len;
        }

        // Triangulate the resulting polygon (using a fan) if it has at least 3 vertices.
        if poly_len < 3 {
            return;
        }
        for i in 1..poly_len - 1 {
            // Interpolate the vertex attributes
            // TODO: Clean up this code
            let triangle = Triangle {
                v0: Vertex {
                    position: poly[0].position,
                    normal: poly[0].barycentric.x * triangle.v0.normal
                        + poly[0].barycentric.y * triangle.v1.normal
                        + poly[0].barycentric.z * triangle.v2.normal,
                    texcoords: poly[0].barycentric.x * triangle.v0.texcoords
                        + poly[0].barycentric.y * triangle.v1.texcoords
                        + poly[0].barycentric.z * triangle.v2.texcoords,
                },
                v1: Vertex {
                    position: poly[i].position,
                    normal: poly[i].barycentric.x * triangle.v0.normal
                        + poly[i].barycentric.y * triangle.v1.normal
                        + poly[i].barycentric.z * triangle.v2.normal,
                    texcoords: poly[i].barycentric.x * triangle.v0.texcoords
                        + poly[i].barycentric.y * triangle.v1.texcoords
                        + poly[i].barycentric.z * triangle.v2.texcoords,
                },
                v2: Vertex {
                    position: poly[i + 1].position,
                    normal: poly[i + 1].barycentric.x * triangle.v0.normal
                        + poly[i + 1].barycentric.y * triangle.v1.normal
                        + poly[i + 1].barycentric.z * triangle.v2.normal,
                    texcoords: poly[i + 1].barycentric.x * triangle.v0.texcoords
                        + poly[i + 1].barycentric.y * triangle.v1.texcoords
                        + poly[i + 1].barycentric.z * triangle.v2.texcoords,
                },
                mesh_index: triangle.mesh_index,
                primitive_index: triangle.primitive_index,
            };

            // Bin the triangle
            self.bin_triangle(triangle);
        }
    }

    // Projects a triangle to screen space and bins it into tiles
    fn bin_triangle(&self, triangle: Triangle) {
        // Compute the screen space bounding box of the triangle
        let p0 = self.clip_to_screen(triangle.v0.position);
        let p1 = self.clip_to_screen(triangle.v1.position);
        let p2 = self.clip_to_screen(triangle.v2.position);

        // NOTE: At this point triangles are CW front-facing because of screen coordinate conversion

        // Compute signed area
        let signed_area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);

        // Backface culling
        if signed_area > 0.0 {
            return;
        }

        // Set up the values which the rasterizer will need

        let w0 = triangle.v0.position.w;
        let w1 = triangle.v1.position.w;
        let w2 = triangle.v2.position.w;

        let one_over_w = [1.0 / w0, 1.0 / w1, 1.0 / w2];

        let z_over_w = [
            triangle.v0.position.z / w0,
            triangle.v1.position.z / w1,
            triangle.v2.position.z / w2,
        ];

        let normal_over_w = [
            triangle.v0.normal / w0,
            triangle.v1.normal / w1,
            triangle.v2.normal / w2,
        ];

        let uv_over_w = [
            triangle.v0.texcoords / w0,
            triangle.v1.texcoords / w1,
            triangle.v2.texcoords / w2,
        ];

        // Compute triangle bounding box
        let screen_min: IVec2 = IVec2::new(
            p0.x.min(p1.x).min(p2.x).max(0.0).floor() as i32,
            p0.y.min(p1.y).min(p2.y).max(0.0).floor() as i32,
        );
        let screen_max: IVec2 = IVec2::new(
            p0.x.max(p1.x).max(p2.x).min(self.width as f32).ceil() as i32,
            p0.y.max(p1.y).max(p2.y).min(self.height as f32).ceil() as i32,
        );

        // Calculate which bins intersect with the triangle's bounding box
        let min_bin_x = screen_min.x / self.tile_width;
        let min_bin_y = screen_min.y / self.tile_height;
        let max_bin_x = (screen_max.x + self.tile_width - 1) / self.tile_width;
        let max_bin_y = (screen_max.y + self.tile_height - 1) / self.tile_height;

        // Calculate number of bins in x direction
        let bins_x = (self.width + self.tile_width - 1) / self.tile_width;

        // Send the packet to all intersecting bins
        for y in min_bin_y..max_bin_y {
            for x in min_bin_x..max_bin_x {
                let bin_index = y * bins_x + x;
                if bin_index >= 0 && bin_index < self.tiles_binner.len() as i32 {
                    let binner = &self.tiles_binner[bin_index as usize];

                    // Clip the triangle bounds to this tile's bounds
                    let tile_clipped_min = IVec2::new(
                        screen_min.x.max(binner.screen_min.x),
                        screen_min.y.max(binner.screen_min.y),
                    );
                    let tile_clipped_max = IVec2::new(
                        screen_max.x.min(binner.screen_max.x),
                        screen_max.y.min(binner.screen_max.y),
                    );

                    // Only send if there's actually an intersection
                    if tile_clipped_min.x < tile_clipped_max.x
                        && tile_clipped_min.y < tile_clipped_max.y
                    {
                        // Create a RasterPacket clipped to the tile
                        let packet = RasterPacket {
                            screen_min: tile_clipped_min,
                            screen_max: tile_clipped_max,
                            pos_screen: [p0, p1, p2],
                            z_over_w: z_over_w,
                            one_over_w: one_over_w,
                            normal_over_w: normal_over_w,
                            uv_over_w: uv_over_w,
                            primitive_index: triangle.primitive_index,
                            mesh_index: triangle.mesh_index,
                            one_over_area_vec: Vec4::splat(1.0 / signed_area as f32),
                        };
                        binner.channel.send(RasterMessage::Packet(packet)).ok();
                    }
                }
            }
        }
    }

    // Computes the 2D screen positions for a clip space vertex
    fn clip_to_screen(&self, vertex: Vec4) -> Vec2 {
        // Perform perspective divide
        let ndc = Vec2::new(vertex.x / vertex.w, vertex.y / vertex.w);

        // Convert to screen coordinates
        let screen_x = (ndc.x + 1.0) * self.width as f32 / 2.0;
        let screen_y = (1.0 - ndc.y) * self.height as f32 / 2.0;

        Vec2::new(screen_x, screen_y)
    }
}
