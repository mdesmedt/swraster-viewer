use crate::bumpqueue::{BumpPool, BumpQueue};
use crate::rendercamera::RenderCamera;
use crate::scene::{BoundingSphere, Node, Scene};
use crate::tilerasterizer::*;
use glam::{IVec2, Mat3A, Mat4, UVec4, Vec2, Vec3, Vec4};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
******************************************************************************/

const TILE_SIZE: i32 = 32;

pub struct RenderBuffer<'a> {
    pub width: usize,
    pub height: usize,
    pub pixels: &'a mut [u8],
}

#[derive(Copy, Clone)]
pub struct Vertex {
    pub pos_clip: Vec4,
    pub pos_world: Vec3,
    pub normal: Vec3,
    pub tangent: Vec3,
    pub uv: Vec2,
}

impl Vertex {
    pub fn new_empty() -> Self {
        Self {
            pos_clip: Vec4::ZERO,
            pos_world: Vec3::ZERO,
            normal: Vec3::ZERO,
            tangent: Vec3::ZERO,
            uv: Vec2::ZERO,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Triangle {
    pub vertices: [Vertex; 3],
    pub mesh_index: usize,
    pub primitive_index: usize,
}

// A packet containing a triangle sent from the clipper to the rasterizer for a tile
#[derive(Clone, Copy)]
pub struct RasterPacket {
    pub screen_min_pixels: IVec2,
    pub screen_max_pixels: IVec2,
    pub pos_screen_subpixels: [IVec2; 3],
    pub z_over_w: [f32; 3],
    pub one_over_area: f32,
    pub one_over_w: [f32; 3],
    pub normals: [Vec3; 3],
    pub tangents: [Vec3; 3],
    pub uv_over_w: [Vec2; 3],
    pub pos_world_over_w: [Vec3; 3],
    pub mesh_index: u32,
    pub primitive_index: u32,
    pub avg_z: OrderedFloat<f32>,
    pub du_dv: Vec4,
}

impl<'a> RenderBuffer<'a> {
    pub fn new(width: usize, height: usize, pixels: &'a mut [u8]) -> Self {
        Self {
            width,
            height,
            pixels,
        }
    }

    pub fn clear(&mut self) {
        for pixel in self.pixels.chunks_exact_mut(4) {
            pixel[0] = 0x00; // R
            pixel[1] = 0x00; // G
            pixel[2] = 0x00; // B

            // Assume A is never written to
            //pixel[3] = 0xff; // A
        }
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, channel: usize, color: u8) {
        if x < self.width && y < self.height {
            self.pixels[(y * self.width + x) * 4 + channel] = color;
        }
    }
}

// Creates both the binner and the rasterizer tiles for specified screen bounds
// The binner contains the sender and the rasterizer contains the receiver for the channel
fn create_screen_tile(
    screen_min: IVec2,
    screen_max: IVec2,
    pool: Arc<BumpPool<RasterPacket>>,
) -> TileRasterizer {
    let width = (screen_max.x - screen_min.x) as usize;
    let height = (screen_max.y - screen_min.y) as usize;
    let width_quads = width / 2;
    let height_quads = height / 2;
    TileRasterizer {
        screen_min,
        screen_max,
        packets_opaque: BumpQueue::new(pool.clone()),
        packets_translucent: BumpQueue::new(pool.clone()),
        color: vec![UVec4::ZERO; width_quads * height_quads],
        depth: vec![Vec4::splat(f32::INFINITY); width_quads * height_quads],
        packet_index: vec![UVec4::ZERO; width_quads * height_quads],
        bary0: vec![Vec4::ZERO; width_quads * height_quads],
        bary1: vec![Vec4::ZERO; width_quads * height_quads],
    }
}

#[inline]
fn test_sphere_frustum(sphere: &BoundingSphere, camera: &RenderCamera) -> bool {
    // Transform the sphere to view space
    let center_view = camera.view_matrix * sphere.center.extend(1.0);
    let view_clip_planes = camera.view_clip_planes;
    for plane in view_clip_planes {
        let distance = plane.dot(center_view);
        if distance < -sphere.radius {
            return false;
        }
    }
    true
}

// Renderer which manages clipping, culling, binning and rasterization
pub struct Renderer {
    width: i32,
    #[allow(dead_code)]
    height: i32,
    width_subpixels: i32,
    height_subpixels: i32,
    width_f: f32,
    height_f: f32,
    tiles: Vec<TileRasterizer>,
    nodes_by_distance: Vec<usize>,
    timer_clipbin: Duration,
    timer_rasterizer: Duration,
    last_print_time: Instant,
    frame_count: u32,
}

impl Renderer {
    pub fn new(width: i32, height: i32) -> Self {
        // Compute the number of tiles
        let tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
        let tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
        // Create the packet pool
        let packet_pool = Arc::new(BumpPool::new());
        // Create the tiles
        let mut tiles = vec![];
        for y in 0..tiles_y {
            for x in 0..tiles_x {
                let screen_min = IVec2::new(x * TILE_SIZE, y * TILE_SIZE);
                let screen_max = IVec2::new((x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE);
                let tile = create_screen_tile(screen_min, screen_max, packet_pool.clone());
                tiles.push(tile);
            }
        }
        Self {
            width,
            height,
            width_subpixels: width * SUBPIXEL_SCALE,
            height_subpixels: height * SUBPIXEL_SCALE,
            width_f: width as f32,
            height_f: height as f32,
            tiles,
            nodes_by_distance: Vec::new(),
            timer_clipbin: Duration::ZERO,
            timer_rasterizer: Duration::ZERO,
            last_print_time: Instant::now(),
            frame_count: 0,
        }
    }

    // Main render function for the scene
    pub fn render_scene(&mut self, scene: &Scene, camera: &RenderCamera) {
        // Compute nodes by distance, front to back
        self.compute_nodes_by_distance(scene, camera);

        // Clip and bin primitives in each node in the scene in parallel
        let clipbin_start = Instant::now();
        self.nodes_by_distance.par_iter().for_each(|node_index| {
            let node = &scene.nodes[*node_index];
            self.render_node(scene, camera, node, camera.view_project_matrix);
        });
        self.timer_clipbin += clipbin_start.elapsed();

        // Rasterize each bin
        // One thread per bin avoids having to lock the tile constantly
        let rasterizer_start = Instant::now();
        self.tiles.par_iter_mut().for_each(|tile| {
            tile.render_tile(scene, camera);
        });
        self.timer_rasterizer += rasterizer_start.elapsed();

        let now = Instant::now();
        if now.duration_since(self.last_print_time) >= Duration::from_secs(1) {
            let clipbin_avg = self.timer_clipbin.as_secs_f64() / self.frame_count as f64;
            let rasterizer_avg = self.timer_rasterizer.as_secs_f64() / self.frame_count as f64;
            let clipbin_ms = clipbin_avg * 1000.0;
            let rasterizer_ms = rasterizer_avg * 1000.0;
            println!("Clipping:       {:.2} ms", clipbin_ms);
            println!("Rasterization:  {:.2} ms", rasterizer_ms);
            println!("Total:          {:.2} ms", clipbin_ms + rasterizer_ms);
            println!();
            self.last_print_time = now;
            self.frame_count = 0;
            self.timer_clipbin = Duration::ZERO;
            self.timer_rasterizer = Duration::ZERO;
        }
        self.frame_count += 1;
    }

    // Copy all tiles pixels to the backbuffer
    pub fn blit_to_buffer(&self, buffer: &mut RenderBuffer) {
        let num_tiles_x = (self.width + TILE_SIZE - 1) / TILE_SIZE;
        buffer
            .pixels
            .par_chunks_mut(buffer.width * 4 * 2) // 2 rows of quads
            .enumerate()
            .for_each(|(quad_y_index, buffer_rows)| {
                let quad_y = quad_y_index as i32;
                let pixel_y = quad_y * 2;
                let tile_y = pixel_y / TILE_SIZE;

                // Iterate through all tiles in this row
                for tile_x in 0..num_tiles_x {
                    let tile_index = tile_y * num_tiles_x + tile_x;
                    let tile = &self.tiles[tile_index as usize];

                    // Calculate the local y coordinate within this tile
                    let local_y_pixels = pixel_y - tile.screen_min.y;
                    let local_y_quad = local_y_pixels / 2;

                    let tile_width_pixels = tile.screen_max.x - tile.screen_min.x;
                    let tile_width_quads = tile_width_pixels / 2;

                    // Copy quads from the tile to the buffer
                    let src_index_base = local_y_quad * tile_width_quads;
                    for quad_x in 0..tile_width_quads {
                        let src_index = src_index_base + quad_x;
                        let base_pixel_x = tile.screen_min.x + quad_x as i32 * 2;

                        // Copy each element of the quad to individual pixels
                        let colors = tile.color[src_index as usize].to_array();
                        for sub_y in 0..2 {
                            for sub_x in 0..2 {
                                // Write the 4 pixels in the X dimension
                                let pixel_x = base_pixel_x + sub_x;
                                if pixel_x < self.width {
                                    // Read the u32 packed color
                                    let rgba = colors[(sub_y * 2 + sub_x) as usize];

                                    // Unpack the components
                                    let r: u8 = (rgba >> 0 & 0xFF) as u8;
                                    let g: u8 = (rgba >> 8 & 0xFF) as u8;
                                    let b: u8 = (rgba >> 16 & 0xFF) as u8;

                                    // Write the components to the backbuffer
                                    let dst_offset = (pixel_x + sub_y * self.width) as usize * 4;
                                    // TODO: Fix this being backwards after switching to pixels
                                    buffer_rows[dst_offset] = b;
                                    buffer_rows[dst_offset + 1] = g;
                                    buffer_rows[dst_offset + 2] = r;

                                    // Assume A is never written to
                                    //buffer_row[offset + 3] = 0xFF;
                                }
                            }
                        }
                    }
                }
            });
    }

    fn compute_nodes_by_distance(&mut self, scene: &Scene, camera: &RenderCamera) {
        self.nodes_by_distance.clear();
        self.nodes_by_distance
            .extend(scene.nodes.iter().enumerate().map(|(i, _)| i));
        self.nodes_by_distance.sort_by_key(|&i| {
            let node = &scene.nodes[i];
            let bounding_sphere = &node.bounding_sphere_world;
            let distance = (camera.position - bounding_sphere.center).length_squared();
            OrderedFloat(distance)
        });
    }

    fn render_node(
        &self,
        scene: &Scene,
        camera: &RenderCamera,
        node: &Node,
        view_project_matrix: Mat4,
    ) {
        // Compute projection matrix
        let model_matrix = node.transform;
        let mvp_matrix = view_project_matrix * model_matrix;

        // Render the node's mesh if it has one
        if let Some(mesh_index) = node.mesh_index {
            self.render_mesh(scene, camera, mesh_index, model_matrix, mvp_matrix);
        }
    }

    fn render_mesh(
        &self,
        scene: &Scene,
        camera: &RenderCamera,
        mesh_index: usize,
        model_matrix: Mat4,
        mvp_matrix: Mat4,
    ) {
        // Render opaque primitives
        let mesh = &scene.meshes[mesh_index];
        mesh.primitives_opaque
            .par_iter()
            .for_each(|primitive_index| {
                self.render_primitive::<true>(
                    scene,
                    camera,
                    mesh_index,
                    *primitive_index,
                    model_matrix,
                    mvp_matrix,
                );
            });

        // Render translucent primitives
        mesh.primitives_translucent
            .par_iter()
            .for_each(|primitive_index| {
                self.render_primitive::<false>(
                    scene,
                    camera,
                    mesh_index,
                    *primitive_index,
                    model_matrix,
                    mvp_matrix,
                );
            });
    }

    // Assembles a primitive and sends it to the clipper
    fn render_primitive<const MODE_OPAQUE: bool>(
        &self,
        scene: &Scene,
        camera: &RenderCamera,
        mesh_index: usize,
        primitive_index: usize,
        model_matrix: Mat4,
        mvp_matrix: Mat4,
    ) {
        let mesh = &scene.meshes[mesh_index];
        let primitive = &mesh.primitives[primitive_index];

        // Convert bounding sphere to world space
        // TODO: Cache world space bounding sphere in the node somehow?
        let bounding_sphere_world = model_matrix * &primitive.bounding_sphere;

        // Frustum culling with bounding sphere
        if !test_sphere_frustum(&bounding_sphere_world, camera) {
            return; // Cull the primitive
        }

        let positions = &primitive.positions;
        let normals = &primitive.normals;
        let tangents = &primitive.tangents;
        let texcoords = &primitive.texcoords;
        let indices = &primitive.indices;

        // Compute normal to world rotation matrix (assume uniform scaling)
        let rotation_matrix = Mat3A::from_mat4(model_matrix);

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
                let local0 = positions[i0];
                let local1 = positions[i1];
                let local2 = positions[i2];

                // Compute world space positions
                let world0 = model_matrix * local0.extend(1.0);
                let world1 = model_matrix * local1.extend(1.0);
                let world2 = model_matrix * local2.extend(1.0);

                // Project vertices to clip space
                let clip0 = mvp_matrix * local0.extend(1.0);
                let clip1 = mvp_matrix * local1.extend(1.0);
                let clip2 = mvp_matrix * local2.extend(1.0);

                // Rotate normals to world space (assume uniform scaling)
                let normal_world0 = rotation_matrix * normals[i0];
                let normal_world1 = rotation_matrix * normals[i1];
                let normal_world2 = rotation_matrix * normals[i2];

                // Rotate tangents to world space (assume uniform scaling)
                let tangent_world0 = rotation_matrix * tangents[i0];
                let tangent_world1 = rotation_matrix * tangents[i1];
                let tangent_world2 = rotation_matrix * tangents[i2];

                let triangle = Triangle {
                    vertices: [
                        Vertex {
                            pos_clip: clip0,
                            pos_world: world0.truncate(),
                            normal: normal_world0,
                            tangent: tangent_world0,
                            uv: texcoords[i0],
                        },
                        Vertex {
                            pos_clip: clip1,
                            pos_world: world1.truncate(),
                            normal: normal_world1,
                            tangent: tangent_world1,
                            uv: texcoords[i1],
                        },
                        Vertex {
                            pos_clip: clip2,
                            pos_world: world2.truncate(),
                            normal: normal_world2,
                            tangent: tangent_world2,
                            uv: texcoords[i2],
                        },
                    ],
                    mesh_index,
                    primitive_index,
                };

                // Send the triangle to the clipper
                self.clip_against_frustum::<MODE_OPAQUE>(triangle);
            }
        });
    }

    // Clips a triangle against all 6 frustum planes in clip space
    // TODO: Can we get away with only clipping against the near and far planes?
    // TODO: Not robust because it does not do vertex deduplication/welding, but it is fast-ish.
    fn clip_against_frustum<const MODE_OPAQUE: bool>(&self, triangle: Triangle) {
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
        #[inline]
        fn is_inside(v: Vec4, plane: Vec4) -> bool {
            plane.dot(v) >= 0.0
        }

        // Intersect edge with plane
        #[inline]
        fn intersect(v0: Vertex, v1: Vertex, plane: Vec4) -> Vertex {
            let d0 = plane.dot(v0.pos_clip);
            let d1 = plane.dot(v1.pos_clip);
            let t = d0 / (d0 - d1);
            Vertex {
                pos_clip: v0.pos_clip + (v1.pos_clip - v0.pos_clip) * t,
                pos_world: v0.pos_world + (v1.pos_world - v0.pos_world) * t,
                normal: v0.normal + (v1.normal - v0.normal) * t,
                tangent: v0.tangent + (v1.tangent - v0.tangent) * t,
                uv: v0.uv + (v1.uv - v0.uv) * t,
            }
        }

        // Use a fixed-size buffer (assumed bound: 16 vertices) for the polygon.
        let mut poly: [Vertex; 16] = [Vertex::new_empty(); 16];
        let mut poly_len: usize = 3; // Start with the original triangle (3 vertices)
        poly[0] = triangle.vertices[0];
        poly[1] = triangle.vertices[1];
        poly[2] = triangle.vertices[2];

        // Use a second fixed buffer (new_poly) for the "new" polygon.
        let mut new_poly: [Vertex; 16] = [Vertex::new_empty(); 16];

        for &plane in &PLANES {
            if poly_len == 0 {
                return;
            }
            let mut new_len: usize = 0;
            for i in 0..poly_len {
                let curr = poly[i];
                let prev = poly[(i + poly_len - 1) % poly_len];
                let curr_in = is_inside(curr.pos_clip, plane);
                let prev_in = is_inside(prev.pos_clip, plane);
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
            let triangle = Triangle {
                vertices: [poly[0], poly[i], poly[i + 1]],
                mesh_index: triangle.mesh_index,
                primitive_index: triangle.primitive_index,
            };

            // Bin the triangle
            self.bin_triangle::<MODE_OPAQUE>(triangle);
        }
    }

    // Projects a triangle to screen space and bins it into tiles
    fn bin_triangle<const MODE_OPAQUE: bool>(&self, triangle: Triangle) {
        // Compute the screen space bounding box of the triangle
        let p0 = self.clip_to_screen_subpixels(triangle.vertices[0].pos_clip);
        let p1 = self.clip_to_screen_subpixels(triangle.vertices[1].pos_clip);
        let p2 = self.clip_to_screen_subpixels(triangle.vertices[2].pos_clip);

        // NOTE: At this point triangles are CW front-facing because of screen coordinate conversion
        // TODO: Just figure out how to keep CCW facing triangles in screen space and deal with it

        // Compute signed area
        let signed_area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);

        // Backface culling
        if signed_area > 0 {
            return;
        }

        // Set up the values which the rasterizer will need

        let one_over_w = [
            1.0 / triangle.vertices[0].pos_clip.w,
            1.0 / triangle.vertices[1].pos_clip.w,
            1.0 / triangle.vertices[2].pos_clip.w,
        ];

        let one_over_area = 1.0 / (signed_area.abs() as f32);

        let z_over_w = [
            triangle.vertices[0].pos_clip.z * one_over_w[0],
            triangle.vertices[1].pos_clip.z * one_over_w[1],
            triangle.vertices[2].pos_clip.z * one_over_w[2],
        ];

        let normals = [
            triangle.vertices[0].normal,
            triangle.vertices[1].normal,
            triangle.vertices[2].normal,
        ];
        let tangents = [
            triangle.vertices[0].tangent,
            triangle.vertices[1].tangent,
            triangle.vertices[2].tangent,
        ];

        let uv_over_w = [
            triangle.vertices[0].uv * one_over_w[0],
            triangle.vertices[1].uv * one_over_w[1],
            triangle.vertices[2].uv * one_over_w[2],
        ];

        let pos_world_over_w = [
            triangle.vertices[0].pos_world * one_over_w[0],
            triangle.vertices[1].pos_world * one_over_w[1],
            triangle.vertices[2].pos_world * one_over_w[2],
        ];

        // Compute uv derivatives for texture mapping during shading
        let dx1 = (p1.x - p0.x) as f32;
        let dx2 = (p2.x - p0.x) as f32;
        let dy1 = (p1.y - p0.y) as f32;
        let dy2 = (p2.y - p0.y) as f32;
        let uv0 = uv_over_w[0];
        let uv1 = uv_over_w[1];
        let uv2 = uv_over_w[2];
        let du1 = uv1.x - uv0.x;
        let du2 = uv2.x - uv0.x;
        let dv1 = uv1.y - uv0.y;
        let dv2 = uv2.y - uv0.y;
        let du_dx = du1 * dy2 - du2 * dy1;
        let du_dy = du2 * dx1 - du1 * dx2;
        let dv_dx = dv1 * dy2 - dv2 * dy1;
        let dv_dy = dv2 * dx1 - dv1 * dx2;
        let du_dv = Vec4::new(du_dx, du_dy, dv_dx, dv_dy) * Vec4::splat(one_over_area);

        // Compute triangle bounding box
        let screen_min_subpixels: IVec2 = IVec2::new(
            p0.x.min(p1.x).min(p2.x).max(0),
            p0.y.min(p1.y).min(p2.y).max(0),
        );
        let screen_max_subpixels: IVec2 = IVec2::new(
            p0.x.max(p1.x).max(p2.x).min(self.width_subpixels),
            p0.y.max(p1.y).max(p2.y).min(self.height_subpixels),
        );
        let screen_min_pixels = screen_min_subpixels >> SUBPIXEL_SHIFT;
        let screen_max_pixels = (screen_max_subpixels + SUBPIXEL_SCALE - 1) >> SUBPIXEL_SHIFT;

        // Calculate which bins intersect with the triangle's bounding box
        let min_bin_x = screen_min_pixels.x / TILE_SIZE;
        let min_bin_y = screen_min_pixels.y / TILE_SIZE;
        let max_bin_x = (screen_max_pixels.x + TILE_SIZE - 1) / TILE_SIZE;
        let max_bin_y = (screen_max_pixels.y + TILE_SIZE - 1) / TILE_SIZE;

        // Calculate number of bins in x direction
        let bins_x = (self.width + TILE_SIZE - 1) / TILE_SIZE;

        // Compute average z for translucency sorting later
        let avg_z = if !MODE_OPAQUE {
            OrderedFloat(
                (triangle.vertices[0].pos_clip.z
                    + triangle.vertices[1].pos_clip.z
                    + triangle.vertices[2].pos_clip.z)
                    / 3.0,
            )
        } else {
            OrderedFloat(0.0)
        };

        // Send the packet to all intersecting bins
        for y in min_bin_y..max_bin_y {
            for x in min_bin_x..max_bin_x {
                let bin_index = y * bins_x + x;
                if bin_index >= 0 && bin_index < self.tiles.len() as i32 {
                    let tile = &self.tiles[bin_index as usize];

                    // Clip the triangle bounds to this tile's bounds
                    let tile_clipped_min = IVec2::new(
                        screen_min_pixels.x.max(tile.screen_min.x),
                        screen_min_pixels.y.max(tile.screen_min.y),
                    );
                    let tile_clipped_max = IVec2::new(
                        screen_max_pixels.x.min(tile.screen_max.x),
                        screen_max_pixels.y.min(tile.screen_max.y),
                    );

                    // Create a RasterPacket clipped to the tile
                    let packet = RasterPacket {
                        screen_min_pixels: tile_clipped_min,
                        screen_max_pixels: tile_clipped_max,
                        pos_screen_subpixels: [p0, p1, p2],
                        z_over_w: z_over_w,
                        one_over_w: one_over_w,
                        normals: normals,
                        tangents: tangents,
                        uv_over_w: uv_over_w,
                        pos_world_over_w: pos_world_over_w,
                        primitive_index: triangle.primitive_index as u32,
                        mesh_index: triangle.mesh_index as u32,
                        one_over_area: one_over_area,
                        avg_z: avg_z,
                        du_dv: du_dv,
                    };

                    if MODE_OPAQUE {
                        tile.packets_opaque.push(packet);
                    } else {
                        tile.packets_translucent.push(packet);
                    }
                }
            }
        }
    }

    // Computes the 2D screen positions for a clip space vertex
    fn clip_to_screen_subpixels(&self, vertex: Vec4) -> IVec2 {
        // Perform perspective divide
        let ndc = vertex / vertex.w;

        // Convert to screen coordinates
        let screen_x = (ndc.x + 1.0) * self.width_f / 2.0;
        let screen_y = (1.0 - ndc.y) * self.height_f / 2.0;

        IVec2::new(
            (screen_x * SUBPIXEL_SCALE_F).round() as i32,
            (screen_y * SUBPIXEL_SCALE_F).round() as i32,
        )
    }
}
