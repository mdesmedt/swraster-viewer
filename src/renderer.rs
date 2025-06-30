use crate::rendercamera::RenderCamera;
use crate::scene::{BoundingSphere, Node, Scene};
use crate::tilerasterizer::TileRasterizer;
use crossbeam_channel::{bounded, Sender};
use glam::{IVec2, Mat3A, Mat4, UVec4, Vec2, Vec3, Vec3A, Vec4};
use rayon::prelude::*;
use std::ops::{Add, Mul};
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
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u32>,
}

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: Vec4,
    pub normal: Vec3,
    pub texcoords: Vec2,
}

#[derive(Copy, Clone)]
pub struct Triangle {
    pub v0: Vertex,
    pub v1: Vertex,
    pub v2: Vertex,
    pub mesh_index: usize,
    pub primitive_index: usize,
}

// Helper struct for the clipper
#[derive(Copy, Clone)]
pub struct ClipVertex {
    pub position: Vec4,
    pub barycentric: Vec3A,
}

impl ClipVertex {
    pub fn new_empty() -> Self {
        Self {
            position: Vec4::ZERO,
            barycentric: Vec3A::ZERO,
        }
    }
}

// A packet containing a triangle sent from the clipper to the rasterizer for a tile
pub struct RasterPacket {
    pub screen_min: IVec2,
    pub screen_max: IVec2,
    pub pos_screen: [Vec2; 3],
    pub z_over_w: [f32; 3],
    pub one_over_w: [f32; 3],
    pub one_over_area_vec: Vec4,
    pub normal_over_w: [Vec3; 3],
    pub uv_over_w: [Vec2; 3],
    pub primitive_index: usize,
    pub mesh_index: usize,
}

pub enum RasterMessage {
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
pub struct TileBinner {
    pub screen_min: IVec2,
    pub screen_max: IVec2,
    pub channel: Sender<RasterMessage>,
}

// Creates both the binner and the rasterizer tiles for specified screen bounds
// The binner contains the sender and the rasterizer contains the receiver for the channel
fn create_screen_tile(screen_min: IVec2, screen_max: IVec2) -> (TileBinner, TileRasterizer) {
    let (triangle_sender, triangle_receiver) = bounded(1024);
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
            scene.root_nodes.par_iter().for_each(|node_index| {
                let node = &scene.nodes[*node_index];
                self.render_node(
                    scene,
                    camera,
                    node,
                    Mat4::IDENTITY,
                    camera.view_project_matrix,
                );
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

    fn render_node(
        &self,
        scene: &Scene,
        camera: &RenderCamera,
        node: &Node,
        parent_transform: Mat4,
        view_project_matrix: Mat4,
    ) {
        // Compute projection matrix
        let model_matrix = parent_transform * node.transform;
        let mvp_matrix = view_project_matrix * model_matrix;

        // Extract the rotation matrix from the model matrix for normal rotation to world space
        let rotation_matrix = Mat3A::from_mat4(model_matrix);

        // Render the node's mesh if it has one
        if let Some(mesh_index) = node.mesh_index {
            self.render_mesh(
                scene,
                camera,
                mesh_index,
                model_matrix,
                mvp_matrix,
                rotation_matrix,
            );
        }

        // Recursively render child nodes in parallel
        node.children.par_iter().for_each(|&child_index| {
            let child = &scene.nodes[child_index];
            self.render_node(scene, camera, child, model_matrix, view_project_matrix);
        });
    }

    fn render_mesh(
        &self,
        scene: &Scene,
        camera: &RenderCamera,
        mesh_index: usize,
        model_matrix: Mat4,
        mvp_matrix: Mat4,
        rotation_matrix: Mat3A,
    ) {
        // Process primitives in parallel
        let mesh = &scene.meshes[mesh_index];
        mesh.primitives
            .par_iter()
            .enumerate()
            .for_each(|(primitive_index, _)| {
                self.render_primitive(
                    scene,
                    camera,
                    mesh_index,
                    primitive_index,
                    model_matrix,
                    mvp_matrix,
                    rotation_matrix,
                );
            });
    }

    // Assembles a primitive and sends it to the clipper
    fn render_primitive(
        &self,
        scene: &Scene,
        camera: &RenderCamera,
        mesh_index: usize,
        primitive_index: usize,
        model_matrix: Mat4,
        mvp_matrix: Mat4,
        rotation_matrix: Mat3A,
    ) {
        let mesh = &scene.meshes[mesh_index];
        let primitive = &mesh.primitives[primitive_index];

        // Convert bounding sphere to world space
        // TODO: Cache world space bounding sphere in the node somehow?
        let center_world = model_matrix * primitive.bounding_sphere.center.extend(1.0);
        // TODO: Does not handle non-uniform scaling
        let radius_world = primitive.bounding_sphere.radius * model_matrix.x_axis.length();
        let bounding_sphere_world = BoundingSphere {
            center: center_world.truncate(),
            radius: radius_world,
        };

        // Frustum culling with bounding sphere
        if !test_sphere_frustum(&bounding_sphere_world, camera) {
            return; // Cull the primitive
        }

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
                let clip0 = mvp_matrix * world0.extend(1.0);
                let clip1 = mvp_matrix * world1.extend(1.0);
                let clip2 = mvp_matrix * world2.extend(1.0);

                // Read normals
                let normal0 = normals[i0];
                let normal1 = normals[i1];
                let normal2 = normals[i2];

                // Rotate normals to world space
                let normal_world0 = rotation_matrix * normal0;
                let normal_world1 = rotation_matrix * normal1;
                let normal_world2 = rotation_matrix * normal2;

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
        #[inline]
        fn is_inside(v: Vec4, plane: Vec4) -> bool {
            plane.dot(v) >= 0.0
        }

        // Intersect edge with plane
        #[inline]
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

        #[inline]
        fn interp<T>(a: T, b: T, c: T, barycentric: Vec3A) -> T
        where
            T: Copy + Add<Output = T> + Mul<f32, Output = T>,
        {
            a * barycentric.x + b * barycentric.y + c * barycentric.z
        }

        for i in 1..poly_len - 1 {
            // Interpolate the vertex attributes
            let triangle = Triangle {
                v0: Vertex {
                    position: poly[0].position,
                    normal: interp(
                        triangle.v0.normal,
                        triangle.v1.normal,
                        triangle.v2.normal,
                        poly[0].barycentric,
                    ),
                    texcoords: interp(
                        triangle.v0.texcoords,
                        triangle.v1.texcoords,
                        triangle.v2.texcoords,
                        poly[0].barycentric,
                    ),
                },
                v1: Vertex {
                    position: poly[i].position,
                    normal: interp(
                        triangle.v0.normal,
                        triangle.v1.normal,
                        triangle.v2.normal,
                        poly[i].barycentric,
                    ),
                    texcoords: interp(
                        triangle.v0.texcoords,
                        triangle.v1.texcoords,
                        triangle.v2.texcoords,
                        poly[i].barycentric,
                    ),
                },
                v2: Vertex {
                    position: poly[i + 1].position,
                    normal: interp(
                        triangle.v0.normal,
                        triangle.v1.normal,
                        triangle.v2.normal,
                        poly[i + 1].barycentric,
                    ),
                    texcoords: interp(
                        triangle.v0.texcoords,
                        triangle.v1.texcoords,
                        triangle.v2.texcoords,
                        poly[i + 1].barycentric,
                    ),
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
