use crate::scene::{Node, Primitive, Scene};
use glam::Vec3;
use parry3d::{
    math::{Isometry, Point, Vector},
    query::Ray,
    shape::{SharedShape, TriMesh},
};

/// A raytracer that converts scene data to parry3d meshes and provides ray intersection testing
pub struct RayTracer {
    shapes: Vec<(SharedShape, Isometry<f32>)>,
}

impl RayTracer {
    /// Create a new raytracer from a scene
    pub fn new(scene: &Scene) -> Self {
        let mut shapes = Vec::new();

        // Convert each node with a mesh to parry3d shapes
        for node in &scene.nodes {
            if let Some(mesh_index) = node.mesh_index {
                let mesh = &scene.meshes[mesh_index];

                for primitive in &mesh.primitives {
                    let trimesh = Self::primitive_to_trimesh(node, primitive);
                    let shape = SharedShape::new(trimesh);
                    shapes.push((shape, Isometry::identity()));
                }
            }
        }

        // TODO: Add acceleration structure

        Self { shapes }
    }

    /// Convert a primitive to a parry3d TriMesh
    fn primitive_to_trimesh(node: &Node, primitive: &Primitive) -> TriMesh {
        let mut vertices = Vec::with_capacity(primitive.positions.len());
        let mut indices = Vec::with_capacity(primitive.indices.len());

        for position in &primitive.positions {
            let pos_world = node.transform.transform_point3(*position);
            vertices.push(Point::new(pos_world.x, pos_world.y, pos_world.z));
        }

        for i in (0..primitive.indices.len()).step_by(3) {
            if i + 2 < primitive.indices.len() {
                indices.push([
                    primitive.indices[i],
                    primitive.indices[i + 1],
                    primitive.indices[i + 2],
                ]);
            }
        }

        TriMesh::new(vertices, indices).expect("Failed to create TriMesh")
    }

    /// Test if a ray intersects with the scene
    pub fn ray_intersect(&self, origin: Vec3, direction: Vec3) -> bool {
        let dir = Vector::new(direction.x, direction.y, direction.z);
        let ray = Ray::new(Point::new(origin.x, origin.y, origin.z), dir);

        for (shape, isometry) in &self.shapes {
            if let Some(intersection) =
                shape.cast_ray_and_get_normal(isometry, &ray, f32::INFINITY, false)
            {
                // Accept only front-facing intersections
                if intersection.normal.dot(&dir) < 0.0 {
                    return true;
                }
            }
        }
        false
    }
}
