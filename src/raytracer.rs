use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use bvh::ray::Ray;
use glam::{Vec2, Vec3A};
use nalgebra::{Point3, Vector3};

use crate::scene::Scene;

const EPSILON: f32 = 1.0e-5;

#[derive(Clone)]
struct BvhTriangle {
    index: usize,
    aabb: Aabb<f32, 3>,
    node_index: usize,
}

impl Bounded<f32, 3> for BvhTriangle {
    fn aabb(&self) -> Aabb<f32, 3> {
        self.aabb
    }
}

impl BHShape<f32, 3> for BvhTriangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

#[derive(Clone)]
pub struct Triangle {
    pub p0: Vec3A,
    pub p1: Vec3A,
    pub p2: Vec3A,
    pub n: Vec3A,
    pub uv0: Vec2,
    pub uv1: Vec2,
    pub uv2: Vec2,
    pub material_index: usize,
}

#[derive(Clone, Copy)]
pub struct Hit {
    pub triangle_index: usize,
    pub bary_u: f32,
    pub bary_v: f32,
    pub bary_w: f32,
    pub position: Vec3A,
    pub normal: Vec3A,
    pub material_index: usize,
}

impl Hit {
    pub fn uv(&self, triangle: &Triangle) -> Vec2 {
        triangle.uv0 * self.bary_w + triangle.uv1 * self.bary_u + triangle.uv2 * self.bary_v
    }
}

pub struct RayTracer {
    triangles: Vec<Triangle>,
    bvh_shapes: Vec<BvhTriangle>,
    bvh: Bvh<f32, 3>,
}

impl RayTracer {
    pub fn new(scene: &Scene) -> Self {
        let mut triangles = Vec::new();
        let mut bvh_shapes = Vec::new();

        for node in &scene.nodes {
            let Some(mesh_index) = node.mesh_index else {
                continue;
            };
            let mesh = &scene.meshes[mesh_index];
            for primitive in &mesh.primitives {
                let indices = &primitive.indices;
                for tri in (0..indices.len()).step_by(3) {
                    let i0 = indices[tri] as usize;
                    let i1 = indices[tri + 1] as usize;
                    let i2 = indices[tri + 2] as usize;

                    let p0 = Vec3A::from_vec4(node.transform * primitive.positions[i0]);
                    let p1 = Vec3A::from_vec4(node.transform * primitive.positions[i1]);
                    let p2 = Vec3A::from_vec4(node.transform * primitive.positions[i2]);

                    let face_n = (p1 - p0).cross(p2 - p0);
                    if face_n.length_squared() <= 1.0e-12 {
                        continue;
                    }
                    let n = face_n.normalize();

                    let uv0 = primitive.texcoords[i0];
                    let uv1 = primitive.texcoords[i1];
                    let uv2 = primitive.texcoords[i2];

                    let min = p0.min(p1).min(p2) - Vec3A::splat(EPSILON);
                    let max = p0.max(p1).max(p2) + Vec3A::splat(EPSILON);

                    let tri_index = triangles.len();
                    triangles.push(Triangle {
                        p0,
                        p1,
                        p2,
                        n,
                        uv0,
                        uv1,
                        uv2,
                        material_index: primitive.material_index,
                    });
                    bvh_shapes.push(BvhTriangle {
                        index: tri_index,
                        aabb: Aabb::with_bounds(
                            Point3::new(min.x, min.y, min.z),
                            Point3::new(max.x, max.y, max.z),
                        ),
                        node_index: 0,
                    });
                }
            }
        }

        let bvh = Bvh::build(&mut bvh_shapes);
        Self {
            triangles,
            bvh_shapes,
            bvh,
        }
    }

    pub fn trace_nearest(
        &self,
        origin: Vec3A,
        direction: Vec3A,
        t_min: f32,
        t_max: f32,
    ) -> Option<Hit> {
        let ray = make_bvh_ray(origin, direction);
        let candidates = self.bvh.traverse(&ray, &self.bvh_shapes);

        let mut best_t = t_max;
        let mut best_hit = None;

        for shape in candidates {
            let tri = &self.triangles[shape.index];
            let Some((t, u, v)) = ray_triangle_intersect(origin, direction, tri, t_min, best_t)
            else {
                continue;
            };

            let w = 1.0 - u - v;
            let position = origin + direction * t;
            best_t = t;
            best_hit = Some(Hit {
                triangle_index: shape.index,
                bary_u: u,
                bary_v: v,
                bary_w: w,
                position,
                normal: tri.n,
                material_index: tri.material_index,
            });
        }

        best_hit
    }

    pub fn trace_transmittance(
        &self,
        origin: Vec3A,
        direction: Vec3A,
        scene: &Scene,
        t_min: f32,
        t_max: f32,
    ) -> f32 {
        let ray = make_bvh_ray(origin, direction);
        let candidates = self.bvh.traverse(&ray, &self.bvh_shapes);

        let mut hits = Vec::new();
        for shape in candidates {
            let tri = &self.triangles[shape.index];
            let Some((t, _, _)) = ray_triangle_intersect(origin, direction, tri, t_min, t_max)
            else {
                continue;
            };
            hits.push((t, tri.material_index));
        }

        hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut transmittance = 1.0;
        for (_, material_index) in hits {
            let material = &scene.materials[material_index];
            if !material.is_translucent {
                return 0.0;
            }
            transmittance *= material.transmission;
            if transmittance <= 0.0001 {
                return 0.0;
            }
        }
        transmittance
    }

    pub fn triangle(&self, index: usize) -> &Triangle {
        &self.triangles[index]
    }

    pub fn triangles(&self) -> &[Triangle] {
        &self.triangles
    }
}

fn make_bvh_ray(origin: Vec3A, direction: Vec3A) -> Ray<f32, 3> {
    Ray::new(
        Point3::new(origin.x, origin.y, origin.z),
        Vector3::new(direction.x, direction.y, direction.z),
    )
}

fn ray_triangle_intersect(
    origin: Vec3A,
    direction: Vec3A,
    tri: &Triangle,
    t_min: f32,
    t_max: f32,
) -> Option<(f32, f32, f32)> {
    let edge1 = tri.p1 - tri.p0;
    let edge2 = tri.p2 - tri.p0;
    let pvec = direction.cross(edge2);
    let det = edge1.dot(pvec);

    if det.abs() <= 1.0e-8 {
        return None;
    }

    let inv_det = 1.0 / det;
    let tvec = origin - tri.p0;
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let qvec = tvec.cross(edge1);
    let v = direction.dot(qvec) * inv_det;
    if v < 0.0 || (u + v) > 1.0 {
        return None;
    }

    let t = edge2.dot(qvec) * inv_det;
    if t < t_min || t > t_max {
        return None;
    }

    Some((t, u, v))
}
