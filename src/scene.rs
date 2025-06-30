use glam::{Mat4, Quat, Vec2, Vec3, Vec3A, Vec4};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

use crate::texture::*;

#[derive(Debug)]
pub enum SceneError {
    MissingData(String),
    InvalidData(String),
    ConversionError(String),
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SceneError::MissingData(msg) => write!(f, "Missing data: {}", msg),
            SceneError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            SceneError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
        }
    }
}

impl Error for SceneError {}
pub type SceneResult<T> = Result<T, SceneError>;

pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f32,
}

pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub nodes: Vec<Node>,
    pub materials: Vec<Material>,
    pub cameras: Vec<SceneCamera>,
    pub root_nodes: Vec<usize>, // Indices into nodes array
}

pub struct Node {
    pub name: Option<String>,
    pub transform: Mat4, // Model matrix
    pub mesh_index: Option<usize>,
    pub camera_index: Option<usize>,
    pub children: Vec<usize>, // Indices into nodes array
}

pub struct Mesh {
    pub name: Option<String>,
    pub primitives: Vec<Primitive>,
    pub light: Light,
}

pub struct Primitive {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub texcoords: Vec<Vec2>,
    pub indices: Vec<u32>,
    pub material_index: Option<usize>,
    pub bounding_sphere: BoundingSphere,
}

pub struct Material {
    pub name: Option<String>,
    pub base_color_factor: Vec4,
    pub base_color_texture: Option<TextureAndSampler>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<TextureAndSampler>,
    pub normal_texture: Option<TextureAndSampler>,
    pub emissive_factor: Vec3,
    pub emissive_texture: Option<TextureAndSampler>,
    pub occlusion_texture: Option<TextureAndSampler>,
    pub is_alpha_tested: bool,
    pub alpha_cutoff_vec: Vec4,
}

pub struct SceneCamera {
    pub name: Option<String>,
    pub transform: Mat4, // View matrix
    pub projection: Projection,
}

pub enum Projection {
    Perspective {
        fov: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    },
    Orthographic {
        xmag: f32,
        ymag: f32,
        near: f32,
        far: f32,
    },
}

impl Scene {
    pub fn from_gltf(
        document: &gltf::Document,
        _gltf_scene: &gltf::Scene, // TODO: Implement GLTF scene logic
        buffers: &[gltf::buffer::Data],
        texture_cache: &mut TextureCache,
    ) -> SceneResult<Self> {
        let mut scene = Scene {
            meshes: Vec::new(),
            nodes: Vec::new(),
            materials: Vec::new(),
            cameras: Vec::new(),
            root_nodes: Vec::new(),
        };

        // Pre-allocate vectors with capacity hints
        let mesh_count = document.meshes().len();
        let material_count = document.materials().len();
        let camera_count = document.cameras().len();
        let node_count = document.nodes().len();

        scene.meshes.reserve(mesh_count);
        scene.materials.reserve(material_count);
        scene.cameras.reserve(camera_count);
        scene.nodes.reserve(node_count);

        // Collect meshes
        for mesh in document.meshes() {
            scene.meshes.push(Mesh::from_gltf(&mesh, buffers)?);
        }

        // Collect materials using the provided texture cache
        for material in document.materials() {
            scene.materials.push(Material::from_gltf(
                &material,
                document,
                buffers,
                texture_cache,
            )?);
        }

        // Collect cameras
        for camera in document.cameras() {
            scene.cameras.push(SceneCamera::from_gltf(&camera)?);
        }

        // Build node hierarchy
        let mut node_indices = HashMap::with_capacity(node_count);

        // First collect all nodes
        for node in document.nodes() {
            let node_index = scene.nodes.len();
            node_indices.insert(node.index(), node_index);
            scene.nodes.push(Node::from_gltf(&node)?);
        }

        // Then set up parent-child relationships
        for node in document.nodes() {
            if let Some(node_index) = node_indices.get(&node.index()) {
                let children: Vec<usize> = node
                    .children()
                    .filter_map(|child| node_indices.get(&child.index()).copied())
                    .collect();
                scene.nodes[*node_index].children = children;
            }
        }

        // Set root nodes (nodes that are not children of any other node)
        let child_indices: HashSet<_> = document
            .nodes()
            .flat_map(|node| node.children().map(|child| child.index()))
            .collect();

        scene.root_nodes = document
            .nodes()
            .filter(|node| !child_indices.contains(&node.index()))
            .filter_map(|node| node_indices.get(&node.index()).copied())
            .collect();

        Ok(scene)
    }
}

impl Node {
    fn from_gltf(node: &gltf::Node) -> SceneResult<Self> {
        let transform: Mat4 = match node.transform() {
            gltf::scene::Transform::Matrix { matrix } => {
                // Convert [[f32; 4]; 4] to Mat4
                Mat4::from_cols_array_2d(&matrix)
            }
            gltf::scene::Transform::Decomposed {
                translation,
                rotation,
                scale,
            } => {
                let translation: Vec3 = Vec3::new(translation[0], translation[1], translation[2]);
                let rotation: Quat =
                    Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
                let scale: Vec3 = Vec3::new(scale[0], scale[1], scale[2]);

                let translation_matrix: Mat4 = Mat4::from_translation(translation);
                let rotation_matrix: Mat4 = Mat4::from_quat(rotation);
                let scale_matrix: Mat4 = Mat4::from_scale(scale);

                translation_matrix * rotation_matrix * scale_matrix
            }
        };

        Ok(Node {
            name: node.name().map(String::from),
            transform,
            mesh_index: node.mesh().map(|m| m.index()),
            camera_index: node.camera().map(|c| c.index()),
            children: Vec::new(), // Will be set up later
        })
    }
}

impl Mesh {
    fn from_gltf(mesh: &gltf::Mesh, buffers: &[gltf::buffer::Data]) -> SceneResult<Self> {
        let mut primitives = Vec::with_capacity(mesh.primitives().len());

        for primitive in mesh.primitives() {
            primitives.push(Primitive::from_gltf(&primitive, buffers)?);
        }

        // Create a default per-mesh light for now
        let light_normal = Vec3::new(0.1, 0.5, 0.5).normalize();
        let light_color = Vec3::new(1.0, 1.0, 1.0);

        Ok(Mesh {
            name: mesh.name().map(String::from),
            primitives,
            light: Light {
                normal: light_normal,
                color: light_color,
            },
        })
    }
}

impl Primitive {
    fn from_gltf(primitive: &gltf::Primitive, buffers: &[gltf::buffer::Data]) -> SceneResult<Self> {
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()].0));

        let positions: Vec<Vec3> = reader
            .read_positions()
            .ok_or_else(|| SceneError::MissingData("No positions in primitive".into()))?
            .map(|p| Vec3::new(p[0], p[1], p[2]))
            .collect();

        let normals: Vec<Vec3> = reader
            .read_normals()
            .ok_or_else(|| SceneError::MissingData("No normals in primitive".into()))?
            .map(|n| Vec3::new(n[0], n[1], n[2]))
            .collect();

        let mut texcoords = Vec::new();
        if let Some(file_texcoords) = reader.read_tex_coords(0) {
            texcoords = file_texcoords
                .into_f32()
                .map(|t| Vec2::new(t[0], t[1]))
                .collect();
        } else {
            // Generate dummy texcoords so the rasterizer doesn't complain
            texcoords.resize(positions.len(), Vec2::new(0.0, 0.0));
        }

        let indices: Vec<u32> = reader
            .read_indices()
            .ok_or_else(|| SceneError::MissingData("No indices in primitive".into()))?
            .into_u32()
            .collect();

        let bounding_sphere = compute_bounding_sphere(&positions);

        Ok(Primitive {
            positions,
            normals,
            texcoords,
            indices,
            bounding_sphere,
            material_index: primitive.material().index(),
        })
    }
}

fn compute_bounding_sphere(positions: &[Vec3]) -> BoundingSphere {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);

    for position in positions {
        min = min.min(*position);
        max = max.max(*position);
    }

    let center = min.midpoint(max);
    let radius = (max - min).length() / 2.0;

    BoundingSphere { center, radius }
}

fn get_texture_and_sampler(
    texture_gltf: &gltf::texture::Texture,
    document: &gltf::Document,
    texture_cache: &mut TextureCache,
) -> Option<TextureAndSampler> {
    match texture_gltf.source().source() {
        gltf::image::Source::Uri { uri, .. } => {
            let texture = texture_cache.get_or_create(uri);
            let sampler = if let Some(sampler_index) = texture_gltf.sampler().index() {
                // Convert the sampler
                let gltf_sampler = document.samplers().nth(sampler_index).unwrap();
                Sampler {
                    wrap_s: match gltf_sampler.wrap_s() {
                        gltf::texture::WrappingMode::ClampToEdge => WrapMode::ClampToEdge,
                        gltf::texture::WrappingMode::MirroredRepeat => WrapMode::MirroredRepeat,
                        gltf::texture::WrappingMode::Repeat => WrapMode::Repeat,
                    },
                    wrap_t: match gltf_sampler.wrap_t() {
                        gltf::texture::WrappingMode::ClampToEdge => WrapMode::ClampToEdge,
                        gltf::texture::WrappingMode::MirroredRepeat => WrapMode::MirroredRepeat,
                        gltf::texture::WrappingMode::Repeat => WrapMode::Repeat,
                    },
                    min_filter: match gltf_sampler.min_filter() {
                        Some(gltf::texture::MinFilter::Nearest) => Filter::Nearest,
                        Some(gltf::texture::MinFilter::Linear) => Filter::Linear,
                        Some(gltf::texture::MinFilter::NearestMipmapNearest) => Filter::Nearest,
                        Some(gltf::texture::MinFilter::LinearMipmapNearest) => Filter::Linear,
                        Some(gltf::texture::MinFilter::NearestMipmapLinear) => Filter::Nearest,
                        Some(gltf::texture::MinFilter::LinearMipmapLinear) => Filter::Linear,
                        None => Filter::Linear,
                    },
                    mag_filter: match gltf_sampler.mag_filter() {
                        Some(gltf::texture::MagFilter::Nearest) => Filter::Nearest,
                        Some(gltf::texture::MagFilter::Linear) => Filter::Linear,
                        None => Filter::Linear,
                    },
                }
            } else {
                // Default sampler state
                Sampler {
                    wrap_s: WrapMode::Repeat,
                    wrap_t: WrapMode::Repeat,
                    min_filter: Filter::Linear,
                    mag_filter: Filter::Linear,
                }
            };
            Some(TextureAndSampler { texture, sampler })
        }
        gltf::image::Source::View { .. } => None,
    }
}

impl Material {
    fn from_gltf(
        material: &gltf::Material,
        document: &gltf::Document,
        _buffers: &[gltf::buffer::Data],
        texture_cache: &mut TextureCache,
    ) -> SceneResult<Self> {
        let pbr = material.pbr_metallic_roughness();
        let base_color = pbr.base_color_factor();

        Ok(Material {
            name: material.name().map(String::from),
            base_color_factor: Vec4::new(
                base_color[0],
                base_color[1],
                base_color[2],
                base_color[3],
            ),
            base_color_texture: material
                .pbr_metallic_roughness()
                .base_color_texture()
                .and_then(|tex| get_texture_and_sampler(&tex.texture(), document, texture_cache)),
            metallic_factor: pbr.metallic_factor(),
            roughness_factor: pbr.roughness_factor(),
            metallic_roughness_texture: material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
                .and_then(|tex| get_texture_and_sampler(&tex.texture(), document, texture_cache)),
            normal_texture: material
                .normal_texture()
                .and_then(|tex| get_texture_and_sampler(&tex.texture(), document, texture_cache)),
            emissive_factor: Vec3::new(
                material.emissive_factor()[0],
                material.emissive_factor()[1],
                material.emissive_factor()[2],
            ),
            emissive_texture: material
                .emissive_texture()
                .and_then(|tex| get_texture_and_sampler(&tex.texture(), document, texture_cache)),
            occlusion_texture: material
                .occlusion_texture()
                .and_then(|tex| get_texture_and_sampler(&tex.texture(), document, texture_cache)),
            is_alpha_tested: material.alpha_mode() == gltf::material::AlphaMode::Mask,
            alpha_cutoff_vec: Vec4::splat(material.alpha_cutoff().unwrap_or(0.5)),
        })
    }
}

impl SceneCamera {
    fn from_gltf(camera: &gltf::Camera) -> SceneResult<Self> {
        let projection = match camera.projection() {
            gltf::camera::Projection::Perspective(perspective) => {
                let aspect_ratio = perspective.aspect_ratio().unwrap_or(1.0);
                let far = perspective.zfar().unwrap_or(100.0);

                Projection::Perspective {
                    fov: perspective.yfov(),
                    aspect_ratio,
                    near: perspective.znear(),
                    far,
                }
            }
            gltf::camera::Projection::Orthographic(ortho) => Projection::Orthographic {
                xmag: ortho.xmag(),
                ymag: ortho.ymag(),
                near: ortho.znear(),
                far: ortho.zfar(),
            },
        };

        Ok(SceneCamera {
            name: camera.name().map(String::from),
            transform: Mat4::IDENTITY, // Will be set by node transform
            projection,
        })
    }
}

#[derive(Debug)]
pub struct Light {
    pub normal: Vec3,
    pub color: Vec3,
}

pub struct SceneBounds {
    pub min: Vec3A,
    pub max: Vec3A,
}

impl SceneBounds {
    fn new_empty() -> Self {
        Self {
            min: Vec3A::INFINITY,
            max: Vec3A::NEG_INFINITY,
        }
    }

    fn grow(&mut self, point: Vec3A) {
        self.min = Vec3A::new(
            self.min.x.min(point.x),
            self.min.y.min(point.y),
            self.min.z.min(point.z),
        );
        self.max = Vec3A::new(
            self.max.x.max(point.x),
            self.max.y.max(point.y),
            self.max.z.max(point.z),
        );
    }
}

fn compute_scene_bounds_mesh(
    _scene: &Scene,
    mesh: &Mesh,
    transform: Mat4,
    bounds: &mut SceneBounds,
) {
    for primitive in &mesh.primitives {
        for position in &primitive.positions {
            // Transform vertex position by node's transform
            let pos = Vec4::new(position[0], position[1], position[2], 1.0);
            let transformed = transform * pos;
            bounds.grow(Vec3A::new(transformed.x, transformed.y, transformed.z));
        }
    }
}

fn compute_scene_bounds_node(
    scene: &Scene,
    node: &Node,
    parent_transform: Mat4,
    bounds: &mut SceneBounds,
) {
    // Get node's transform
    let transform = parent_transform * node.transform;

    // Process node's mesh if it has one
    if let Some(mesh_index) = node.mesh_index {
        let mesh = &scene.meshes[mesh_index];
        compute_scene_bounds_mesh(scene, &mesh, transform, bounds);
    }

    // Process child nodes
    for child_index in &node.children {
        let child = &scene.nodes[*child_index];
        compute_scene_bounds_node(scene, child, transform, bounds);
    }
}

pub fn compute_scene_bounds(scene: &Scene) -> SceneBounds {
    let mut bounds = SceneBounds::new_empty();

    // Process all nodes in the scene
    for node in &scene.nodes {
        compute_scene_bounds_node(scene, node, Mat4::IDENTITY, &mut bounds);
    }

    bounds
}
