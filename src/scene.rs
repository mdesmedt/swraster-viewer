use glam::{Mat4, Quat, Vec2, Vec3, Vec3A, Vec4};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

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
    pub center: Vec3A,
    pub radius: f32,
}

impl BoundingSphere {
    pub fn new() -> Self {
        Self {
            center: Vec3A::ZERO,
            radius: 0.0,
        }
    }

    pub fn grow_by_sphere(&mut self, sphere: &BoundingSphere) {
        let distance = (self.center - sphere.center).length();
        if distance + sphere.radius > self.radius {
            self.radius = distance + sphere.radius;
        }
    }
}

// Implement multiplication of Mat4 by BoundingSphere
impl std::ops::Mul<&BoundingSphere> for Mat4 {
    type Output = BoundingSphere;

    fn mul(self, sphere: &BoundingSphere) -> Self::Output {
        let max_scale = (self.x_axis.length() + self.y_axis.length() + self.z_axis.length()) / 3.0;
        BoundingSphere {
            center: self.transform_point3a(sphere.center),
            radius: sphere.radius * max_scale,
        }
    }
}

pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub nodes: Vec<Node>,
    pub materials: Vec<Material>,
    pub cameras: Vec<SceneCamera>,
    pub bounds: SceneBounds,
    pub light: Light,
    pub voxel_grid: crate::voxelgrid::VoxelGrid,
    pub cubemap: TextureAndSampler,
    pub cubemap_specular: TextureAndSampler,
    pub irradiance_sh: [Vec3A; 4],
    pub brdf_lut: TextureAndSampler,
}

pub struct Node {
    pub name: Option<String>,
    pub transform: Mat4, // Local to world matrix
    pub mesh_index: Option<usize>,
    pub camera_index: Option<usize>,
    pub bounding_sphere_world: BoundingSphere,
}

pub struct Mesh {
    pub name: Option<String>,
    pub primitives: Vec<Primitive>,
    pub primitives_opaque: Vec<usize>,
    pub primitives_translucent: Vec<usize>,
}

pub struct Primitive {
    pub positions: Vec<Vec4>,
    pub normals: Vec<Vec3A>,
    pub tangents: Vec<Vec4>,
    pub texcoords: Vec<Vec2>,
    pub indices: Vec<u32>,
    pub material_index: usize,
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
    pub emissive_factor: Vec3A,
    pub emissive_texture: Option<TextureAndSampler>,
    pub occlusion_texture: Option<TextureAndSampler>,
    pub occlusion_strength: f32,
    pub is_alpha_tested: bool,
    pub is_translucent: bool,
    pub transmission: f32,
    pub alpha_cutoff_vec: Vec4,
    pub transmission_texture: Option<TextureAndSampler>,
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
        texture_cache_builtin: &mut TextureCache,
    ) -> SceneResult<Self> {
        let cubemap_texture =
            texture_cache_builtin.get_or_create("cubemap.jpg", TextureType::Cubemap);
        let cubemap_sampler = Sampler {
            wrap_s: WrapMode::ClampToEdge,
            wrap_t: WrapMode::ClampToEdge,
            _min_filter: Filter::Linear,
            _mag_filter: Filter::Linear,
        };
        let cubemap = TextureAndSampler {
            texture: cubemap_texture.clone(),
            sampler: cubemap_sampler,
        };
        let cubemap_ggx_cache_path = Path::new("assets/cubemap.ggx");
        let cubemap_specular_texture = match try_load_prefiltered_specular_cubemap_cache(
            cubemap_ggx_cache_path,
            cubemap.texture.width,
            cubemap.texture.height,
        ) {
            Ok(Some(texture)) => {
                println!("Loaded GGX cubemap cache: {}", cubemap_ggx_cache_path.display());
                Arc::new(texture)
            }
            Ok(None) => {
                println!(
                    "Building GGX cubemap cache: {}",
                    cubemap_ggx_cache_path.display()
                );
                let texture = generate_prefiltered_specular_cubemap(&cubemap, 64);
                if let Err(err) =
                    save_prefiltered_specular_cubemap_cache(cubemap_ggx_cache_path, &texture)
                {
                    eprintln!(
                        "Failed to write GGX cubemap cache {}: {}",
                        cubemap_ggx_cache_path.display(),
                        err
                    );
                }
                Arc::new(texture)
            }
            Err(err) => {
                eprintln!(
                    "Failed to read GGX cubemap cache {} (recomputing): {}",
                    cubemap_ggx_cache_path.display(),
                    err
                );
                let texture = generate_prefiltered_specular_cubemap(&cubemap, 64);
                if let Err(write_err) =
                    save_prefiltered_specular_cubemap_cache(cubemap_ggx_cache_path, &texture)
                {
                    eprintln!(
                        "Failed to write GGX cubemap cache {}: {}",
                        cubemap_ggx_cache_path.display(),
                        write_err
                    );
                }
                Arc::new(texture)
            }
        };
        let cubemap_specular = TextureAndSampler {
            texture: cubemap_specular_texture,
            sampler: Sampler {
                wrap_s: WrapMode::ClampToEdge,
                wrap_t: WrapMode::ClampToEdge,
                _min_filter: Filter::Linear,
                _mag_filter: Filter::Linear,
            },
        };
        let irradiance_sh = compute_irradiance_sh4(&cubemap);
        let brdf_lut = TextureAndSampler {
            texture: Arc::new(generate_brdf_lut(128)),
            sampler: Sampler {
                wrap_s: WrapMode::ClampToEdge,
                wrap_t: WrapMode::ClampToEdge,
                _min_filter: Filter::Linear,
                _mag_filter: Filter::Linear,
            },
        };

        let mut scene = Scene {
            meshes: Vec::new(),
            nodes: Vec::new(),
            materials: Vec::new(),
            cameras: Vec::new(),
            bounds: SceneBounds::new_empty(),
            light: Light {
                direction: Vec3A::new(-0.2, 1.0, 0.5).normalize(),
                color: Vec3A::new(1.0, 1.0, 0.95) * 5.0,
            },
            voxel_grid: crate::voxelgrid::VoxelGrid::new(1, 1, 1, Vec3A::ZERO, Vec3A::ONE),
            cubemap,
            cubemap_specular,
            irradiance_sh,
            brdf_lut,
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

        // Load materials in parallel
        let gltf_materials = document.materials().collect::<Vec<_>>();
        scene.materials = gltf_materials
            .par_iter()
            .map(|material| Material::from_gltf(material, document, buffers, texture_cache))
            .collect::<SceneResult<Vec<_>>>()?;

        // Collect cameras
        for camera in document.cameras() {
            scene.cameras.push(SceneCamera::from_gltf(&camera)?);
        }

        // Pre-separate opaque and translucent primitives
        for mesh in &mut scene.meshes {
            for (index, primitive) in mesh.primitives.iter().enumerate() {
                if scene.materials[primitive.material_index].is_translucent {
                    mesh.primitives_translucent.push(index);
                } else {
                    mesh.primitives_opaque.push(index);
                }
            }
        }

        // Flatten the GLTF node hierarchy into a single vector for performance reasons

        // Collect all nodes with their local transforms
        let mut node_indices = HashMap::with_capacity(node_count);
        let mut node_transforms = HashMap::with_capacity(node_count);
        for node in document.nodes() {
            let node_index = scene.nodes.len();
            node_indices.insert(node.index(), node_index);
            let local_transform = Node::get_local_transform(&node);
            node_transforms.insert(node.index(), local_transform);
            scene.nodes.push(Node::from_gltf(&node, local_transform)?);
        }

        // Find root nodes (nodes that are not children of any other node)
        let child_indices: HashSet<_> = document
            .nodes()
            .flat_map(|node| node.children().map(|child| child.index()))
            .collect();
        let root_nodes: Vec<_> = document
            .nodes()
            .filter(|node| !child_indices.contains(&node.index()))
            .collect();

        // Compute final transforms by traversing from root nodes
        for root_node in root_nodes {
            Self::compute_final_transform_recursive(
                &root_node,
                &node_indices,
                &node_transforms,
                Mat4::IDENTITY,
                &mut scene.nodes,
            );
        }

        // Compute world-space bounding spheres for all nodes
        for node in &mut scene.nodes {
            let transform = node.transform;
            if let Some(mesh_index) = node.mesh_index {
                let mesh = &scene.meshes[mesh_index];
                for primitive in &mesh.primitives {
                    let transformed_sphere = transform * &primitive.bounding_sphere;
                    node.bounding_sphere_world
                        .grow_by_sphere(&transformed_sphere);
                }
            }
        }

        // Compute scene bounds
        // Process all nodes in the scene
        for node in &scene.nodes {
            if let Some(mesh_index) = node.mesh_index {
                let mesh = &scene.meshes[mesh_index];
                for primitive in &mesh.primitives {
                    for position in &primitive.positions {
                        // Transform vertex position by node's transform
                        let pos = Vec4::new(position[0], position[1], position[2], 1.0);
                        let transformed = node.transform * pos;
                        scene
                            .bounds
                            .grow(Vec3A::new(transformed.x, transformed.y, transformed.z));
                    }
                }
            }
        }
        scene.bounds.center = (scene.bounds.min + scene.bounds.max) * 0.5;
        scene.bounds.diagonal = (scene.bounds.max - scene.bounds.min).length();

        Ok(scene)
    }

    fn compute_final_transform_recursive(
        node: &gltf::Node,
        node_indices: &HashMap<usize, usize>,
        node_transforms: &HashMap<usize, Mat4>,
        parent_transform: Mat4,
        scene_nodes: &mut [Node],
    ) {
        let node_index = node_indices[&node.index()];
        let local_transform = node_transforms[&node.index()];
        let final_transform = parent_transform * local_transform;

        // Update the node's transform
        scene_nodes[node_index].transform = final_transform;

        // Recursively process all children
        for child in node.children() {
            Self::compute_final_transform_recursive(
                &child,
                node_indices,
                node_transforms,
                final_transform,
                scene_nodes,
            );
        }
    }
}

impl Node {
    fn from_gltf(node: &gltf::Node, transform: Mat4) -> SceneResult<Self> {
        let mut bounding_sphere = BoundingSphere::new();
        bounding_sphere.center = Vec3A::from_vec4(transform * Vec4::new(0.0, 0.0, 0.0, 1.0));
        Ok(Node {
            name: node.name().map(String::from),
            transform,
            mesh_index: node.mesh().map(|m| m.index()),
            camera_index: node.camera().map(|c| c.index()),
            bounding_sphere_world: bounding_sphere,
        })
    }

    fn get_local_transform(node: &gltf::Node) -> Mat4 {
        match node.transform() {
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
        }
    }
}

impl Mesh {
    fn from_gltf(mesh: &gltf::Mesh, buffers: &[gltf::buffer::Data]) -> SceneResult<Self> {
        let mut primitives = Vec::with_capacity(mesh.primitives().len());

        for primitive in mesh.primitives() {
            primitives.push(Primitive::from_gltf(&primitive, buffers)?);
        }

        Ok(Mesh {
            name: mesh.name().map(String::from),
            primitives,
            primitives_opaque: Vec::new(),
            primitives_translucent: Vec::new(),
        })
    }
}

impl Primitive {
    fn from_gltf(primitive: &gltf::Primitive, buffers: &[gltf::buffer::Data]) -> SceneResult<Self> {
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()].0));

        let positions: Vec<Vec4> = reader
            .read_positions()
            .ok_or_else(|| SceneError::MissingData("No positions in primitive".into()))?
            .map(|p| Vec4::new(p[0], p[1], p[2], 1.0))
            .collect();

        let indices: Vec<u32> = reader
            .read_indices()
            .ok_or_else(|| SceneError::MissingData("No indices in primitive".into()))?
            .into_u32()
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

        let normals: Vec<Vec3A> = if let Some(file_normals) = reader.read_normals() {
            file_normals.map(|n| Vec3A::new(n[0], n[1], n[2])).collect()
        } else {
            // Compute smooth normals automatically when they're missing
            println!("Computing automatic vertex normals");
            compute_smooth_normals(&positions, &indices)
        };

        let tangents: Vec<Vec4> = if let Some(file_tangents) = reader.read_tangents() {
            file_tangents
                .map(|t| Vec4::new(t[0], t[1], t[2], t[3]))
                .collect()
        } else {
            // Just clone the normals as tangents, without normal mapping it doesn't matter anyway
            println!("Computing automatic vertex tangents");
            compute_tangents(&positions, &texcoords, &normals, &indices)
        };

        let bounding_sphere = compute_bounding_sphere(&positions);

        Ok(Primitive {
            positions,
            normals,
            tangents,
            texcoords,
            indices,
            bounding_sphere,
            material_index: primitive.material().index().unwrap_or(0), // TODO: This is wrong. Use a default material instead.
        })
    }
}

fn compute_bounding_sphere(positions: &[Vec4]) -> BoundingSphere {
    let mut min = Vec3A::splat(f32::INFINITY);
    let mut max = Vec3A::splat(f32::NEG_INFINITY);

    for position in positions {
        min = min.min(Vec3A::from_vec4(*position));
        max = max.max(Vec3A::from_vec4(*position));
    }

    let center = min.midpoint(max);
    let radius = (max - min).length() / 2.0;

    BoundingSphere { center, radius }
}

fn compute_smooth_normals(positions: &[Vec4], indices: &[u32]) -> Vec<Vec3A> {
    let mut normals = vec![Vec3A::ZERO; positions.len()];

    // Compute face normals and accumulate them at each vertex
    for triangle in indices.chunks_exact(3) {
        let v0 = positions[triangle[0] as usize];
        let v1 = positions[triangle[1] as usize];
        let v2 = positions[triangle[2] as usize];

        // Compute face normal
        let edge1 = Vec3A::from_vec4(v1 - v0);
        let edge2 = Vec3A::from_vec4(v2 - v0);
        let face_normal = edge1.cross(edge2);

        // Accumulate face normal at each vertex of the triangle
        normals[triangle[0] as usize] += face_normal;
        normals[triangle[1] as usize] += face_normal;
        normals[triangle[2] as usize] += face_normal;
    }

    // Normalize all accumulated normals
    for normal in &mut normals {
        if normal.length_squared() > 0.0 {
            *normal = normal.normalize();
        } else {
            // Fallback for vertices with no connected faces
            *normal = Vec3A::new(0.0, 0.0, 1.0);
        }
    }

    assert!(normals.len() == positions.len(), "Incorrect normals count");

    normals
}

pub fn compute_tangents(
    positions: &[Vec4],
    uvs: &[Vec2],
    normals: &[Vec3A],
    indices: &[u32],
) -> Vec<Vec4> {
    let vertex_count = positions.len();
    assert!(uvs.len() == vertex_count && normals.len() == vertex_count);

    // Temporary accumulators
    let mut tan_acc: Vec<Vec3> = vec![Vec3::ZERO; vertex_count];
    let mut bit_acc: Vec<Vec3> = vec![Vec3::ZERO; vertex_count];

    // Output tangents
    let mut out_tangents: Vec<Vec4> = vec![Vec4::ZERO; vertex_count];

    for tri in (0..indices.len()).step_by(3) {
        let i0 = indices[tri] as usize;
        let i1 = indices[tri + 1] as usize;
        let i2 = indices[tri + 2] as usize;

        let p0 = positions[i0].truncate();
        let p1 = positions[i1].truncate();
        let p2 = positions[i2].truncate();

        let uv0 = uvs[i0];
        let uv1 = uvs[i1];
        let uv2 = uvs[i2];

        // Edges in model space
        let edge1 = p1 - p0;
        let edge2 = p2 - p0;

        // Edges in UV space
        let duv1 = uv1 - uv0;
        let duv2 = uv2 - uv0;

        // UV area
        let det = duv1.x * duv2.y - duv2.x * duv1.y;

        // Degenerate triangle
        if det.abs() < 1e-6 {
            continue;
        }

        let tangent = (edge1 * duv2.y - edge2 * duv1.y) / det;
        let bitangent = (edge2 * duv1.x - edge1 * duv2.x) / det;

        // Accumulate
        tan_acc[i0] += tangent;
        tan_acc[i1] += tangent;
        tan_acc[i2] += tangent;

        bit_acc[i0] += bitangent;
        bit_acc[i1] += bitangent;
        bit_acc[i2] += bitangent;
    }

    for i in 0..vertex_count {
        let n = normals[i].to_vec3();
        let t = tan_acc[i];

        // If tangent accumulator is nearly zero, create a fallback tangent
        if t.length_squared() < 1e-6 {
            let helper = if n.x.abs() > 0.9 {
                Vec3::new(0.0, 1.0, 0.0)
            } else {
                Vec3::new(1.0, 0.0, 0.0)
            };
            let tangent = n.cross(helper).normalize();
            out_tangents[i] = Vec4::new(tangent.x, tangent.y, tangent.z, 1.0);
            continue;
        }

        // Orthonormalize
        let tangent = (t - n * n.dot(t)).normalize();

        // Compute binormal handedness
        let b = bit_acc[i];
        let cross_nt = n.cross(tangent);
        let handedness = if cross_nt.dot(b) < 0.0 { -1.0 } else { 1.0 };

        // Store tangent
        out_tangents[i] = Vec4::new(tangent.x, tangent.y, tangent.z, handedness);
    }

    out_tangents
}

fn get_texture_and_sampler(
    texture_gltf: &gltf::texture::Texture,
    document: &gltf::Document,
    texture_cache: &TextureCache,
    texture_type: TextureType,
) -> Option<TextureAndSampler> {
    match texture_gltf.source().source() {
        gltf::image::Source::Uri { uri, .. } => {
            let texture = texture_cache.get_or_create(uri, texture_type);
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
                    _min_filter: match gltf_sampler.min_filter() {
                        Some(gltf::texture::MinFilter::Nearest) => Filter::Nearest,
                        Some(gltf::texture::MinFilter::Linear) => Filter::Linear,
                        Some(gltf::texture::MinFilter::NearestMipmapNearest) => Filter::Nearest,
                        Some(gltf::texture::MinFilter::LinearMipmapNearest) => Filter::Linear,
                        Some(gltf::texture::MinFilter::NearestMipmapLinear) => Filter::Nearest,
                        Some(gltf::texture::MinFilter::LinearMipmapLinear) => Filter::Linear,
                        None => Filter::Linear,
                    },
                    _mag_filter: match gltf_sampler.mag_filter() {
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
                    _min_filter: Filter::Linear,
                    _mag_filter: Filter::Linear,
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
        texture_cache: &TextureCache,
    ) -> SceneResult<Self> {
        let pbr = material.pbr_metallic_roughness();
        let base_color = pbr.base_color_factor();

        let base_color_texture = material
            .pbr_metallic_roughness()
            .base_color_texture()
            .and_then(|tex| {
                get_texture_and_sampler(&tex.texture(), document, texture_cache, TextureType::SRGB)
            });
        let is_alpha_tested = material.alpha_mode() == gltf::material::AlphaMode::Mask
            && base_color_texture.is_some();

        let is_translucent = material.transmission().is_some();

        Ok(Material {
            name: material.name().map(String::from),
            base_color_factor: Vec4::new(
                base_color[0],
                base_color[1],
                base_color[2],
                base_color[3],
            ),
            base_color_texture,
            metallic_factor: pbr.metallic_factor(),
            roughness_factor: pbr.roughness_factor(),
            metallic_roughness_texture: material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
                .and_then(|tex| {
                    get_texture_and_sampler(
                        &tex.texture(),
                        document,
                        texture_cache,
                        TextureType::MetallicRoughness,
                    )
                }),
            normal_texture: material.normal_texture().and_then(|tex| {
                get_texture_and_sampler(
                    &tex.texture(),
                    document,
                    texture_cache,
                    TextureType::Normal,
                )
            }),
            emissive_factor: Vec3A::new(
                material.emissive_factor()[0],
                material.emissive_factor()[1],
                material.emissive_factor()[2],
            ),
            emissive_texture: material.emissive_texture().and_then(|tex| {
                get_texture_and_sampler(&tex.texture(), document, texture_cache, TextureType::SRGB)
            }),
            occlusion_texture: material.occlusion_texture().and_then(|tex| {
                get_texture_and_sampler(
                    &tex.texture(),
                    document,
                    texture_cache,
                    TextureType::Linear,
                )
            }),
            occlusion_strength: material
                .occlusion_texture()
                .map(|t| t.strength())
                .unwrap_or(1.0),
            is_alpha_tested,
            is_translucent,
            alpha_cutoff_vec: Vec4::splat(material.alpha_cutoff().unwrap_or(0.5)),
            transmission: material
                .transmission()
                .map(|t| t.transmission_factor())
                .unwrap_or(0.0),
            transmission_texture: material
                .transmission()
                .and_then(|t| t.transmission_texture())
                .and_then(|tex| {
                    get_texture_and_sampler(
                        &tex.texture(),
                        document,
                        texture_cache,
                        TextureType::Linear,
                    )
                }),
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

pub struct Light {
    pub direction: Vec3A,
    pub color: Vec3A,
}

pub struct SceneBounds {
    pub min: Vec3A,
    pub max: Vec3A,
    pub center: Vec3A,
    pub diagonal: f32,
}

impl SceneBounds {
    fn new_empty() -> Self {
        Self {
            min: Vec3A::INFINITY,
            max: Vec3A::NEG_INFINITY,
            center: Vec3A::ZERO,
            diagonal: 0.0,
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
