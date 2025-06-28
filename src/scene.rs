use glam::{Mat4, Quat, Vec2, Vec3, Vec3A, Vec4};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

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

type SceneResult<T> = Result<T, SceneError>;

#[derive(Debug)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub nodes: Vec<Node>,
    pub materials: Vec<Material>,
    pub cameras: Vec<SceneCamera>,
    pub root_nodes: Vec<usize>, // Indices into nodes array
}

#[derive(Debug)]
pub struct Node {
    pub name: Option<String>,
    pub transform: Mat4, // Model matrix
    pub mesh_index: Option<usize>,
    pub camera_index: Option<usize>,
    pub children: Vec<usize>, // Indices into nodes array
}

#[derive(Debug)]
pub struct Mesh {
    pub name: Option<String>,
    pub primitives: Vec<Primitive>,
    pub light: Light,
}

#[derive(Debug)]
pub struct Primitive {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub texcoords: Vec<Vec2>,
    pub indices: Vec<u32>,
    pub material_index: Option<usize>,
}

#[derive(Debug)]
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
}

#[derive(Debug)]
pub struct Texture {
    pub uri: Option<String>,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // RGBA8 data
}

#[derive(Debug)]
pub enum WrapMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

#[derive(Debug)]
pub enum Filter {
    Nearest,
    Linear,
}

#[derive(Debug)]
pub struct Sampler {
    pub wrap_s: WrapMode,
    pub wrap_t: WrapMode,
    pub min_filter: Filter,
    pub mag_filter: Filter,
}

#[derive(Debug)]
pub struct TextureAndSampler {
    pub texture: Arc<Texture>,
    pub sampler: Sampler,
}

#[derive(Debug)]
pub struct SceneCamera {
    pub name: Option<String>,
    pub transform: Mat4, // View matrix
    pub projection: Projection,
}

#[derive(Debug)]
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

        Ok(Primitive {
            positions,
            normals,
            texcoords,
            indices,
            material_index: primitive.material().index(),
        })
    }
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
        })
    }
}

impl Texture {
    fn from_gltf(texture: &gltf::Texture) -> SceneResult<Self> {
        let image = texture.source();
        let (uri, width, height) = match image.source() {
            gltf::image::Source::Uri { uri, .. } => (Some(uri.to_string()), 0, 0),
            gltf::image::Source::View { .. } => (None, 0, 0),
        };

        Ok(Texture {
            uri,
            width,
            height,
            data: Vec::new(),
        })
    }
}

impl TextureAndSampler {
    fn apply_wrap_mode(coord: f32, mode: &WrapMode) -> f32 {
        match mode {
            WrapMode::ClampToEdge => coord.clamp(0.0, 1.0),
            WrapMode::Repeat => {
                let wrapped = coord - coord.floor();
                if wrapped < 0.0 {
                    wrapped + 1.0
                } else {
                    wrapped
                }
            }
            WrapMode::MirroredRepeat => {
                let abs_coord = coord.abs();
                let wrapped = abs_coord - abs_coord.floor();
                if coord < 0.0 {
                    1.0 - wrapped
                } else {
                    wrapped
                }
            }
        }
    }

    pub fn sample(&self, uv: Vec2) -> Vec4 {
        if self.texture.data.is_empty() {
            return Vec4::new(1.0, 0.0, 1.0, 1.0); // Magenta for missing texture
        }

        // Apply wrap modes to UV coordinates
        let u = Self::apply_wrap_mode(uv.x, &self.sampler.wrap_s);
        let v = Self::apply_wrap_mode(uv.y, &self.sampler.wrap_t);

        // Convert to pixel coordinates
        let x = (u * (self.texture.width - 1) as f32) as u32;
        let y = (v * (self.texture.height - 1) as f32) as u32;

        // Get pixel index
        let pixel_index = ((y * self.texture.width + x) * 4) as usize;

        if pixel_index + 3 >= self.texture.data.len() {
            return Vec4::new(1.0, 0.0, 1.0, 1.0); // Magenta for out of bounds
        }

        // Read RGBA values and convert to float [0, 1]
        let r = self.texture.data[pixel_index] as f32 / 255.0;
        let g = self.texture.data[pixel_index + 1] as f32 / 255.0;
        let b = self.texture.data[pixel_index + 2] as f32 / 255.0;
        let a = self.texture.data[pixel_index + 3] as f32 / 255.0;

        Vec4::new(r, g, b, a)
    }

    pub fn sample_vec4(&self, u_vec: Vec4, v_vec: Vec4) -> [Vec4; 4] {
        if self.texture.data.is_empty() {
            // Return magenta for missing texture
            let magenta = Vec4::new(1.0, 0.0, 1.0, 1.0);
            return [magenta, magenta, magenta, magenta];
        }

        // Sample four pixels
        let mut pixels = [Vec4::ZERO; 4];

        for i in 0..4 {
            // Apply wrap modes to UV coordinates
            let u = Self::apply_wrap_mode(u_vec[i], &self.sampler.wrap_s);
            let v = Self::apply_wrap_mode(v_vec[i], &self.sampler.wrap_t);

            // Convert to pixel coordinates
            let x = (u * (self.texture.width - 1) as f32) as u32;
            let y = (v * (self.texture.height - 1) as f32) as u32;

            // Get pixel index
            let pixel_index = ((y * self.texture.width + x) * 4) as usize;

            if pixel_index + 3 >= self.texture.data.len() {
                pixels[i] = Vec4::new(1.0, 0.0, 1.0, 1.0); // Magenta for out of bounds
            } else {
                // Read RGBA values and convert to float [0, 1]
                let r = self.texture.data[pixel_index] as f32 / 255.0;
                let g = self.texture.data[pixel_index + 1] as f32 / 255.0;
                let b = self.texture.data[pixel_index + 2] as f32 / 255.0;
                let a = self.texture.data[pixel_index + 3] as f32 / 255.0;

                pixels[i] = Vec4::new(r, g, b, a);
            }
        }

        // Rearrange into [reds, greens, blues, alphas] format
        [
            Vec4::new(pixels[0].x, pixels[1].x, pixels[2].x, pixels[3].x), // All reds
            Vec4::new(pixels[0].y, pixels[1].y, pixels[2].y, pixels[3].y), // All greens
            Vec4::new(pixels[0].z, pixels[1].z, pixels[2].z, pixels[3].z), // All blues
            Vec4::new(pixels[0].w, pixels[1].w, pixels[2].w, pixels[3].w), // All alphas
        ]
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

#[derive(Debug)]
pub struct TextureCache {
    textures: HashMap<String, Arc<Texture>>,
    base_dir: Option<String>,
}

impl TextureCache {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            base_dir: None,
        }
    }

    pub fn with_base_dir(base_dir: String) -> Self {
        Self {
            textures: HashMap::new(),
            base_dir: Some(base_dir),
        }
    }

    pub fn get_or_create(&mut self, uri: &str) -> Arc<Texture> {
        if let Some(texture) = self.textures.get(uri) {
            Arc::clone(texture)
        } else {
            // Load the texture from file
            let texture = match self.load_texture(uri) {
                Ok(tex) => Arc::new(tex),
                Err(e) => {
                    eprintln!("Failed to load texture '{}': {}", uri, e);
                    // Create a fallback texture (checkerboard pattern)
                    Arc::new(self.create_fallback_texture())
                }
            };
            self.textures.insert(uri.to_string(), Arc::clone(&texture));
            texture
        }
    }

    fn load_texture(&self, uri: &str) -> SceneResult<Texture> {
        // Construct the full path based on GLTF spec
        let texture_path = if let Some(ref base_dir) = self.base_dir {
            // If URI is absolute (starts with http://, https://, or file://), use as-is
            if uri.starts_with("http://")
                || uri.starts_with("https://")
                || uri.starts_with("file://")
            {
                uri.to_string()
            } else {
                // Otherwise, make it relative to the base directory
                format!("{}/{}", base_dir, uri)
            }
        } else {
            // No base directory, try URI as-is
            uri.to_string()
        };

        // Try to load the texture
        match image::open(&texture_path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let (width, height) = rgba.dimensions();
                let data = rgba.into_raw();
                Ok(Texture {
                    uri: Some(uri.to_string()),
                    width,
                    height,
                    data,
                })
            }
            Err(e) => Err(SceneError::MissingData(format!(
                "Could not load texture '{}' from path '{}': {}",
                uri, texture_path, e
            ))),
        }
    }

    fn create_fallback_texture(&self) -> Texture {
        // Create a simple checkerboard pattern as fallback
        let width = 64;
        let height = 64;
        let mut data = Vec::with_capacity((width * height * 4) as usize);

        for y in 0..height {
            for x in 0..width {
                let is_checker = ((x / 8) + (y / 8)) % 2 == 0;
                let color = if is_checker { 255 } else { 128 };
                data.extend_from_slice(&[color, color, color, 255]);
            }
        }

        Texture {
            uri: None,
            width,
            height,
            data,
        }
    }

    pub fn unique_texture_count(&self) -> usize {
        self.textures.len()
    }

    pub fn total_texture_data_size(&self) -> usize {
        self.textures
            .values()
            .map(|texture| texture.data.len())
            .sum()
    }
}
