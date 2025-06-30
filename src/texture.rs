use glam::Vec4;
use std::collections::HashMap;
use std::sync::Arc;

use crate::scene::{SceneError, SceneResult};

pub struct Texture {
    pub uri: Option<String>,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // RGBA8 data
}

pub enum WrapMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

pub enum Filter {
    Nearest,
    Linear,
}

pub struct Sampler {
    pub wrap_s: WrapMode,
    pub wrap_t: WrapMode,
    pub min_filter: Filter,
    pub mag_filter: Filter,
}

pub struct TextureAndSampler {
    pub texture: Arc<Texture>,
    pub sampler: Sampler,
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

    pub fn sample_vec4(&self, u_vec: Vec4, v_vec: Vec4) -> [Vec4; 4] {
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

            // Read RGBA values and convert to float [0, 1]
            let r = self.texture.data[pixel_index] as f32 / 255.0;
            let g = self.texture.data[pixel_index + 1] as f32 / 255.0;
            let b = self.texture.data[pixel_index + 2] as f32 / 255.0;
            let a = self.texture.data[pixel_index + 3] as f32 / 255.0;

            pixels[i] = Vec4::new(r, g, b, a);
        }

        // Transpose to [reds, greens, blues, alphas]
        [
            Vec4::new(pixels[0].x, pixels[1].x, pixels[2].x, pixels[3].x), // All reds
            Vec4::new(pixels[0].y, pixels[1].y, pixels[2].y, pixels[3].y), // All greens
            Vec4::new(pixels[0].z, pixels[1].z, pixels[2].z, pixels[3].z), // All blues
            Vec4::new(pixels[0].w, pixels[1].w, pixels[2].w, pixels[3].w), // All alphas
        ]
    }
}

pub struct TextureCache {
    textures: HashMap<String, Arc<Texture>>,
    base_dir: Option<String>,
}

impl TextureCache {
    pub fn new(base_dir: String) -> Self {
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
