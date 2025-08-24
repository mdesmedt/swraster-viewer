use glam::{Mat4, Vec4};
use std::collections::HashMap;
use std::sync::Arc;

use crate::scene::{SceneError, SceneResult};

pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub data: Vec<Vec4>, // Vec4 floating point data
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
    pub _min_filter: Filter,
    pub _mag_filter: Filter,
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

    pub fn sample_cubemap(&self, normal_x: Vec4, normal_y: Vec4, normal_z: Vec4) -> Mat4 {
        let mut u_out = Vec4::ZERO;
        let mut v_out = Vec4::ZERO;

        for i in 0..4 {
            let x = normal_x[i];
            let y = normal_y[i];
            let z = normal_z[i];

            let ax = x.abs();
            let ay = y.abs();
            let az = z.abs();

            // dominant axis → pick face
            let (face, sc, tc, ma) = if ax >= ay && ax >= az {
                if x > 0.0 {
                    // +X
                    (0u32, -z, -y, ax)
                } else {
                    // -X
                    (1u32, z, -y, ax)
                }
            } else if ay >= ax && ay >= az {
                if y > 0.0 {
                    // +Y
                    (2u32, x, z, ay)
                } else {
                    // -Y
                    (3u32, x, -z, ay)
                }
            } else {
                if z > 0.0 {
                    // +Z
                    (4u32, x, -y, az)
                } else {
                    // -Z
                    (5u32, -x, -y, az)
                }
            };

            // [-1,1] → [0,1]
            let u = 0.5 * (sc / ma + 1.0);
            let v = 0.5 * (tc / ma + 1.0);

            // face placement in atlas (col,row)
            let (col, row) = match face {
                0 => (2, 1), // +X
                1 => (0, 1), // -X
                2 => (1, 0), // +Y
                3 => (1, 2), // -Y
                4 => (1, 1), // +Z
                5 => (3, 1), // -Z
                _ => unreachable!(),
            };

            // atlas is 4 faces wide × 3 faces tall
            let u_final = (u + col as f32) / 4.0;
            let v_final = (v + row as f32) / 3.0;

            u_out[i] = u_final;
            v_out[i] = v_final;
        }

        // Sample four texels
        let mut texels = Mat4::ZERO;
        for i in 0..4 {
            // Convert to pixel coordinates
            let x = (u_out[i] * (self.texture.width - 1) as f32) as u32;
            let y = (v_out[i] * (self.texture.height - 1) as f32) as u32;

            // Point sample the texture
            let pixel_index = (y * self.texture.width + x) as usize;
            *texels.col_mut(i) = self.texture.data[pixel_index];
        }
        // Return the samples in columns of X, Y, Z and W
        texels.transpose()
    }

    pub fn sample4(&self, u_vec: Vec4, v_vec: Vec4) -> Mat4 {
        // Sample four texels
        let mut texels = Mat4::ZERO;
        for i in 0..4 {
            // Apply wrap modes to UV coordinates
            let u = Self::apply_wrap_mode(u_vec[i], &self.sampler.wrap_s);
            let v = Self::apply_wrap_mode(v_vec[i], &self.sampler.wrap_t);

            // Texel coordinates
            let x_f = u * (self.texture.width - 1) as f32;
            let y_f = v * (self.texture.height - 1) as f32;

            // Sample four texels
            let x0 = x_f.floor() as u32;
            let y0 = y_f.floor() as u32;
            let x1 = (x0 + 1).min(self.texture.width - 1);
            let y1 = (y0 + 1).min(self.texture.height - 1);
            let p00 = self.texture.data[(y0 * self.texture.width + x0) as usize];
            let p10 = self.texture.data[(y0 * self.texture.width + x1) as usize];
            let p01 = self.texture.data[(y1 * self.texture.width + x0) as usize];
            let p11 = self.texture.data[(y1 * self.texture.width + x1) as usize];

            // Bilinear filter
            let fx = x_f - x0 as f32;
            let fy = y_f - y0 as f32;
            let lerp_x0 = p00 * (1.0 - fx) + p10 * fx;
            let lerp_x1 = p01 * (1.0 - fx) + p11 * fx;
            let final_color = lerp_x0 * (1.0 - fy) + lerp_x1 * fy;

            *texels.col_mut(i) = final_color;
        }
        // Return the samples in columns of X, Y, Z and W
        texels.transpose()
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
                let raw_data = rgba.into_raw();
                let mut data = Vec::with_capacity((width * height) as usize);
                for chunk in raw_data.chunks(4) {
                    if chunk.len() == 4 {
                        let r = chunk[0] as f32 / 255.0;
                        let g = chunk[1] as f32 / 255.0;
                        let b = chunk[2] as f32 / 255.0;
                        let a = chunk[3] as f32 / 255.0;
                        data.push(Vec4::new(r, g, b, a));
                    }
                }
                Ok(Texture {
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
        let mut data = Vec::with_capacity((width * height) as usize);

        for y in 0..height {
            for x in 0..width {
                let is_checker = ((x / 8) + (y / 8)) % 2 == 0;
                let color = if is_checker { 1.0 } else { 0.5 };
                data.push(Vec4::new(color, color, color, 1.0));
            }
        }

        Texture {
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
