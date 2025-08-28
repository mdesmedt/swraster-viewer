use crate::math::*;
use glam::{Mat4, Vec4};
use std::collections::HashMap;
use std::sync::Arc;

use crate::scene::{SceneError, SceneResult};

pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub width_height_vec: Vec4,
    pub data: Vec<Vec4>, // Vec4 floating point data
    pub max_mip_level: u32,
    pub mip_offsets: Vec<usize>,
    pub mip_widths: Vec<u32>,
    pub mip_heights: Vec<u32>,
    pub mip_widths_f: Vec<f32>,
    pub mip_heights_f: Vec<f32>,
}

impl Texture {
    pub fn generate_mipmaps(&mut self) {
        assert!(self.max_mip_level == 0, "Texture already has mipmaps");
        assert!(self.mip_offsets[0] == 0, "Wrong mip 0 offset");

        let width = self.width;
        let height = self.height;
        let num_mips = 1 + (width.max(height) as u32).ilog2();

        for mip in 1..num_mips {
            self.mip_offsets.push(self.data.len());
            let mip_width = (width >> mip).max(1);
            let mip_height = (height >> mip).max(1);

            let prev_offset = self.mip_offsets[(mip - 1) as usize];
            let prev_width = self.mip_widths[(mip - 1) as usize];
            let prev_height = self.mip_heights[(mip - 1) as usize];

            for y in 0..mip_height {
                for x in 0..mip_width {
                    let x0 = x * 2;
                    let y0 = y * 2;
                    let x1 = (x0 + 1).min(prev_width - 1);
                    let y1 = (y0 + 1).min(prev_height - 1);

                    let p00 = self.data[prev_offset + (y0 * prev_width + x0) as usize];
                    let p10 = self.data[prev_offset + (y0 * prev_width + x1) as usize];
                    let p01 = self.data[prev_offset + (y1 * prev_width + x0) as usize];
                    let p11 = self.data[prev_offset + (y1 * prev_width + x1) as usize];

                    let avg = (p00 + p10 + p01 + p11) / 4.0;
                    self.data.push(avg);
                }
            }

            self.mip_widths.push(mip_width);
            self.mip_heights.push(mip_height);
            self.mip_widths_f.push(mip_width as f32);
            self.mip_heights_f.push(mip_height as f32);
            self.max_mip_level += 1;
        }
    }
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

    /// Convert normals into cubemap UVs
    /// Cubemap layout:
    ///     +Y
    /// -X  +Z  +X  -Z
    ///     -Y
    pub fn cubemap_uv_from_normal(n: Vec3x4) -> (Vec4, Vec4) {
        let ax = n.x.abs();
        let ay = n.y.abs();
        let az = n.z.abs();

        // Major-axis masks
        let mx = ax.cmpge(ay) & ax.cmpge(az); // X dominant
        let my = ay.cmpgt(ax) & ay.cmpge(az); // Y dominant

        // Signs as +1 / -1
        let sx = Vec4::select(n.x.cmpge(Vec4::ZERO), Vec4::ONE, Vec4::NEG_ONE);
        let sy = Vec4::select(n.y.cmpge(Vec4::ZERO), Vec4::ONE, Vec4::NEG_ONE);
        let sz = Vec4::select(n.z.cmpge(Vec4::ZERO), Vec4::ONE, Vec4::NEG_ONE);

        // Per-axis candidate (u,v) and denominator
        // X faces:  den=ax, u=-z*sx, v=-y
        let u_x = -n.z * sx;
        let v_x = -n.y;
        let d_x = ax;

        // Y faces:  den=ay, u= x,    v= z*sy
        let u_y = n.x;
        let v_y = n.z * sy;
        let d_y = ay;

        // Z faces:  den=az, u= x*sz, v=-y
        let u_z = n.x * sz;
        let v_z = -n.y;
        let d_z = az;

        // Select the axis
        let u = Vec4::select(mx, u_x, Vec4::select(my, u_y, u_z));
        let v = Vec4::select(mx, v_x, Vec4::select(my, v_y, v_z));
        let mut denom = Vec4::select(mx, d_x, Vec4::select(my, d_y, d_z));

        // Safety against divide-by-zero on degenerate inputs
        denom = denom.max(Vec4::splat(1.0e-19));

        // Face-local UV in [-1,1] -> [0,1]
        let uf = 0.5 * (u / denom + 1.0);
        let vf = 0.5 * (v / denom + 1.0);

        // Face coordinates for cross layout
        // +X → (2,1), -X → (0,1), +Y → (1,0), -Y → (1,2), +Z → (1,1), -Z → (3,1)
        let fx_x = Vec4::select(n.x.cmpge(Vec4::ZERO), Vec4::splat(2.0), Vec4::ZERO);
        let fy_x = Vec4::ONE;

        let fx_y = Vec4::ONE;
        let fy_y = Vec4::select(n.y.cmpge(Vec4::ZERO), Vec4::ZERO, Vec4::splat(2.0));

        let fx_z = Vec4::select(n.z.cmpge(Vec4::ZERO), Vec4::ONE, Vec4::splat(3.0));
        let fy_z = Vec4::ONE;

        let fx = Vec4::select(mx, fx_x, Vec4::select(my, fx_y, fx_z));
        let fy = Vec4::select(mx, fy_x, Vec4::select(my, fy_y, fy_z));

        // Transform face + face-local UV to final UV
        let u_out = (fx + uf) * (1.0 / 4.0);
        let v_out = (fy + vf) * (1.0 / 3.0);

        (u_out, v_out)
    }

    pub fn sample_cubemap(&self, normal: Vec3x4) -> Mat4 {
        // Compute UVs
        let (u, v) = Self::cubemap_uv_from_normal(normal);

        // Sample four texels
        let mut texels = Mat4::ZERO;
        for i in 0..4 {
            *texels.col_mut(i) = self.sample_bilinear(u[i], v[i], 0);
        }
        // Return the samples in columns of X, Y, Z and W
        texels.transpose()
    }

    pub fn sample4(&self, u_vec: Vec4, v_vec: Vec4, du_dv: Vec4) -> Mat4 {
        // Sample four texels
        let mut texels = Mat4::ZERO;
        let mip_level = self.compute_mip_level(du_dv);
        //println!("mip_level: {}", mip_level);
        for i in 0..4 {
            // Apply wrap modes to UV coordinates
            let u = Self::apply_wrap_mode(u_vec[i], &self.sampler.wrap_s);
            let v = Self::apply_wrap_mode(v_vec[i], &self.sampler.wrap_t);

            *texels.col_mut(i) = self.sample_bilinear(u, v, mip_level);
        }
        // Return the samples in columns of X, Y, Z and W
        texels.transpose()
    }

    pub fn sample_point(&self, u: f32, v: f32, mip_level: u32) -> Vec4 {
        let width = self.texture.mip_widths[mip_level as usize];
        let width_f = self.texture.mip_widths_f[mip_level as usize];
        let height_f = self.texture.mip_heights_f[mip_level as usize];

        // Texel coordinates
        let x_f = u * width_f - 0.5;
        let y_f = v * height_f - 0.5;

        let x = x_f as u32;
        let y = y_f as u32;

        let offset = self.texture.mip_offsets[mip_level as usize] + (y * width + x) as usize;
        self.texture.data[offset]
    }

    pub fn sample_bilinear(&self, u: f32, v: f32, mip_level: u32) -> Vec4 {
        let width = self.texture.mip_widths[mip_level as usize];
        let height = self.texture.mip_heights[mip_level as usize];
        let width_f = self.texture.mip_widths_f[mip_level as usize];
        let height_f = self.texture.mip_heights_f[mip_level as usize];

        // Texel coordinates
        let x_f = u * width_f - 0.5;
        let y_f = v * height_f - 0.5;

        // Sample four texels
        let offset = self.texture.mip_offsets[mip_level as usize];
        let x0 = x_f.floor() as u32;
        let y0 = y_f.floor() as u32;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);
        let p00 = self.texture.data[offset + (y0 * width + x0) as usize];
        let p10 = self.texture.data[offset + (y0 * width + x1) as usize];
        let p01 = self.texture.data[offset + (y1 * width + x0) as usize];
        let p11 = self.texture.data[offset + (y1 * width + x1) as usize];

        // Bilinear filter
        let fx = x_f - x0 as f32;
        let fy = y_f - y0 as f32;
        let lerp_x0 = p00 * (1.0 - fx) + p10 * fx;
        let lerp_x1 = p01 * (1.0 - fx) + p11 * fx;
        let final_color = lerp_x0 * (1.0 - fy) + lerp_x1 * fy;

        final_color
    }

    pub fn compute_mip_level(&self, du_dv: Vec4) -> u32 {
        let du_dv_tex = du_dv * self.texture.width_height_vec;

        let dx2 = du_dv_tex.x * du_dv_tex.x + du_dv_tex.z * du_dv_tex.z;
        let dy2 = du_dv_tex.y * du_dv_tex.y + du_dv_tex.w * du_dv_tex.w;

        // Hack to preserve some sharpness on sloped surfaces
        //let footprint = dx2.max(dy2);
        let footprint = (dx2 + dy2) * 0.5;

        let mip = (footprint.max(1.0) as u32).ilog2() >> 1;
        mip.min(self.texture.max_mip_level)
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
        let mut texture = match image::open(&texture_path) {
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
                    width_height_vec: Vec4::new(
                        width as f32,
                        width as f32,
                        height as f32,
                        height as f32,
                    ),
                    data,
                    max_mip_level: 0,
                    mip_offsets: vec![0],
                    mip_widths: vec![width],
                    mip_heights: vec![height],
                    mip_widths_f: vec![width as f32],
                    mip_heights_f: vec![height as f32],
                })
            }
            Err(e) => Err(SceneError::MissingData(format!(
                "Could not load texture '{}' from path '{}': {}",
                uri, texture_path, e
            ))),
        };

        if let Ok(texture) = &mut texture {
            texture.generate_mipmaps();
        }

        texture
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
            width_height_vec: Vec4::new(width as f32, width as f32, height as f32, height as f32),
            data,
            max_mip_level: 0,
            mip_offsets: vec![0],
            mip_widths: vec![width],
            mip_heights: vec![height],
            mip_widths_f: vec![width as f32],
            mip_heights_f: vec![height as f32],
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
