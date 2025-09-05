use crate::math::*;
use glam::{UVec4, Vec3A, Vec4};
use std::collections::HashMap;
use std::sync::Arc;

use crate::scene::{SceneError, SceneResult};

#[derive(Clone, Copy, PartialEq)]
pub enum TextureType {
    SRGB,
    Normal,
    MetallicRoughness,
    Cubemap,
    Linear,
}

pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub width_height_vec: Vec4,
    pub texture_type: TextureType,
    pub data: Vec<Vec4>, // Vec4 floating point data
    pub max_mip_level: u32,
    pub mip_offsets: Vec<u32>,
    pub mip_widths: Vec<u32>,
    pub mip_heights: Vec<u32>,
    pub mip_widths_f: Vec<f32>,
    pub mip_heights_f: Vec<f32>,
    pub array_stride: Vec<u32>, // Stride in texels between faces for each mip level
}

impl Texture {
    pub fn generate_mipmaps(&mut self) {
        assert!(self.max_mip_level == 0, "Texture already has mipmaps");
        assert!(self.mip_offsets[0] == 0, "Wrong mip 0 offset");

        let width = self.width;
        let height = self.height;
        let num_mips = 1 + (width.max(height) as u32).ilog2();

        // Array size: 1 for regular textures, 6 for cubemaps
        let array_size = if self.texture_type == TextureType::Cubemap {
            6
        } else {
            1
        };

        for mip in 1..num_mips {
            self.mip_offsets.push(self.data.len() as u32);
            let mip_width = (width >> mip).max(1);
            let mip_height = (height >> mip).max(1);

            // Update array stride for this mip level
            self.array_stride.push(mip_width * mip_height);

            let prev_offset = self.mip_offsets[(mip - 1) as usize];
            let prev_width = self.mip_widths[(mip - 1) as usize];
            let prev_height = self.mip_heights[(mip - 1) as usize];
            let prev_array_stride = self.array_stride[(mip - 1) as usize];

            // Generate mip for each array slice (face for cubemaps, single slice for regular textures)
            for slice in 0..array_size {
                let slice_offset = prev_offset + (slice * prev_array_stride);

                for y in 0..mip_height {
                    for x in 0..mip_width {
                        let x0 = x * 2;
                        let y0 = y * 2;
                        let x1 = (x0 + 1).min(prev_width - 1);
                        let y1 = (y0 + 1).min(prev_height - 1);

                        let p00 = self.data[(slice_offset + (y0 * prev_width + x0)) as usize];
                        let p10 = self.data[(slice_offset + (y0 * prev_width + x1)) as usize];
                        let p01 = self.data[(slice_offset + (y1 * prev_width + x0)) as usize];
                        let p11 = self.data[(slice_offset + (y1 * prev_width + x1)) as usize];

                        let avg = match self.texture_type {
                            TextureType::SRGB => {
                                let l00 = p00 * p00;
                                let l10 = p10 * p10;
                                let l01 = p01 * p01;
                                let l11 = p11 * p11;
                                let avg = (l00 + l10 + l01 + l11) / 4.0;
                                sqrt_vec(avg)
                            }
                            TextureType::Normal => {
                                let n00 = Vec3A::from_vec4(p00) * 2.0 - Vec3A::ONE;
                                let n10 = Vec3A::from_vec4(p10) * 2.0 - Vec3A::ONE;
                                let n01 = Vec3A::from_vec4(p01) * 2.0 - Vec3A::ONE;
                                let n11 = Vec3A::from_vec4(p11) * 2.0 - Vec3A::ONE;
                                let normal = (n00 + n10 + n01 + n11).normalize();
                                let tangent_space = (normal + Vec3A::ONE) * Vec3A::splat(0.5);
                                tangent_space.extend(0.0)
                            }
                            TextureType::MetallicRoughness => {
                                let metallic = (p00.y + p10.y + p01.y + p11.y) / 4.0;
                                let roughness =
                                    (p00.z * p00.z + p10.z * p10.z + p01.z * p01.z + p11.z * p11.z)
                                        / 4.0;
                                Vec4::new(p00.x, metallic, roughness.sqrt(), p00.w)
                            }
                            _ => (p00 + p10 + p01 + p11) / 4.0,
                        };

                        self.data.push(avg);
                    }
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
    fn apply_wrap_mode(texel: Vec4, dim: Vec4, mode: &WrapMode) -> Vec4 {
        let texel2 = match mode {
            WrapMode::ClampToEdge => texel,
            WrapMode::Repeat => texel - (texel / dim).floor() * dim,
            WrapMode::MirroredRepeat => {
                let two = dim * Vec4::splat(2.0);
                let t = texel - (texel / two).floor() * two;
                t.min(two - t)
            }
        };
        texel2.min(dim - Vec4::splat(1.0))
    }

    /// Convert normals into cubemap UVs and array slice
    /// +X=0, -X=1, +Y=2, -Y=3, +Z=4, -Z=5
    pub fn cubemap_uv_from_normal(n: Vec3x4) -> (Vec4, Vec4, UVec4) {
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

        // Array indices: +X=0, -X=1, +Y=2, -Y=3, +Z=4, -Z=5
        let idx_x = Vec4::select(n.x.cmpge(Vec4::ZERO), Vec4::ZERO, Vec4::ONE); // +X=0, -X=1
        let idx_y = Vec4::select(n.y.cmpge(Vec4::ZERO), Vec4::splat(2.0), Vec4::splat(3.0)); // +Y=2, -Y=3
        let idx_z = Vec4::select(n.z.cmpge(Vec4::ZERO), Vec4::splat(4.0), Vec4::splat(5.0)); // +Z=4, -Z=5

        let array_slice = Vec4::select(mx, idx_x, Vec4::select(my, idx_y, idx_z));

        // Return face-local UV coordinates and array slice
        (uf, vf, array_slice.as_uvec4())
    }

    pub fn sample_cubemap_rgb(&self, normal: Vec3x4, mip_level: UVec4) -> Vec3x4 {
        let (u, v, array_slice) = Self::cubemap_uv_from_normal(normal);

        let width_i = self.texture.mip_widths.gather(mip_level);
        let width_f = self.texture.mip_widths_f.gather(mip_level);
        let height_f = self.texture.mip_heights_f.gather(mip_level);
        let offsets = self.texture.mip_offsets.gather(mip_level)
            + array_slice * self.texture.array_stride.gather(mip_level);

        // Convert to texel coordinates
        // Built-in to this logic is a half-texel offset so u,v maps to
        // [0.5/DIM, 1 - 0.5/DIM] as a hack to hide edge discontinuities
        let x_f = u * (width_f - Vec4::ONE);
        let y_f = v * (height_f - Vec4::ONE);

        let x0 = x_f.floor();
        let y0 = y_f.floor();
        let x1 = x0 + Vec4::ONE;
        let y1 = y0 + Vec4::ONE;

        let fx = x_f - x0;
        let fy = y_f - y0;
        let one_minus_fx = Vec4::ONE - fx;
        let one_minus_fy = Vec4::ONE - fy;

        let x0 = x0.clamp(Vec4::ZERO, width_f);
        let y0 = y0.clamp(Vec4::ZERO, height_f);
        let x1 = x1.clamp(Vec4::ZERO, width_f);
        let y1 = y1.clamp(Vec4::ZERO, height_f);

        let x0_i = x0.as_uvec4();
        let y0_i = y0.as_uvec4();
        let x1_i = x1.as_uvec4();
        let y1_i = y1.as_uvec4();

        let idx00 = offsets + y0_i * width_i + x0_i;
        let idx10 = offsets + y0_i * width_i + x1_i;
        let idx01 = offsets + y1_i * width_i + x0_i;
        let idx11 = offsets + y1_i * width_i + x1_i;

        let p00 = self.gather_rgb(idx00);
        let p10 = self.gather_rgb(idx10);
        let p01 = self.gather_rgb(idx01);
        let p11 = self.gather_rgb(idx11);

        let w00 = one_minus_fx * one_minus_fy;
        let w10 = fx * one_minus_fy;
        let w01 = one_minus_fx * fy;
        let w11 = fx * fy;

        Vec3x4 {
            x: p00.x * w00 + p10.x * w10 + p01.x * w01 + p11.x * w11,
            y: p00.y * w00 + p10.y * w10 + p01.y * w01 + p11.y * w11,
            z: p00.z * w00 + p10.z * w10 + p01.z * w01 + p11.z * w11,
        }
    }

    pub fn sample4_rgb(&self, u_vec: Vec4, v_vec: Vec4, du_dv: Vec4) -> Vec3x4 {
        let mip_level = self.compute_mip_level(du_dv);
        self.sample_bilinear_rgb(u_vec, v_vec, mip_level, UVec4::ZERO)
    }

    pub fn sample4_alpha(&self, u_vec: Vec4, v_vec: Vec4, du_dv: Vec4) -> Vec4 {
        let mip_level = self.compute_mip_level(du_dv);
        self.sample_bilinear_alpha(u_vec, v_vec, mip_level)
    }

    fn gather_rgb(&self, idx: UVec4) -> Vec3x4 {
        let a = self.texture.data[idx.x as usize];
        let b = self.texture.data[idx.y as usize];
        let c = self.texture.data[idx.z as usize];
        let d = self.texture.data[idx.w as usize];

        Vec3x4 {
            x: Vec4::new(a.x, b.x, c.x, d.x),
            y: Vec4::new(a.y, b.y, c.y, d.y),
            z: Vec4::new(a.z, b.z, c.z, d.z),
        }
    }

    pub fn sample_bilinear_rgb(
        &self,
        u: Vec4,
        v: Vec4,
        mip_level: u32,
        array_slice: UVec4,
    ) -> Vec3x4 {
        let mip_usize = mip_level as usize;

        let width_i = UVec4::splat(self.texture.mip_widths[mip_usize]);
        let width_f = Vec4::splat(self.texture.mip_widths_f[mip_usize]);
        let height_f = Vec4::splat(self.texture.mip_heights_f[mip_usize]);
        let offsets = UVec4::splat(self.texture.mip_offsets[mip_usize] as u32)
            + array_slice * UVec4::splat(self.texture.array_stride[mip_usize]);

        let x_f = u * width_f - Vec4::splat(0.5);
        let y_f = v * height_f - Vec4::splat(0.5);

        let x0 = x_f.floor();
        let y0 = y_f.floor();
        let x1 = x0 + Vec4::ONE;
        let y1 = y0 + Vec4::ONE;

        let fx = x_f - x0;
        let fy = y_f - y0;
        let one_minus_fx = Vec4::ONE - fx;
        let one_minus_fy = Vec4::ONE - fy;

        // Apply wrap mode

        let x0 = Self::apply_wrap_mode(x0, width_f, &self.sampler.wrap_s);
        let y0 = Self::apply_wrap_mode(y0, height_f, &self.sampler.wrap_t);
        let x1 = Self::apply_wrap_mode(x1, width_f, &self.sampler.wrap_s);
        let y1 = Self::apply_wrap_mode(y1, height_f, &self.sampler.wrap_t);

        let x0_i = x0.as_uvec4();
        let y0_i = y0.as_uvec4();
        let x1_i = x1.as_uvec4();
        let y1_i = y1.as_uvec4();

        let idx00 = offsets + y0_i * width_i + x0_i;
        let idx10 = offsets + y0_i * width_i + x1_i;
        let idx01 = offsets + y1_i * width_i + x0_i;
        let idx11 = offsets + y1_i * width_i + x1_i;

        let p00 = self.gather_rgb(idx00);
        let p10 = self.gather_rgb(idx10);
        let p01 = self.gather_rgb(idx01);
        let p11 = self.gather_rgb(idx11);

        let w00 = one_minus_fx * one_minus_fy;
        let w10 = fx * one_minus_fy;
        let w01 = one_minus_fx * fy;
        let w11 = fx * fy;

        Vec3x4 {
            x: p00.x * w00 + p10.x * w10 + p01.x * w01 + p11.x * w11,
            y: p00.y * w00 + p10.y * w10 + p01.y * w01 + p11.y * w11,
            z: p00.z * w00 + p10.z * w10 + p01.z * w01 + p11.z * w11,
        }
    }

    fn gather_alpha(&self, idx: UVec4) -> Vec4 {
        let a = self.texture.data[idx.x as usize];
        let b = self.texture.data[idx.y as usize];
        let c = self.texture.data[idx.z as usize];
        let d = self.texture.data[idx.w as usize];

        Vec4::new(a.w, b.w, c.w, d.w)
    }

    pub fn sample_bilinear_alpha(&self, u: Vec4, v: Vec4, mip_level: u32) -> Vec4 {
        let mip_usize = mip_level as usize;

        let width_i = UVec4::splat(self.texture.mip_widths[mip_usize]);
        let width_f = Vec4::splat(self.texture.mip_widths_f[mip_usize]);
        let height_f = Vec4::splat(self.texture.mip_heights_f[mip_usize]);
        let offsets = UVec4::splat(self.texture.mip_offsets[mip_usize] as u32);

        let x_f = u * width_f - Vec4::splat(0.5);
        let y_f = v * height_f - Vec4::splat(0.5);

        let x0 = x_f.floor();
        let y0 = y_f.floor();
        let x1 = x0 + Vec4::ONE;
        let y1 = y0 + Vec4::ONE;

        let fx = x_f - x0;
        let fy = y_f - y0;
        let one_minus_fx = Vec4::ONE - fx;
        let one_minus_fy = Vec4::ONE - fy;

        // Apply wrap mode

        let x0 = Self::apply_wrap_mode(x0, width_f, &self.sampler.wrap_s);
        let y0 = Self::apply_wrap_mode(y0, height_f, &self.sampler.wrap_t);
        let x1 = Self::apply_wrap_mode(x1, width_f, &self.sampler.wrap_s);
        let y1 = Self::apply_wrap_mode(y1, height_f, &self.sampler.wrap_t);

        let x0_i = x0.as_uvec4();
        let y0_i = y0.as_uvec4();
        let x1_i = x1.as_uvec4();
        let y1_i = y1.as_uvec4();

        let idx00 = offsets + y0_i * width_i + x0_i;
        let idx10 = offsets + y0_i * width_i + x1_i;
        let idx01 = offsets + y1_i * width_i + x0_i;
        let idx11 = offsets + y1_i * width_i + x1_i;

        let p00 = self.gather_alpha(idx00);
        let p10 = self.gather_alpha(idx10);
        let p01 = self.gather_alpha(idx01);
        let p11 = self.gather_alpha(idx11);

        let lerp_x0 = p00 * one_minus_fx + p10 * fx;
        let lerp_x1 = p01 * one_minus_fx + p11 * fx;

        lerp_x0 * one_minus_fy + lerp_x1 * fy
    }

    pub fn compute_mip_level(&self, du_dv: Vec4) -> u32 {
        let du_dv_tex = du_dv * self.texture.width_height_vec;

        let dx2 = du_dv_tex.x * du_dv_tex.x + du_dv_tex.z * du_dv_tex.z;
        let dy2 = du_dv_tex.y * du_dv_tex.y + du_dv_tex.w * du_dv_tex.w;

        // Hack to preserve some sharpness on sloped surfaces, normal code is:
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

    pub fn get_or_create(&mut self, uri: &str, texture_type: TextureType) -> Arc<Texture> {
        if let Some(texture) = self.textures.get(uri) {
            Arc::clone(texture)
        } else {
            // Load the texture from file
            let texture = match self.load_texture(uri, texture_type) {
                Ok(tex) => Arc::new(tex),
                Err(e) => {
                    eprintln!("Failed to load texture '{}': {}", uri, e);
                    // Create a fallback texture (checkerboard pattern)
                    Arc::new(self.create_fallback_texture(texture_type))
                }
            };
            self.textures.insert(uri.to_string(), Arc::clone(&texture));
            texture
        }
    }

    fn load_texture(&self, uri: &str, texture_type: TextureType) -> SceneResult<Texture> {
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

                if texture_type == TextureType::Cubemap {
                    // Input cubemap layout:
                    //     +Y
                    // -X  +Z  +X  -Z
                    //     -Y
                    let face_width = width / 4;
                    let face_height = height / 3;
                    let mut data = Vec::with_capacity((face_width * face_height * 6) as usize);

                    let face_positions = [(2, 1), (0, 1), (1, 0), (1, 2), (1, 1), (3, 1)];

                    for (face_x, face_y) in face_positions.iter() {
                        let start_x = face_x * face_width;
                        let start_y = face_y * face_height;

                        for y in 0..face_height {
                            for x in 0..face_width {
                                let src_x = start_x + x;
                                let src_y = start_y + y;
                                let src_idx = (src_y * width + src_x) as usize * 4;

                                let r = raw_data[src_idx] as f32 / 255.0;
                                let g = raw_data[src_idx + 1] as f32 / 255.0;
                                let b = raw_data[src_idx + 2] as f32 / 255.0;
                                let a = raw_data[src_idx + 3] as f32 / 255.0;
                                data.push(Vec4::new(r, g, b, a));
                            }
                        }
                    }

                    Ok(Texture {
                        width: face_width,
                        height: face_height,
                        width_height_vec: Vec4::new(
                            face_width as f32,
                            face_width as f32,
                            face_height as f32,
                            face_height as f32,
                        ),
                        texture_type,
                        data,
                        max_mip_level: 0,
                        mip_offsets: vec![0],
                        mip_widths: vec![face_width],
                        mip_heights: vec![face_height],
                        mip_widths_f: vec![face_width as f32],
                        mip_heights_f: vec![face_height as f32],
                        array_stride: vec![face_width * face_height],
                    })
                } else {
                    // 2D texture
                    let mut data = Vec::with_capacity((width * height) as usize);
                    for chunk in raw_data.chunks_exact(4) {
                        let r = chunk[0] as f32 / 255.0;
                        let g = chunk[1] as f32 / 255.0;
                        let b = chunk[2] as f32 / 255.0;
                        let a = chunk[3] as f32 / 255.0;
                        data.push(Vec4::new(r, g, b, a));
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
                        texture_type,
                        data,
                        max_mip_level: 0,
                        mip_offsets: vec![0],
                        mip_widths: vec![width],
                        mip_heights: vec![height],
                        mip_widths_f: vec![width as f32],
                        mip_heights_f: vec![height as f32],
                        array_stride: vec![0],
                    })
                }
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

    fn create_fallback_texture(&self, texture_type: TextureType) -> Texture {
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
            texture_type,
            max_mip_level: 0,
            mip_offsets: vec![0],
            mip_widths: vec![width],
            mip_heights: vec![height],
            mip_widths_f: vec![width as f32],
            mip_heights_f: vec![height as f32],
            array_stride: vec![if texture_type == TextureType::Cubemap {
                width * height
            } else {
                0
            }],
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
