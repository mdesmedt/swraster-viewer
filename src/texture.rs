use crate::math::*;
use crate::util::*;
use dashmap::DashMap;
use glam::{UVec4, Vec2, Vec3A, Vec4};
use std::sync::{Arc, OnceLock};

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
    pub data: Vec<u32>, // RGBA8 data
    pub max_mip_level: u32,
    pub max_mip_level_vec: Vec4,
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

                        let p00 = self.load_vec4((slice_offset + (y0 * prev_width + x0)) as usize);
                        let p10 = self.load_vec4((slice_offset + (y0 * prev_width + x1)) as usize);
                        let p01 = self.load_vec4((slice_offset + (y1 * prev_width + x0)) as usize);
                        let p11 = self.load_vec4((slice_offset + (y1 * prev_width + x1)) as usize);

                        let avg = match self.texture_type {
                            TextureType::SRGB => {
                                let l00 = srgb_to_linear(p00);
                                let l10 = srgb_to_linear(p10);
                                let l01 = srgb_to_linear(p01);
                                let l11 = srgb_to_linear(p11);
                                let avg = (l00 + l10 + l01 + l11) / 4.0;
                                linear_to_srgb_vec4(avg)
                            }
                            TextureType::MetallicRoughness => {
                                let roughness =
                                    (p00.y * p00.y + p10.y * p10.y + p01.y * p01.y + p11.y * p11.y)
                                        / 4.0;
                                let metallic = (p00.z + p10.z + p01.z + p11.z) / 4.0;
                                Vec4::new(p00.x, roughness.sqrt(), metallic, p00.w)
                            }
                            TextureType::Normal => {
                                let n00 = p00 * 2.0 - 1.0;
                                let n10 = p10 * 2.0 - 1.0;
                                let n01 = p01 * 2.0 - 1.0;
                                let n11 = p11 * 2.0 - 1.0;
                                let avg = (n00 + n10 + n01 + n11) / 4.0;
                                (avg + 1.0) / 2.0
                            }
                            _ => (p00 + p10 + p01 + p11) / 4.0,
                        };

                        self.data.push(rgba8_pack_vec4(avg));
                    }
                }
            }

            self.mip_widths.push(mip_width);
            self.mip_heights.push(mip_height);
            self.mip_widths_f.push(mip_width as f32);
            self.mip_heights_f.push(mip_height as f32);
            self.max_mip_level += 1;
            self.max_mip_level_vec += Vec4::ONE;
        }
    }

    pub fn load_vec4(&self, idx: usize) -> Vec4 {
        rgba8_unpack_vec4(self.data[idx])
    }
}

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    (bits as f32) * 2.328_306_4e-10
}

fn hammersley(i: u32, n: u32) -> Vec2 {
    Vec2::new(i as f32 / n as f32, radical_inverse_vdc(i))
}

fn importance_sample_ggx(xi: Vec2, n: Vec3A, roughness: f32) -> Vec3A {
    let a = roughness * roughness;
    let phi = 2.0 * std::f32::consts::PI * xi.x;
    let cos_theta = ((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

    let h_tangent = Vec3A::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);

    let up = if n.z.abs() < 0.999 {
        Vec3A::Z
    } else {
        Vec3A::X
    };
    let tangent = n.cross(up).normalize();
    let bitangent = n.cross(tangent);

    (tangent * h_tangent.x + bitangent * h_tangent.y + n * h_tangent.z).normalize()
}

fn integrate_brdf(ndotv: f32, roughness: f32) -> Vec2 {
    let v = Vec3A::new((1.0 - ndotv * ndotv).max(0.0).sqrt(), 0.0, ndotv);
    let n = Vec3A::Z;
    let sample_count = 128;
    let mut a = 0.0;
    let mut b = 0.0;

    for i in 0..sample_count {
        let xi = hammersley(i, sample_count);
        let h = importance_sample_ggx(xi, n, roughness);
        let l = (h * (2.0 * v.dot(h)) - v).normalize();

        let ndotl = l.z.max(0.0);
        let ndoth = h.z.max(0.0);
        let vdoth = v.dot(h).max(0.0);

        if ndotl > 0.0 {
            let alpha = roughness * roughness;
            let k = (alpha + 1.0) * (alpha + 1.0) * 0.125;
            let g_v = ndotv / (ndotv * (1.0 - k) + k);
            let g_l = ndotl / (ndotl * (1.0 - k) + k);
            let g_vis = (g_v * g_l * vdoth / (ndoth * ndotv.max(1.0e-5))).max(0.0);
            let fc = (1.0 - vdoth).powi(5);

            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }

    Vec2::new(a / sample_count as f32, b / sample_count as f32)
}

pub fn generate_brdf_lut(size: u32) -> Texture {
    let mut data = Vec::with_capacity((size * size) as usize);
    let size_f = size as f32;

    for y in 0..size {
        let roughness = ((y as f32 + 0.5) / size_f).clamp(0.0, 1.0);
        for x in 0..size {
            let ndotv = ((x as f32 + 0.5) / size_f).clamp(0.0, 1.0);
            let integrated = integrate_brdf(ndotv.max(1.0e-4), roughness.max(1.0e-4));
            let texel = Vec4::new(
                integrated.x.clamp(0.0, 1.0),
                integrated.y.clamp(0.0, 1.0),
                0.0,
                1.0,
            );
            data.push(rgba8_pack_vec4(texel));
        }
    }

    let mut texture = Texture {
        width: size,
        height: size,
        width_height_vec: Vec4::new(size as f32, size as f32, size as f32, size as f32),
        texture_type: TextureType::Linear,
        data,
        max_mip_level: 0,
        max_mip_level_vec: Vec4::ZERO,
        mip_offsets: vec![0],
        mip_widths: vec![size],
        mip_heights: vec![size],
        mip_widths_f: vec![size as f32],
        mip_heights_f: vec![size as f32],
        array_stride: vec![0],
    };
    texture.generate_mipmaps();
    texture
}

fn cubemap_face_uv_to_direction(face: u32, u: f32, v: f32) -> Vec3A {
    let dir = match face {
        0 => Vec3A::new(1.0, -v, -u),  // +X
        1 => Vec3A::new(-1.0, -v, u),  // -X
        2 => Vec3A::new(u, 1.0, v),    // +Y
        3 => Vec3A::new(u, -1.0, -v),  // -Y
        4 => Vec3A::new(u, -v, 1.0),   // +Z
        _ => Vec3A::new(-u, -v, -1.0), // -Z
    };
    dir.normalize()
}

fn cubemap_direction_to_face_uv(n: Vec3A) -> (u32, f32, f32) {
    let ax = n.x.abs();
    let ay = n.y.abs();
    let az = n.z.abs();

    if ax >= ay && ax >= az {
        if n.x >= 0.0 {
            (0, (-n.z / ax) * 0.5 + 0.5, (-n.y / ax) * 0.5 + 0.5)
        } else {
            (1, (n.z / ax) * 0.5 + 0.5, (-n.y / ax) * 0.5 + 0.5)
        }
    } else if ay > ax && ay >= az {
        if n.y >= 0.0 {
            (2, (n.x / ay) * 0.5 + 0.5, (n.z / ay) * 0.5 + 0.5)
        } else {
            (3, (n.x / ay) * 0.5 + 0.5, (-n.z / ay) * 0.5 + 0.5)
        }
    } else if n.z >= 0.0 {
        (4, (n.x / az) * 0.5 + 0.5, (-n.y / az) * 0.5 + 0.5)
    } else {
        (5, (-n.x / az) * 0.5 + 0.5, (-n.y / az) * 0.5 + 0.5)
    }
}

fn sample_cubemap_direction_linear(cubemap: &TextureAndSampler, dir: Vec3A) -> Vec3A {
    let (face, u, v) = cubemap_direction_to_face_uv(dir);
    let sampled = cubemap.sample_bilinear_rgb(
        Vec4::splat(u.clamp(0.0, 1.0)),
        Vec4::splat(v.clamp(0.0, 1.0)),
        0,
        UVec4::splat(face),
    );
    let srgb = sampled.extract_lane(0);
    Vec3A::new(
        srgb_to_linear_scalar(srgb.x),
        srgb_to_linear_scalar(srgb.y),
        srgb_to_linear_scalar(srgb.z),
    )
}

pub fn compute_irradiance_sh4(cubemap: &TextureAndSampler) -> [Vec3A; 4] {
    let mut sh = [Vec3A::ZERO; 4];
    let width = cubemap.texture.width as f32;
    let height = cubemap.texture.height as f32;
    let texel_omega = (4.0 / width) * (4.0 / height);

    for face in 0..6 {
        for y in 0..cubemap.texture.height {
            let v = ((y as f32 + 0.5) / height) * 2.0 - 1.0;
            for x in 0..cubemap.texture.width {
                let u = ((x as f32 + 0.5) / width) * 2.0 - 1.0;
                let dir = cubemap_face_uv_to_direction(face, u, v);
                let weight = texel_omega / (1.0 + u * u + v * v).powf(1.5);
                let color = sample_cubemap_direction_linear(cubemap, dir);

                let x = dir.x;
                let y = dir.y;
                let z = dir.z;
                let basis = [0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x];

                for i in 0..4 {
                    sh[i] += color * (basis[i] * weight);
                }
            }
        }
    }

    // Convolve radiance SH to irradiance SH (Lambert)
    sh[0] *= std::f32::consts::PI;
    for coeff in &mut sh[1..4] {
        *coeff *= 2.0 * std::f32::consts::PI / 3.0;
    }

    // Fold SH basis constants into coefficients
    sh[0] *= 0.282095;
    sh[1] *= 0.488603;
    sh[2] *= 0.488603;
    sh[3] *= 0.488603;
    sh
}

pub fn generate_prefiltered_specular_cubemap(
    cubemap: &TextureAndSampler,
    sample_count: u32,
) -> Texture {
    let base_width = cubemap.texture.width;
    let base_height = cubemap.texture.height;
    let num_mips = 1 + (base_width.max(base_height)).ilog2();
    let max_mip = num_mips - 1;

    let mut data = Vec::new();
    let mut mip_offsets = Vec::with_capacity(num_mips as usize);
    let mut mip_widths = Vec::with_capacity(num_mips as usize);
    let mut mip_heights = Vec::with_capacity(num_mips as usize);
    let mut mip_widths_f = Vec::with_capacity(num_mips as usize);
    let mut mip_heights_f = Vec::with_capacity(num_mips as usize);
    let mut array_stride = Vec::with_capacity(num_mips as usize);

    for mip in 0..num_mips {
        let mip_width = (base_width >> mip).max(1);
        let mip_height = (base_height >> mip).max(1);
        let roughness = if max_mip > 0 {
            mip as f32 / max_mip as f32
        } else {
            0.0
        };

        mip_offsets.push(data.len() as u32);
        mip_widths.push(mip_width);
        mip_heights.push(mip_height);
        mip_widths_f.push(mip_width as f32);
        mip_heights_f.push(mip_height as f32);
        array_stride.push(mip_width * mip_height);

        for face in 0..6 {
            for y in 0..mip_height {
                let v = ((y as f32 + 0.5) / mip_height as f32) * 2.0 - 1.0;
                for x in 0..mip_width {
                    let u = ((x as f32 + 0.5) / mip_width as f32) * 2.0 - 1.0;
                    let r = cubemap_face_uv_to_direction(face, u, v);

                    let color = if mip == 0 {
                        sample_cubemap_direction_linear(cubemap, r)
                    } else {
                        let mut accum = Vec3A::ZERO;
                        let mut total_weight = 0.0;
                        for i in 0..sample_count {
                            let xi = hammersley(i, sample_count);
                            let h = importance_sample_ggx(xi, r, roughness.max(0.045));
                            let l = (h * (2.0 * r.dot(h)) - r).normalize();
                            let ndotl = r.dot(l).max(0.0);
                            if ndotl > 0.0 {
                                accum += sample_cubemap_direction_linear(cubemap, l) * ndotl;
                                total_weight += ndotl;
                            }
                        }
                        if total_weight > 0.0 {
                            accum / total_weight
                        } else {
                            sample_cubemap_direction_linear(cubemap, r)
                        }
                    };

                    data.push(rgba8_pack_vec4(Vec4::new(color.x, color.y, color.z, 1.0)));
                }
            }
        }
    }

    Texture {
        width: base_width,
        height: base_height,
        width_height_vec: Vec4::new(
            base_width as f32,
            base_width as f32,
            base_height as f32,
            base_height as f32,
        ),
        texture_type: TextureType::Linear,
        data,
        max_mip_level: max_mip,
        max_mip_level_vec: Vec4::splat(max_mip as f32),
        mip_offsets,
        mip_widths,
        mip_heights,
        mip_widths_f,
        mip_heights_f,
        array_stride,
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

        assert!(mip_level.x < 20);

        let width_i = self.texture.mip_widths.gather(mip_level);
        let width_f = self.texture.mip_widths_f.gather(mip_level);
        let height_f = self.texture.mip_heights_f.gather(mip_level);
        let offsets = self.texture.mip_offsets.gather(mip_level)
            + array_slice * self.texture.array_stride.gather(mip_level);

        let x = (u * (width_f - Vec4::ONE)).round();
        let y = (v * (height_f - Vec4::ONE)).round();
        let x = x.clamp(Vec4::ZERO, width_f - Vec4::ONE).as_uvec4();
        let y = y.clamp(Vec4::ZERO, height_f - Vec4::ONE).as_uvec4();

        let idx = offsets + y * width_i + x;
        self.gather_rgb(idx)
    }

    pub fn sample_cubemap_trilinear_rgb(&self, normal: Vec3x4, mip_level: Vec4) -> Vec3x4 {
        let mip = mip_level.clamp(Vec4::ZERO, self.texture.max_mip_level_vec);
        let mip0f = mip.floor();
        let mip1f = (mip0f + Vec4::ONE).min(self.texture.max_mip_level_vec);
        let t = mip - mip0f;

        let c0 = self.sample_cubemap_rgb(normal, mip0f.as_uvec4());
        let c1 = self.sample_cubemap_rgb(normal, mip1f.as_uvec4());
        Vec3x4::new(
            c0.x + (c1.x - c0.x) * t,
            c0.y + (c1.y - c0.y) * t,
            c0.z + (c1.z - c0.z) * t,
        )
    }

    pub fn sample4_rgb(&self, u_vec: Vec4, v_vec: Vec4, du_dv: Vec4) -> Vec3x4 {
        let mip_level = self.compute_mip_level(du_dv);
        let mip_usize = mip_level as usize;

        let width_i = UVec4::splat(self.texture.mip_widths[mip_usize]);
        let width_f = Vec4::splat(self.texture.mip_widths_f[mip_usize]);
        let height_f = Vec4::splat(self.texture.mip_heights_f[mip_usize]);
        let offsets = UVec4::splat(self.texture.mip_offsets[mip_usize] as u32);

        let x = Self::apply_wrap_mode((u_vec * width_f).floor(), width_f, &self.sampler.wrap_s)
            .as_uvec4();
        let y = Self::apply_wrap_mode((v_vec * height_f).floor(), height_f, &self.sampler.wrap_t)
            .as_uvec4();

        let idx = offsets + y * width_i + x;
        self.gather_rgb(idx)
    }

    pub fn sample4_alpha(&self, u_vec: Vec4, v_vec: Vec4, du_dv: Vec4) -> Vec4 {
        let mip_level = self.compute_mip_level(du_dv);
        let mip_usize = mip_level as usize;

        let width_i = UVec4::splat(self.texture.mip_widths[mip_usize]);
        let width_f = Vec4::splat(self.texture.mip_widths_f[mip_usize]);
        let height_f = Vec4::splat(self.texture.mip_heights_f[mip_usize]);
        let offsets = UVec4::splat(self.texture.mip_offsets[mip_usize] as u32);

        let x = Self::apply_wrap_mode((u_vec * width_f).floor(), width_f, &self.sampler.wrap_s)
            .as_uvec4();
        let y = Self::apply_wrap_mode((v_vec * height_f).floor(), height_f, &self.sampler.wrap_t)
            .as_uvec4();

        let idx = offsets + y * width_i + x;
        self.gather_alpha(idx)
    }

    fn gather_rgb(&self, idx: UVec4) -> Vec3x4 {
        let a = self.texture.load_vec4(idx.x as usize);
        let b = self.texture.load_vec4(idx.y as usize);
        let c = self.texture.load_vec4(idx.z as usize);
        let d = self.texture.load_vec4(idx.w as usize);

        Vec3x4 {
            x: Vec4::new(a.x, b.x, c.x, d.x),
            y: Vec4::new(a.y, b.y, c.y, d.y),
            z: Vec4::new(a.z, b.z, c.z, d.z),
        }
    }

    #[allow(dead_code)]
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
        let a = self.texture.load_vec4(idx.x as usize);
        let b = self.texture.load_vec4(idx.y as usize);
        let c = self.texture.load_vec4(idx.z as usize);
        let d = self.texture.load_vec4(idx.w as usize);

        Vec4::new(a.w, b.w, c.w, d.w)
    }

    #[allow(dead_code)]
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
    textures: DashMap<String, Arc<OnceLock<Arc<Texture>>>>,
    base_dir: Option<String>,
}

impl TextureCache {
    pub fn new(base_dir: String) -> Self {
        Self {
            textures: DashMap::new(),
            base_dir: Some(base_dir),
        }
    }

    pub fn get_or_create(&self, uri: &str, texture_type: TextureType) -> Arc<Texture> {
        let cell = self
            .textures
            .entry(uri.to_string())
            .or_insert_with(|| Arc::new(OnceLock::new()))
            .clone();

        cell.get_or_init(|| {
            Arc::new(match self.load_texture(uri, texture_type) {
                Ok(tex) => tex,
                Err(e) => {
                    panic!("Failed to load texture '{}': {}", uri, e);
                }
            })
        })
        .clone()
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

                                let r = raw_data[src_idx];
                                let g = raw_data[src_idx + 1];
                                let b = raw_data[src_idx + 2];
                                let a = raw_data[src_idx + 3];
                                data.push(rgba8_pack_u8(r, g, b, a));
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
                        max_mip_level_vec: Vec4::ZERO,
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
                        let r = chunk[0];
                        let g = chunk[1];
                        let b = chunk[2];
                        let a = chunk[3];
                        data.push(rgba8_pack_u8(r, g, b, a));
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
                        max_mip_level_vec: Vec4::ZERO,
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

    pub fn unique_texture_count(&self) -> usize {
        self.textures.len()
    }

    pub fn total_texture_data_size(&self) -> usize {
        self.textures
            .iter()
            .map(|item| item.value().get().unwrap().data.len() * std::mem::size_of::<u32>())
            .sum()
    }
}
