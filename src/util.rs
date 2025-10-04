use crate::math::*;
use glam::{BVec4, BVec4A, UVec4, Vec3A, Vec4};

pub fn bvec4_to_bvec4a(b: BVec4) -> BVec4A {
    let arr: [bool; 4] = b.into();
    BVec4A::from(arr)
}

pub fn bvec4a_to_bvec4(b: BVec4A) -> BVec4 {
    let arr: [bool; 4] = b.into();
    BVec4::from(arr)
}

#[allow(dead_code)]
pub fn tonemap_aces(color: Vec3x4) -> Vec3x4 {
    let a = Vec4::splat(2.51);
    let b = Vec4::splat(0.03);
    let c = Vec4::splat(2.43);
    let d = Vec4::splat(0.59);
    let e = Vec4::splat(0.14);

    let map_channel = |v: Vec4| {
        let numerator = v * (a * v + b);
        let denominator = v * (c * v + d) + e;
        (numerator / denominator).clamp(Vec4::ZERO, Vec4::ONE)
    };

    Vec3x4 {
        x: map_channel(color.x),
        y: map_channel(color.y),
        z: map_channel(color.z),
    }
}

pub fn tonemap(color: Vec3x4) -> Vec3x4 {
    // Contrast control
    let k = 0.2;
    color / (color + k) * (1.0 + k)
}

pub fn srgb_to_linear_scalar(scalar: f32) -> f32 {
    if scalar <= 0.04045 {
        scalar / 12.92
    } else {
        f32::powf((scalar + 0.055) / 1.055, 2.4)
    }
}

pub fn srgb_to_linear(color: Vec4) -> Vec4 {
    Vec4::new(
        srgb_to_linear_scalar(color.x),
        srgb_to_linear_scalar(color.y),
        srgb_to_linear_scalar(color.z),
        color.w,
    )
}

pub fn linear_to_srgb_scalar(scalar: f32) -> f32 {
    if scalar <= 0.0031308 {
        scalar * 12.92
    } else {
        f32::powf(scalar, 1.0 / 2.4) * 1.055 - 0.055
    }
}

pub fn linear_to_srgb_vec4(color: Vec4) -> Vec4 {
    Vec4::new(
        linear_to_srgb_scalar(color.x),
        linear_to_srgb_scalar(color.y),
        linear_to_srgb_scalar(color.z),
        color.w,
    )
}

pub fn rgba8_unpack_vec4(rgba8: u32) -> Vec4 {
    let r = ((rgba8 >> 24) & 0xFF) as f32 / 255.0;
    let g = ((rgba8 >> 16) & 0xFF) as f32 / 255.0;
    let b = ((rgba8 >> 8) & 0xFF) as f32 / 255.0;
    let a = ((rgba8 >> 0) & 0xFF) as f32 / 255.0;
    Vec4::new(r, g, b, a)
}

pub fn rgba8_pack_vec4(color: Vec4) -> u32 {
    (((color.x * 255.0) as u32) << 24)
        | (((color.y * 255.0) as u32) << 16)
        | (((color.z * 255.0) as u32) << 8)
        | (((color.w * 255.0) as u32) << 0)
}

pub fn rgba8_pack_u8(r: u8, g: u8, b: u8, a: u8) -> u32 {
    ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (a as u32)
}

pub fn srgb_to_linear_fast(x: Vec3x4) -> Vec3x4 {
    return x * (x * (x * 0.305306011 + 0.682171111) + 0.012522878);
}

// Gather functions for Vec<f32> and Vec<u32>

pub trait GatherU32 {
    fn gather(&self, indices: UVec4) -> UVec4;
}

pub trait GatherF32 {
    fn gather(&self, indices: UVec4) -> Vec4;
}

impl GatherU32 for Vec<u32> {
    #[inline(always)]
    fn gather(&self, indices: UVec4) -> UVec4 {
        UVec4::new(
            self[indices.x as usize],
            self[indices.y as usize],
            self[indices.z as usize],
            self[indices.w as usize],
        )
    }
}

impl GatherF32 for Vec<f32> {
    #[inline(always)]
    fn gather(&self, indices: UVec4) -> Vec4 {
        Vec4::new(
            self[indices.x as usize],
            self[indices.y as usize],
            self[indices.z as usize],
            self[indices.w as usize],
        )
    }
}

pub struct ScalarInterpolator {
    pub a: f32,
    pub da: f32,
    pub db: f32,
}

impl ScalarInterpolator {
    pub fn new(a: f32, b: f32, c: f32) -> Self {
        let da = b - a;
        let db = c - a;
        Self { a, da, db }
    }

    pub fn from_array(arr: [f32; 3]) -> Self {
        let a = arr[0];
        let da = arr[1] - a;
        let db = arr[2] - a;
        Self { a, da, db }
    }

    pub fn interpolate(&self, bary1: Vec4, bary2: Vec4) -> Vec4 {
        self.a + bary1 * self.da + bary2 * self.db
    }
}

pub struct Vec3Interpolator {
    pub a: Vec3A,
    pub da: Vec3A,
    pub db: Vec3A,
}

impl Vec3Interpolator {
    pub fn from_array(arr: [Vec3A; 3]) -> Self {
        let a = arr[0];
        let da = arr[1] - a;
        let db = arr[2] - a;
        Self { a, da, db }
    }

    pub fn interpolate(&self, bary1: Vec4, bary2: Vec4) -> Vec3x4 {
        let x = self.a.x + bary1 * self.da.x + bary2 * self.db.x;
        let y = self.a.y + bary1 * self.da.y + bary2 * self.db.y;
        let z = self.a.z + bary1 * self.da.z + bary2 * self.db.z;
        Vec3x4::new(x, y, z)
    }
}
