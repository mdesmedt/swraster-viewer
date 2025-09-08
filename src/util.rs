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

#[allow(dead_code)]
pub fn tonemap_reinhard(color: Vec3x4) -> Vec3x4 {
    let denom_rcp = (color + Vec3x4::ONE).recip();
    color * denom_rcp
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
