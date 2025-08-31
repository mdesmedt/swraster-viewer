use glam::{BVec4, BVec4A, Mat4, UVec4, Vec3A, Vec4};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
pub fn sqrt_vec(v: Vec4) -> Vec4 {
    let x: float32x4_t = v.into();
    let x = unsafe { vsqrtq_f32(x) };
    Vec4::from(x)
}

#[cfg(target_arch = "aarch64")]
pub fn rsqrt_vec(v: Vec4) -> Vec4 {
    let x: float32x4_t = v.into();
    let y0 = unsafe { vrsqrteq_f32(x) };
    let y1 = unsafe { vmulq_f32(y0, vrsqrtsq_f32(vmulq_f32(x, y0), y0)) };
    Vec4::from(y1)
}

#[cfg(target_arch = "x86_64")]
pub fn sqrt_vec(v: Vec4) -> Vec4 {
    let x: __m128 = v.into();
    let x = unsafe { _mm_sqrt_ps(x) };
    Vec4::from(x)
}

#[cfg(target_arch = "x86_64")]
pub fn rsqrt_vec(v: Vec4) -> Vec4 {
    let x: __m128 = v.into();
    let x = unsafe { _mm_rsqrt_ps(x) };
    Vec4::from(x)
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn sqrt_vec(v: Vec4) -> Vec4 {
    v.map(|x| x.sqrt())
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn rsqrt_vec(v: Vec4) -> Vec4 {
    v.map(|x| 1.0 / x.sqrt())
}

#[derive(Clone, Copy)]
pub struct Vec3x4 {
    pub x: Vec4,
    pub y: Vec4,
    pub z: Vec4,
}

impl Vec3x4 {
    pub fn new(x: Vec4, y: Vec4, z: Vec4) -> Self {
        Self { x, y, z }
    }

    pub fn from_f32(x: f32, y: f32, z: f32) -> Self {
        Self::new(Vec4::splat(x), Vec4::splat(y), Vec4::splat(z))
    }

    pub fn from_vec3a(v: Vec3A) -> Self {
        Self::new(Vec4::splat(v.x), Vec4::splat(v.y), Vec4::splat(v.z))
    }

    pub fn from_vec4(x: Vec4) -> Self {
        Self::new(x, x, x)
    }

    pub fn dot(&self, other: Vec3x4) -> Vec4 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn length_squared(&self) -> Vec4 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalize(&self) -> Vec3x4 {
        let one_over_length = rsqrt_vec(self.length_squared());
        Vec3x4::new(
            self.x * one_over_length,
            self.y * one_over_length,
            self.z * one_over_length,
        )
    }

    pub fn cross(&self, other: Vec3x4) -> Vec3x4 {
        Vec3x4::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn reflect(&self, normal: Vec3x4) -> Vec3x4 {
        *self - normal * self.dot(normal) * 2.0
    }

    pub fn clamp(&self, min: Vec4, max: Vec4) -> Vec3x4 {
        Vec3x4::new(
            self.x.clamp(min, max),
            self.y.clamp(min, max),
            self.z.clamp(min, max),
        )
    }

    pub fn transform_direction_transposed(mat_transposed: Mat4, v: Vec3x4) -> Vec3x4 {
        let row0 = mat_transposed.col(0);
        let row1 = mat_transposed.col(1);
        let row2 = mat_transposed.col(2);

        let out_x = v.x * row0.x + v.y * row0.y + v.z * row0.z + row0.w;
        let out_y = v.x * row1.x + v.y * row1.y + v.z * row1.z + row1.w;
        let out_z = v.x * row2.x + v.y * row2.y + v.z * row2.z + row2.w;

        Vec3x4 {
            x: out_x,
            y: out_y,
            z: out_z,
        }
    }

    pub fn select(mask: BVec4A, a: Vec3x4, b: Vec3x4) -> Vec3x4 {
        Vec3x4::new(
            Vec4::select(mask, a.x, b.x),
            Vec4::select(mask, a.y, b.y),
            Vec4::select(mask, a.z, b.z),
        )
    }
}

impl Add for Vec3x4 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl AddAssign for Vec3x4 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl AddAssign<Vec4> for Vec3x4 {
    fn add_assign(&mut self, other: Vec4) {
        self.x += other;
        self.y += other;
        self.z += other;
    }
}

impl AddAssign<f32> for Vec3x4 {
    fn add_assign(&mut self, other: f32) {
        self.x += other;
        self.y += other;
        self.z += other;
    }
}

impl Sub for Vec3x4 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Sub<f32> for Vec3x4 {
    type Output = Self;

    fn sub(self, other: f32) -> Self {
        Self::new(self.x - other, self.y - other, self.z - other)
    }
}

impl SubAssign for Vec3x4 {
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl SubAssign<Vec4> for Vec3x4 {
    fn sub_assign(&mut self, other: Vec4) {
        self.x -= other;
        self.y -= other;
        self.z -= other;
    }
}

impl SubAssign<f32> for Vec3x4 {
    fn sub_assign(&mut self, other: f32) {
        self.x -= other;
        self.y -= other;
        self.z -= other;
    }
}

impl Mul for Vec3x4 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl Mul<f32> for Vec3x4 {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl Mul<Vec4> for Vec3x4 {
    type Output = Self;

    fn mul(self, vec4: Vec4) -> Self {
        Self::new(self.x * vec4, self.y * vec4, self.z * vec4)
    }
}

impl MulAssign for Vec3x4 {
    fn mul_assign(&mut self, other: Self) {
        self.x *= other.x;
        self.y *= other.y;
        self.z *= other.z;
    }
}

impl MulAssign<Vec4> for Vec3x4 {
    fn mul_assign(&mut self, vec4: Vec4) {
        self.x *= vec4;
        self.y *= vec4;
        self.z *= vec4;
    }
}

impl MulAssign<f32> for Vec3x4 {
    fn mul_assign(&mut self, scalar: f32) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
    }
}

// Utility functions
// TODO: Move these to a separate file

pub fn pack_colors(color: Vec3x4) -> UVec4 {
    // Gamma correct the final colors before packing
    // Apply gamma 2.0 instead of 2.2 for perf
    let color_r = sqrt_vec(color.x);
    let color_g = sqrt_vec(color.y);
    let color_b = sqrt_vec(color.z);

    UVec4::new(
        ((color_r.x * 255.0) as u32) << 16
            | ((color_g.x * 255.0) as u32) << 8
            | ((color_b.x * 255.0) as u32),
        ((color_r.y * 255.0) as u32) << 16
            | ((color_g.y * 255.0) as u32) << 8
            | ((color_b.y * 255.0) as u32),
        ((color_r.z * 255.0) as u32) << 16
            | ((color_g.z * 255.0) as u32) << 8
            | ((color_b.z * 255.0) as u32),
        ((color_r.w * 255.0) as u32) << 16
            | ((color_g.w * 255.0) as u32) << 8
            | ((color_b.w * 255.0) as u32),
    )
}

pub fn unpack_colors(color: UVec4) -> Vec3x4 {
    // Extract red (bits 16-23)
    let color_r = Vec4::new(
        ((color.x >> 16) & 0xFF) as f32 / 255.0,
        ((color.y >> 16) & 0xFF) as f32 / 255.0,
        ((color.z >> 16) & 0xFF) as f32 / 255.0,
        ((color.w >> 16) & 0xFF) as f32 / 255.0,
    );

    // Extract green (bits 8-15)
    let color_g = Vec4::new(
        ((color.x >> 8) & 0xFF) as f32 / 255.0,
        ((color.y >> 8) & 0xFF) as f32 / 255.0,
        ((color.z >> 8) & 0xFF) as f32 / 255.0,
        ((color.w >> 8) & 0xFF) as f32 / 255.0,
    );

    // Extract blue (bits 0-7)
    let color_b = Vec4::new(
        (color.x & 0xFF) as f32 / 255.0,
        (color.y & 0xFF) as f32 / 255.0,
        (color.z & 0xFF) as f32 / 255.0,
        (color.w & 0xFF) as f32 / 255.0,
    );

    Vec3x4::new(color_r * color_r, color_g * color_g, color_b * color_b)
}

pub fn bvec4_to_bvec4a(b: BVec4) -> BVec4A {
    let arr: [bool; 4] = b.into();
    BVec4A::from(arr)
}

pub fn bvec4a_to_bvec4(b: BVec4A) -> BVec4 {
    let arr: [bool; 4] = b.into();
    BVec4::from(arr)
}

pub fn interpolate_attribute(
    a: f32,
    b: f32,
    c: f32,
    bary0: Vec4,
    bary1: Vec4,
    bary2: Vec4,
) -> Vec4 {
    a * bary0 + b * bary1 + c * bary2
}

pub fn interpolate_attribute_vec3x4(
    a: Vec3A,
    b: Vec3A,
    c: Vec3A,
    bary0: Vec4,
    bary1: Vec4,
    bary2: Vec4,
) -> Vec3x4 {
    let x = interpolate_attribute(a.x, b.x, c.x, bary0, bary1, bary2);
    let y = interpolate_attribute(a.y, b.y, c.y, bary0, bary1, bary2);
    let z = interpolate_attribute(a.z, b.z, c.z, bary0, bary1, bary2);
    Vec3x4::new(x, y, z)
}

pub fn aces_tonemap(color: Vec3x4) -> Vec3x4 {
    let a = Vec4::splat(2.51);
    let b = Vec4::splat(0.03);
    let c = Vec4::splat(2.43);
    let d = Vec4::splat(0.59);
    let e = Vec4::splat(0.14);

    let map_channel = |v: Vec4| {
        let numerator   = v * (a * v + b);
        let denominator = v * (c * v + d) + e;
        (numerator / denominator).clamp(Vec4::ZERO, Vec4::ONE)
    };

    Vec3x4 {
        x: map_channel(color.x),
        y: map_channel(color.y),
        z: map_channel(color.z),
    }
}
