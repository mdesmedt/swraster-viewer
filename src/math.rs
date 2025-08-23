use glam::{Vec3, Vec4};
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

    pub fn from_vec3(v: Vec3) -> Self {
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
