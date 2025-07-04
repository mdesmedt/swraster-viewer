use glam::Vec4;

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
