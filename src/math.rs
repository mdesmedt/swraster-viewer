use glam::Vec4;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
pub fn sqrt_vec(v: Vec4) -> Vec4 {
    let l_m128: float32x4_t = v.into();
    let l_m128 = unsafe { vsqrtq_f32(l_m128) };
    Vec4::from(l_m128)
}

#[cfg(target_arch = "aarch64")]
pub fn rsqrt_vec(v: Vec4) -> Vec4 {
    let l_m128: float32x4_t = v.into();
    let l_m128 = unsafe { vrsqrteq_f32(l_m128) };
    Vec4::from(l_m128)
}

#[cfg(target_arch = "x86_64")]
pub fn sqrt_vec(v: Vec4) -> Vec4 {
    let l_m128: __m128 = v.into();
    let l_m128 = unsafe { _mm_sqrt_ps(l_m128) };
    Vec4::from(l_m128)
}


#[cfg(target_arch = "x86_64")]
pub fn rsqrt_vec(v: Vec4) -> Vec4 {
    let l_m128: __m128 = v.into();
    let l_m128 = unsafe { _mm_rsqrt_ps(l_m128) };
    Vec4::from(l_m128)
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn sqrt_vec(v: Vec4) -> Vec4 {
    v.map(|x| x.sqrt())
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn rsqrt_vec(v: Vec4) -> Vec4 {
    v.map(|x| 1.0 / x.sqrt())
}
