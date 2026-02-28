use crate::raytracer::RayTracer;
use crate::scene::Scene;
use crate::util::srgb_to_linear_scalar;
use crate::voxelgrid::VoxelGrid;
use glam::{UVec4, Vec3A, Vec4};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const SH4_COEFF_COUNT: usize = 4;
const SH4_FLOATS_PER_VOXEL: usize = SH4_COEFF_COUNT * 4;
const RAY_EPSILON: f32 = 1.0e-4;
const GI_CACHE_MAGIC: [u8; 4] = *b"VGI0";
const GI_CACHE_VERSION: u32 = 5;
const GI_CACHE_HEADER_BYTES: usize = 4 + 4 + 4 + 4 + 4;

const GI_PRIMARY_SAMPLES_PER_VOXEL: u32 = 64;
const GI_BOUNCE_ALBEDO_SCALE: f32 = 1.0;
const GI_ACTIVE_DILATION_RADIUS: usize = 2;
const GI_MIN_BRIGHTNESS: f32 = 0.005;

pub fn try_load_gi_cache(path: &Path, voxel_grid: &mut VoxelGrid) -> io::Result<bool> {
    if !path.exists() {
        return Ok(false);
    }

    let expected_size = expected_gi_bytes(voxel_grid);
    let bytes = fs::read(path)?;
    if bytes.len() != expected_size {
        return Ok(false);
    }

    if bytes[0..4] != GI_CACHE_MAGIC {
        return Ok(false);
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != GI_CACHE_VERSION {
        return Ok(false);
    }
    let file_w = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let file_h = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let file_d = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    let (w, h, d) = voxel_grid.dimensions();
    if (file_w, file_h, file_d) != (w, h, d) {
        return Ok(false);
    }

    let mut coeffs = vec![[Vec4::ZERO; SH4_COEFF_COUNT]; voxel_grid.voxel_count()];
    let mut byte_index = GI_CACHE_HEADER_BYTES;
    for voxel_coeffs in &mut coeffs {
        for coeff in voxel_coeffs.iter_mut().take(SH4_COEFF_COUNT) {
            let r = f32::from_le_bytes(bytes[byte_index..byte_index + 4].try_into().unwrap());
            byte_index += 4;
            let g = f32::from_le_bytes(bytes[byte_index..byte_index + 4].try_into().unwrap());
            byte_index += 4;
            let b = f32::from_le_bytes(bytes[byte_index..byte_index + 4].try_into().unwrap());
            byte_index += 4;
            let w = f32::from_le_bytes(bytes[byte_index..byte_index + 4].try_into().unwrap());
            byte_index += 4;
            *coeff = Vec4::new(r, g, b, w);
        }
    }

    voxel_grid.set_all_gi_sh4(coeffs);
    Ok(true)
}

pub fn save_gi_cache(path: &Path, voxel_grid: &VoxelGrid) -> io::Result<()> {
    let gi_data = voxel_grid.gi_sh4();
    let mut bytes = Vec::with_capacity(expected_gi_bytes(voxel_grid));
    bytes.extend_from_slice(&GI_CACHE_MAGIC);
    bytes.extend_from_slice(&GI_CACHE_VERSION.to_le_bytes());
    let (w, h, d) = voxel_grid.dimensions();
    bytes.extend_from_slice(&(w as u32).to_le_bytes());
    bytes.extend_from_slice(&(h as u32).to_le_bytes());
    bytes.extend_from_slice(&(d as u32).to_le_bytes());
    for voxel_coeffs in gi_data {
        for coeff in voxel_coeffs {
            bytes.extend_from_slice(&coeff.x.to_le_bytes());
            bytes.extend_from_slice(&coeff.y.to_le_bytes());
            bytes.extend_from_slice(&coeff.z.to_le_bytes());
            bytes.extend_from_slice(&coeff.w.to_le_bytes());
        }
    }
    fs::write(path, bytes)
}

pub fn expected_gi_bytes(voxel_grid: &VoxelGrid) -> usize {
    GI_CACHE_HEADER_BYTES + expected_gi_payload_bytes(voxel_grid)
}

fn expected_gi_payload_bytes(voxel_grid: &VoxelGrid) -> usize {
    voxel_grid.voxel_count() * SH4_FLOATS_PER_VOXEL * std::mem::size_of::<f32>()
}

pub fn initialize_voxel_gi_from_scene(
    scene: &Scene,
    voxel_grid: &mut VoxelGrid,
    irradiance_scale: f32,
    sky_visibility: f32,
) {
    let total = voxel_grid.voxel_count();
    let (w, h, _) = voxel_grid.dimensions();
    let mut coeffs = vec![[Vec4::ZERO; SH4_COEFF_COUNT]; total];
    for (index, out) in coeffs.iter_mut().enumerate() {
        let z = index / (w * h);
        let rem = index % (w * h);
        let y = rem / w;
        let x = rem % w;
        for (i, sh) in scene.irradiance_sh.iter().enumerate() {
            out[i] = Vec4::new(
                sh.x * irradiance_scale,
                sh.y * irradiance_scale,
                sh.z * irradiance_scale,
                0.0,
            );
        }
        out[0].w = voxel_grid.get_light_intensity(x, y, z);
        out[1].w = sky_visibility;
    }
    voxel_grid.set_all_gi_sh4(coeffs);
}

pub fn build_active_voxel_mask(raytracer: &RayTracer, scene: &Scene, voxel_grid: &mut VoxelGrid) {
    let (width, height, depth) = voxel_grid.dimensions();
    let voxel_size = voxel_grid.voxel_size();
    let world_min = voxel_grid.world_min();
    let voxel_count = voxel_grid.voxel_count();
    let mut occupied = vec![false; voxel_count];
    let mut normal_acc = vec![Vec3A::ZERO; voxel_count];
    let mut albedo_acc = vec![Vec3A::ZERO; voxel_count];
    let mut sample_count = vec![0u32; voxel_count];

    let min_voxel_edge = voxel_size.x.min(voxel_size.y.min(voxel_size.z)).max(1.0e-6);
    let voxel_area_ref = min_voxel_edge * min_voxel_edge;

    for tri in raytracer.triangles() {
        let material = &scene.materials[tri.material_index];
        let albedo = Vec3A::new(
            material.base_color_factor.x,
            material.base_color_factor.y,
            material.base_color_factor.z,
        );

        let e1 = tri.p1 - tri.p0;
        let e2 = tri.p2 - tri.p0;
        let tri_area = 0.5 * e1.cross(e2).length();
        let target_samples = ((tri_area / voxel_area_ref) * 2.0).ceil() as usize;
        let target_samples = target_samples.clamp(1, 4096);
        let n = (target_samples as f32).sqrt().ceil() as usize;

        for iu in 0..n {
            for iv in 0..(n - iu) {
                let u = (iu as f32 + 0.5) / n as f32;
                let v = (iv as f32 + 0.5) / n as f32;
                let w = 1.0 - u - v;
                if w < 0.0 {
                    continue;
                }
                let p = tri.p0 * w + tri.p1 * u + tri.p2 * v;
                let fx = (p.x - world_min.x) / voxel_size.x;
                let fy = (p.y - world_min.y) / voxel_size.y;
                let fz = (p.z - world_min.z) / voxel_size.z;
                if fx < 0.0 || fy < 0.0 || fz < 0.0 {
                    continue;
                }
                let x = fx.floor() as usize;
                let y = fy.floor() as usize;
                let z = fz.floor() as usize;
                if x >= width || y >= height || z >= depth {
                    continue;
                }

                let index = voxel_grid.voxel_index(x, y, z);
                occupied[index] = true;
                normal_acc[index] += tri.n;
                albedo_acc[index] += albedo;
                sample_count[index] += 1;
            }
        }
    }

    let mut surface_normals = vec![Vec3A::ZERO; voxel_count];
    let mut surface_albedo = vec![Vec3A::ZERO; voxel_count];
    for i in 0..voxel_count {
        let count = sample_count[i];
        if count == 0 {
            continue;
        }
        let inv = 1.0 / count as f32;
        let n = normal_acc[i] * inv;
        surface_normals[i] = if n.length_squared() > 1.0e-12 {
            n.normalize()
        } else {
            Vec3A::Y
        };
        surface_albedo[i] = (albedo_acc[i] * inv).clamp(Vec3A::ZERO, Vec3A::ONE);
    }

    // Dilate occupied mask to keep trilinear GI sampling away from zero-valued fringes.
    let mut dilated = occupied.clone();
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let index = voxel_grid.voxel_index(x, y, z);
                if !occupied[index] {
                    continue;
                }
                let z_start = z.saturating_sub(GI_ACTIVE_DILATION_RADIUS);
                let y_start = y.saturating_sub(GI_ACTIVE_DILATION_RADIUS);
                let x_start = x.saturating_sub(GI_ACTIVE_DILATION_RADIUS);
                let z_end = (z + GI_ACTIVE_DILATION_RADIUS).min(depth - 1);
                let y_end = (y + GI_ACTIVE_DILATION_RADIUS).min(height - 1);
                let x_end = (x + GI_ACTIVE_DILATION_RADIUS).min(width - 1);
                for nz in z_start..=z_end {
                    for ny in y_start..=y_end {
                        for nx in x_start..=x_end {
                            let nindex = voxel_grid.voxel_index(nx, ny, nz);
                            dilated[nindex] = true;
                        }
                    }
                }
            }
        }
    }

    let active = dilated.iter().filter(|&&v| v).count();
    let pct = (active as f32 / dilated.len() as f32) * 100.0;
    println!(
        "Active voxel mask: {}/{} ({:.1}%) after dilation",
        active,
        dilated.len(),
        pct
    );
    voxel_grid.set_surface_properties(surface_normals, surface_albedo);
    voxel_grid.set_occupied_mask(occupied);
    voxel_grid.set_active_mask(dilated);
}

pub fn compute_sun_visibility(raytracer: &RayTracer, scene: &Scene, voxel_grid: &mut VoxelGrid) {
    let total = voxel_grid.voxel_count();

    let (width, height, _) = voxel_grid.dimensions();
    let light_direction = scene.light.direction.normalize();
    let voxel_size = voxel_grid.voxel_size();
    let center_min = voxel_grid.world_min() + voxel_size * 0.5;
    let bias = voxel_size.length() * 3.0;
    let active_mask = voxel_grid.active_mask().map(|m| m.to_vec());
    let active_total = active_mask
        .as_ref()
        .map(|mask| mask.iter().filter(|&&v| v).count())
        .unwrap_or(total);
    let mut out = vec![1.0; total];

    if active_total == 0 {
        voxel_grid.set_light_intensity_data(out);
        return;
    }

    out.par_iter_mut()
        .enumerate()
        .for_each(|(index, out_intensity)| {
            if let Some(mask) = &active_mask {
                if !mask[index] {
                    return;
                }
            }

            let z = index / (width * height);
            let rem = index % (width * height);
            let y = rem / width;
            let x = rem % width;
            let voxel_center = center_min + Vec3A::new(x as f32, y as f32, z as f32) * voxel_size;
            let ray_origin = voxel_center + light_direction * bias;
            let transmittance = raytracer.trace_transmittance(
                ray_origin,
                light_direction,
                scene,
                RAY_EPSILON,
                f32::INFINITY,
            );
            *out_intensity = transmittance;
        });

    voxel_grid.set_light_intensity_data(out);
    voxel_grid.blur_grid();
}

pub fn bake_voxel_gi(raytracer: &RayTracer, scene: &Scene, voxel_grid: &mut VoxelGrid) {
    let total = voxel_grid.voxel_count();
    let start = Instant::now();
    let (width, height, _) = voxel_grid.dimensions();
    let active_mask = voxel_grid.active_mask().map(|m| m.to_vec());
    let voxel_size = voxel_grid.voxel_size();
    let min_voxel_edge = voxel_size.x.min(voxel_size.y).min(voxel_size.z).max(1.0e-6);
    let center_min = voxel_grid.world_min() + voxel_size * 0.5;
    let light_dir = scene.light.direction.normalize();
    let active_total = active_mask
        .as_ref()
        .map(|mask| mask.iter().filter(|&&v| v).count())
        .unwrap_or(total);
    let primary_dirs = build_uniform_sphere_samples(GI_PRIMARY_SAMPLES_PER_VOXEL);

    let primary_weight = (4.0 * PI) / (primary_dirs.len() as f32);
    let primary_bias = (voxel_size.length() * 0.35).max(RAY_EPSILON * 4.0);
    let hit_bias = (voxel_size.length() * 0.05).max(RAY_EPSILON * 8.0);
    let gi_tmin = (min_voxel_edge * 0.15).max(RAY_EPSILON);
    let done = AtomicUsize::new(0);
    let print_step = (active_total / 100).max(1);
    initialize_voxel_gi_from_scene(scene, voxel_grid, 1.0, 1.0);
    let mut out_coeffs = voxel_grid.gi_sh4().clone();

    if active_total == 0 {
        voxel_grid.set_all_gi_sh4(out_coeffs);
        return;
    }

    out_coeffs
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, coeffs_out)| {
            if let Some(mask) = &active_mask {
                if !mask[index] {
                    return;
                }
            }

            let z = index / (width * height);
            let rem = index % (width * height);
            let y = rem / width;
            let x = rem % width;
            let voxel_center = center_min + Vec3A::new(x as f32, y as f32, z as f32) * voxel_size;
            let sun_visibility = voxel_grid.get_light_intensity(x, y, z);

            let mut radiance_sh = [Vec3A::ZERO; SH4_COEFF_COUNT];
            let mut sky_hit_count = 0u32;
            for dir in &primary_dirs {
                let ray_origin = voxel_center + *dir * primary_bias;
                let incident_radiance = if let Some(hit) =
                    raytracer.trace_nearest(ray_origin, *dir, gi_tmin, f32::INFINITY)
                {
                    estimate_surface_bounce_radiance(
                        raytracer, scene, &hit, *dir, light_dir, hit_bias, gi_tmin,
                    )
                } else {
                    sky_hit_count += 1;
                    sample_sky_radiance(scene, *dir)
                };

                if incident_radiance.length_squared() <= 0.0 {
                    continue;
                }
                project_radiance_sh4(*dir, incident_radiance, primary_weight, &mut radiance_sh);
            }

            let irradiance = radiance_to_irradiance_sh4(radiance_sh);
            let sky_visibility = sky_hit_count as f32 / primary_dirs.len() as f32;
            for c in 0..SH4_COEFF_COUNT {
                let rgb = irradiance[c].max(Vec3A::ZERO);
                coeffs_out[c] = Vec4::new(rgb.x, rgb.y, rgb.z, 0.0);
            }
            apply_min_brightness_floor(coeffs_out, &scene.irradiance_sh);
            coeffs_out[0].w = sun_visibility;
            coeffs_out[1].w = sky_visibility;

            let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
            if completed % print_step == 0 || completed == active_total {
                print_progress(
                    "Computing GI",
                    completed,
                    active_total,
                    start.elapsed().as_secs_f32(),
                );
            }
        });

    voxel_grid.set_all_gi_sh4(out_coeffs);
    voxel_grid.blur_gi_sh4();
}

fn build_uniform_sphere_samples(sample_count: u32) -> Vec<Vec3A> {
    let mut dirs = Vec::with_capacity(sample_count as usize);
    for i in 0..sample_count {
        let xi = hammersley(i, sample_count);
        let z = 1.0 - 2.0 * xi.x;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = 2.0 * PI * xi.y;
        dirs.push(Vec3A::new(r * phi.cos(), r * phi.sin(), z).normalize());
    }
    dirs
}

fn project_radiance_sh4(
    dir: Vec3A,
    radiance: Vec3A,
    weight: f32,
    out: &mut [Vec3A; SH4_COEFF_COUNT],
) {
    let basis = [
        0.282095,
        0.488603 * dir.y,
        0.488603 * dir.z,
        0.488603 * dir.x,
    ];
    for i in 0..SH4_COEFF_COUNT {
        out[i] += radiance * (basis[i] * weight);
    }
}

fn radiance_to_irradiance_sh4(radiance_sh: [Vec3A; SH4_COEFF_COUNT]) -> [Vec3A; SH4_COEFF_COUNT] {
    let mut out = radiance_sh;
    out[0] *= PI * 0.282095;
    out[1] *= (2.0 * PI / 3.0) * 0.488603;
    out[2] *= (2.0 * PI / 3.0) * 0.488603;
    out[3] *= (2.0 * PI / 3.0) * 0.488603;
    out
}

fn eval_sh4_irradiance(coeffs: &[Vec3A; SH4_COEFF_COUNT], normal: Vec3A) -> Vec3A {
    coeffs[0] + coeffs[1] * normal.y + coeffs[2] * normal.z + coeffs[3] * normal.x
}

fn estimate_surface_bounce_radiance(
    raytracer: &RayTracer,
    scene: &Scene,
    hit: &crate::raytracer::Hit,
    incoming_dir: Vec3A,
    light_dir: Vec3A,
    hit_bias: f32,
    t_min: f32,
) -> Vec3A {
    let tri = raytracer.triangle(hit.triangle_index);
    let material = &scene.materials[hit.material_index];

    let mut normal = hit.normal;
    if normal.dot(-incoming_dir) < 0.0 {
        normal = -normal;
    }

    let mut albedo = Vec3A::new(
        material.base_color_factor.x,
        material.base_color_factor.y,
        material.base_color_factor.z,
    );
    if let Some(base_color_texture) = &material.base_color_texture {
        let uv = hit.uv(tri);
        let tex = base_color_texture
            .sample_bilinear_rgb(Vec4::splat(uv.x), Vec4::splat(uv.y), 0, UVec4::ZERO)
            .extract_lane(0);
        let tex_linear = Vec3A::new(
            srgb_to_linear_scalar(tex.x),
            srgb_to_linear_scalar(tex.y),
            srgb_to_linear_scalar(tex.z),
        );
        albedo *= tex_linear;
    }
    albedo = albedo.clamp(Vec3A::ZERO, Vec3A::ONE) * GI_BOUNCE_ALBEDO_SCALE;
    if albedo.length_squared() <= 0.0 {
        return Vec3A::ZERO;
    }

    let dir_bias = hit_bias * 0.5;
    let shading_origin = hit.position + normal * hit_bias;

    let n_dot_l = normal.dot(light_dir).max(0.0);
    let e_sun = if n_dot_l > 0.0 {
        let sun_origin = shading_origin + light_dir * dir_bias;
        let vis = raytracer.trace_transmittance(sun_origin, light_dir, scene, t_min, f32::INFINITY);
        scene.light.color * (n_dot_l * vis)
    } else {
        Vec3A::ZERO
    };

    // Approximate local skylight using a few deterministic hemisphere visibility rays.
    let sky_vis =
        sky_visibility_hemisphere4(raytracer, scene, shading_origin, normal, dir_bias, t_min);
    let e_sky = eval_sh4_irradiance(&scene.irradiance_sh, normal).max(Vec3A::ZERO) * sky_vis;
    let e_total = (e_sun + e_sky).max(Vec3A::ZERO);
    if e_total.length_squared() <= 0.0 {
        return Vec3A::ZERO;
    }

    albedo * (e_total / PI)
}

fn sky_visibility_hemisphere4(
    raytracer: &RayTracer,
    scene: &Scene,
    origin: Vec3A,
    normal: Vec3A,
    dir_bias: f32,
    t_min: f32,
) -> f32 {
    let (tangent, bitangent) = tangent_basis(normal);
    let dirs_local = [
        Vec3A::new(0.0, 0.0, 1.0),
        Vec3A::new(0.866_025_4, 0.0, 0.5),
        Vec3A::new(-0.433_012_7, 0.75, 0.5),
        Vec3A::new(-0.433_012_7, -0.75, 0.5),
    ];

    let mut vis_sum = 0.0f32;
    for d in dirs_local {
        let dir = (tangent * d.x + bitangent * d.y + normal * d.z).normalize();
        let ray_origin = origin + dir * dir_bias;
        vis_sum += raytracer.trace_transmittance(ray_origin, dir, scene, t_min, f32::INFINITY);
    }
    vis_sum * 0.25
}

fn tangent_basis(normal: Vec3A) -> (Vec3A, Vec3A) {
    let up = if normal.z.abs() < 0.999 {
        Vec3A::Z
    } else {
        Vec3A::X
    };
    let tangent = normal.cross(up).normalize();
    let bitangent = normal.cross(tangent);
    (tangent, bitangent)
}

fn sample_sky_radiance(scene: &Scene, dir: Vec3A) -> Vec3A {
    let sky = scene
        .cubemap
        .sample_cubemap_rgb(
            crate::math::Vec3x4::new(Vec4::splat(dir.x), Vec4::splat(dir.y), Vec4::splat(dir.z)),
            UVec4::ZERO,
        )
        .extract_lane(0);
    Vec3A::new(
        srgb_to_linear_scalar(sky.x),
        srgb_to_linear_scalar(sky.y),
        srgb_to_linear_scalar(sky.z),
    )
}

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    (bits as f32) * 2.328_306_4e-10
}

fn hammersley(i: u32, n: u32) -> glam::Vec2 {
    glam::Vec2::new(i as f32 / n as f32, radical_inverse_vdc(i))
}

fn print_progress(label: &str, done: usize, total: usize, elapsed_sec: f32) {
    let pct = if total > 0 {
        done as f32 / total as f32
    } else {
        1.0
    };
    let eta_sec = if done > 0 {
        elapsed_sec * ((total - done) as f32 / done as f32)
    } else {
        0.0
    };
    println!(
        "{}: {}/{} ({:.1}%) elapsed {:.1}s eta {:.1}s",
        label,
        done,
        total,
        pct * 100.0,
        elapsed_sec,
        eta_sec
    );
}

fn apply_min_brightness_floor(
    coeffs: &mut [Vec4; SH4_COEFF_COUNT],
    scene_irradiance_sh: &[Vec3A; 4],
) {
    let scene_l0 = scene_irradiance_sh[0] * GI_MIN_BRIGHTNESS;
    let floor_lum = luminance(scene_l0);
    if floor_lum <= 0.0 {
        return;
    }

    let current_l0 = Vec3A::new(coeffs[0].x, coeffs[0].y, coeffs[0].z);
    let current_lum = luminance(current_l0);
    if current_lum >= floor_lum {
        return;
    }

    let t = ((floor_lum - current_lum) / floor_lum).clamp(0.0, 1.0);
    for i in 0..SH4_COEFF_COUNT {
        let current = Vec3A::new(coeffs[i].x, coeffs[i].y, coeffs[i].z);
        let target = scene_irradiance_sh[i] * GI_MIN_BRIGHTNESS;
        let blended = current.lerp(target, t);
        coeffs[i].x = blended.x;
        coeffs[i].y = blended.y;
        coeffs[i].z = blended.z;
    }
}

fn luminance(c: Vec3A) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}
