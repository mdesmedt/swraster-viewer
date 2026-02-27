use glam::{BVec4A, UVec4, Vec4};

use crate::math::*;
use crate::rendercamera::RenderCamera;
use crate::renderer::RasterPacket;
use crate::scene::{Material, Scene};
use crate::tilerasterizer::TileRasterizer;
use crate::util::*;

pub struct RasterizerShaderParams<'a> {
    pub index_in_tile: usize,
    pub packet_index: usize,
    pub bary1: Vec4,
    pub bary2: Vec4,
    pub z: Vec4,
    pub w: Vec4,
    pub mask: BVec4A,
    pub depth_from_depth_test: Vec4,
    pub packet: &'a RasterPacket,
    pub material: &'a Material,
    pub camera: &'a RenderCamera,
    pub scene: &'a Scene,
}

pub trait RasterizerShader {
    fn shade(&self, params: RasterizerShaderParams, tile: &mut TileRasterizer);
}

// Shader which outputs a vbuffer for opaque fragments
pub struct VBufferOpaqueShader;

impl RasterizerShader for VBufferOpaqueShader {
    fn shade(&self, params: RasterizerShaderParams, tile: &mut TileRasterizer) {
        let mut mask = params.mask;
        let material = params.material;
        let bary1 = params.bary1;
        let bary2 = params.bary2;
        let packet_index_vec = UVec4::splat(params.packet_index as u32);

        if material.is_alpha_tested {
            let alpha_test_mask = get_alpha_test_mask(&params, params.w);
            mask &= alpha_test_mask;
        }
        // Store depth, packet index and barycentrics
        let mask_bvec4 = bvec4a_to_bvec4(mask);
        tile.packet_index[params.index_in_tile] = UVec4::select(
            mask_bvec4,
            packet_index_vec,
            tile.packet_index[params.index_in_tile],
        );
        tile.bary1[params.index_in_tile] =
            Vec4::select(mask, bary1, tile.bary1[params.index_in_tile]);
        tile.bary2[params.index_in_tile] =
            Vec4::select(mask, bary2, tile.bary2[params.index_in_tile]);
        if !material.is_alpha_tested {
            // Write the result from the early Z test
            tile.depth[params.index_in_tile] = params.depth_from_depth_test;
        } else {
            tile.depth[params.index_in_tile] =
                Vec4::select(mask, params.z, tile.depth[params.index_in_tile]);
        }
    }
}

// Shader which executes forward shading for translucent fragments
pub struct TranslucentForwardShader;

impl RasterizerShader for TranslucentForwardShader {
    fn shade(&self, params: RasterizerShaderParams, tile: &mut TileRasterizer) {
        let current_color = tile.color[params.index_in_tile];
        let shading_params = PbrShaderParams::from_rasterizer_params(&params, current_color);
        let color = pbr_shader::<true>(shading_params);
        tile.color[params.index_in_tile] =
            Vec3x4::select(params.mask, color, tile.color[params.index_in_tile]);
    }
}

pub struct PbrShaderParams<'a> {
    pub bary1: Vec4,
    pub bary2: Vec4,
    pub current_color: Vec3x4,
    pub packet: &'a RasterPacket,
    pub material: &'a Material,
    pub camera: &'a RenderCamera,
    pub scene: &'a Scene,
}

impl<'a> PbrShaderParams<'a> {
    fn from_rasterizer_params(params: &RasterizerShaderParams<'a>, current_color: Vec3x4) -> Self {
        Self {
            bary1: params.bary1,
            bary2: params.bary2,
            current_color: current_color,
            packet: params.packet,
            material: params.material,
            camera: params.camera,
            scene: params.scene,
        }
    }
}

fn eval_irradiance_sh(coeffs: &[glam::Vec3A; 4], normal: Vec3x4) -> Vec3x4 {
    let x = normal.x;
    let y = normal.y;
    let z = normal.z;

    Vec3x4::from_vec3a(coeffs[0])
        + Vec3x4::from_vec3a(coeffs[1]) * y
        + Vec3x4::from_vec3a(coeffs[2]) * z
        + Vec3x4::from_vec3a(coeffs[3]) * x
}

pub fn pbr_shader<const TRANSLUCENT: bool>(shading_params: PbrShaderParams) -> Vec3x4 {
    const EPS: Vec4 = Vec4::splat(1e-6);
    const PI: Vec4 = Vec4::splat(std::f32::consts::PI);

    // Fetch parameters from the struct for convenience
    let bary1 = shading_params.bary1;
    let bary2 = shading_params.bary2;
    let packet = shading_params.packet;
    let material = shading_params.material;
    let camera = shading_params.camera;
    let scene = shading_params.scene;

    // Attribute interpolation
    let w = 1.0 / packet.one_over_w.interpolate(bary1, bary2);
    let input_normal = packet.normals.interpolate(bary1, bary2).normalize();
    let input_tangent = packet.tangents.interpolate(bary1, bary2).normalize();
    let tangent_sign = packet.tangent_sign.interpolate(bary1, bary2);
    let pos_world = packet.pos_world_over_w.interpolate(bary1, bary2) * w;
    let uv_x = packet.u_over_w.interpolate(bary1, bary2) * w;
    let uv_y = packet.v_over_w.interpolate(bary1, bary2) * w;
    let du_dv = packet.du_dv * w;

    // Re-orthonormalize TBN and apply normal map when available
    let tangent_world =
        (input_tangent - input_normal * input_normal.dot(input_tangent)).normalize();
    let tangent_handedness = Vec4::select(tangent_sign.cmpge(Vec4::ZERO), Vec4::ONE, Vec4::NEG_ONE);
    let bitangent_world = input_normal.cross(tangent_world) * tangent_handedness;
    let mut normal_world = input_normal.normalize();
    if let Some(normal_map) = &material.normal_texture {
        let tangent_space_normal = normal_map.sample4_rgb(uv_x, uv_y, du_dv) * 2.0 - 1.0;
        normal_world = tangent_world * tangent_space_normal.x
            + bitangent_world * tangent_space_normal.y
            + input_normal * tangent_space_normal.z;
        normal_world = normal_world.normalize();
    }

    // Fetch the light
    let light = &scene.light;
    let light_dir = Vec3x4::new(
        Vec4::splat(light.direction.x),
        Vec4::splat(light.direction.y),
        Vec4::splat(light.direction.z),
    );
    let light_color = Vec3x4::new(
        Vec4::splat(light.color.x),
        Vec4::splat(light.color.y),
        Vec4::splat(light.color.z),
    );

    // Compute the view direction for each pixel
    let view_dir = Vec3x4::from_vec3a(camera.position) - pos_world;
    let view_normal = view_dir.normalize(); // V

    // Parameters for direct BRDF
    let n_dot_l = normal_world.dot(light_dir).max(Vec4::ZERO);
    let half_vector = (light_dir + view_normal).normalize();
    let n_dot_h = normal_world.dot(half_vector).max(Vec4::ZERO);
    let n_dot_v = normal_world.dot(view_normal).max(Vec4::splat(1.0e-4));
    let v_dot_h = view_normal.dot(half_vector).max(Vec4::ZERO);

    // Get voxel grid lighting if available, for shadows
    let voxel_light_intensity = if let Some(voxel_grid) = &scene.voxel_grid {
        voxel_grid.get_filtered_light_intensity_vec(pos_world)
    } else {
        Vec4::ONE
    };

    let mut base_color_diffuse = Vec3x4::from_f32(
        material.base_color_factor.x,
        material.base_color_factor.y,
        material.base_color_factor.z,
    );

    // If we have a base texture, sample it
    if let Some(diffuse_texture) = &material.base_color_texture {
        base_color_diffuse *= srgb_to_linear_fast(diffuse_texture.sample4_rgb(uv_x, uv_y, du_dv));
    }

    let mut roughness = Vec4::splat(material.roughness_factor);
    let mut metallic = Vec4::splat(material.metallic_factor);

    // If we have a metallic/roughness texture, sample it
    if let Some(spec_texture) = &material.metallic_roughness_texture {
        let metallic_roughness_mat = spec_texture.sample4_rgb(uv_x, uv_y, du_dv);
        roughness *= metallic_roughness_mat.y;
        metallic *= metallic_roughness_mat.z;
    }
    roughness = roughness.clamp(Vec4::splat(0.045), Vec4::ONE);
    metallic = metallic.clamp(Vec4::ZERO, Vec4::ONE);

    let mut ao = Vec4::ONE;
    if let Some(occlusion_texture) = &material.occlusion_texture {
        ao = occlusion_texture.sample4_rgb(uv_x, uv_y, du_dv).x;
        ao = Vec4::ONE + (ao - Vec4::ONE) * Vec4::splat(material.occlusion_strength);
    }

    // Compute f0 for dielectrics (0.04) or base color for metals
    let f0 = Vec3x4::lerp(Vec3x4::splat(0.04), base_color_diffuse, metallic);

    // Direct Schlick Fresnel
    let one_minus_vh = Vec4::ONE - v_dot_h;
    let one_minus_vh2 = one_minus_vh * one_minus_vh;
    let one_minus_vh4 = one_minus_vh2 * one_minus_vh2;
    let one_minus_vh5 = one_minus_vh4 * one_minus_vh; // (1 - V·H)^5
    let brdf_f_direct = f0 + (Vec3x4::ONE - f0) * one_minus_vh5;

    // GGX / Smith-Schlick terms
    let alpha = roughness * roughness;
    let alpha_2 = alpha * alpha;
    let ndh_2 = n_dot_h * n_dot_h;
    let denom_d = ndh_2 * (alpha_2 - Vec4::ONE) + Vec4::ONE;
    let brdf_d = alpha_2 / (PI * (denom_d * denom_d) + EPS);

    // Schlick-GGX geometry
    let k = roughness + Vec4::ONE;
    let k = (k * k) * Vec4::splat(0.125); // (roughness+1)^2 / 8
    let gv_denom = n_dot_v * (Vec4::ONE - k) + k;
    let gv = n_dot_v / (gv_denom + EPS);
    let gl_denom = n_dot_l * (Vec4::ONE - k) + k;
    let gl = n_dot_l / (gl_denom + EPS);
    let brdf_g = gv * gl;

    // Direct lighting with explicit energy conservation
    let specular_dg = (brdf_d * brdf_g) / (Vec4::splat(4.0) * n_dot_l * n_dot_v + EPS);
    let k_s_direct = brdf_f_direct;
    let k_d_direct = (Vec3x4::ONE - k_s_direct) * (Vec4::ONE - metallic);
    let lambert = base_color_diffuse * Vec4::splat(1.0 / std::f32::consts::PI);
    let color_direct_diffuse = light_color * k_d_direct * lambert * n_dot_l * voxel_light_intensity;
    let color_direct_specular =
        light_color * brdf_f_direct * specular_dg * n_dot_l * voxel_light_intensity;

    // Indirect diffuse/specular IBL
    let one_minus_nv = Vec4::ONE - n_dot_v;
    let one_minus_nv2 = one_minus_nv * one_minus_nv;
    let one_minus_nv4 = one_minus_nv2 * one_minus_nv2;
    let one_minus_nv5 = one_minus_nv4 * one_minus_nv;

    let one_minus_roughness = Vec4::ONE - roughness;
    let f_ibl_90 = f0.max(one_minus_roughness);
    let k_s_ibl = f0 + (f_ibl_90 - f0) * one_minus_nv5;
    let k_d_ibl = (Vec3x4::ONE - k_s_ibl) * (Vec4::ONE - metallic);

    let irradiance = eval_irradiance_sh(&scene.irradiance_sh, normal_world);
    let irradiance = irradiance.max(Vec4::ZERO);
    let mut color_indirect_diffuse =
        irradiance * base_color_diffuse * k_d_ibl * Vec4::splat(1.0 / std::f32::consts::PI);

    // HACK: Darken ambient diffuse with shadow factor
    color_indirect_diffuse *= voxel_light_intensity * Vec4::splat(0.75) + Vec4::splat(0.25);

    let reflect_dir = (view_normal * -1.0).reflect(normal_world);
    let spec_mip = roughness * scene.cubemap_specular.texture.max_mip_level_vec;
    let prefiltered_env = scene
        .cubemap_specular
        .sample_cubemap_trilinear_rgb(reflect_dir, spec_mip);
    let brdf_lut = scene.brdf_lut.sample_bilinear_rgb(
        n_dot_v.clamp(Vec4::ZERO, Vec4::ONE),
        roughness.clamp(Vec4::ZERO, Vec4::ONE),
        0,
        UVec4::ZERO,
    );
    let mut brdf_spec_factor = f0 * brdf_lut.x;
    brdf_spec_factor += brdf_lut.y;
    let mut color_indirect_specular = prefiltered_env * brdf_spec_factor;

    // AO only affects indirect lighting
    color_indirect_diffuse *= ao;
    let ao_spec = Vec4::ONE + (ao - Vec4::ONE) * Vec4::splat(0.5);
    color_indirect_specular *= ao_spec;

    // Assemble lit color
    let mut color = if TRANSLUCENT {
        let mut transmission = Vec4::splat(material.transmission);
        if let Some(transmission_texture) = &material.transmission_texture {
            let transmission_mat = transmission_texture.sample4_rgb(uv_x, uv_y, du_dv);
            transmission *= transmission_mat.x;
        }
        transmission = transmission.clamp(Vec4::ZERO, Vec4::ONE);

        let inv_transmission = Vec4::ONE - transmission;
        (shading_params.current_color * base_color_diffuse * transmission)
            + ((color_direct_diffuse + color_indirect_diffuse) * inv_transmission)
            + color_direct_specular
            + color_indirect_specular
    } else {
        color_direct_diffuse
            + color_direct_specular
            + color_indirect_diffuse
            + color_indirect_specular
    };

    // Sample and add emissive texture if provided
    if let Some(emissive_texture) = &material.emissive_texture {
        let emissive_mat = srgb_to_linear_fast(emissive_texture.sample4_rgb(uv_x, uv_y, du_dv));
        color += emissive_mat * material.emissive_factor;
    }

    // Debug: Show world space position
    //color = pos_world;

    // Debug: Show normal
    // color = input_normal;

    // Debug: Show voxel lighting
    //color = Vec3x4::from_vec4(voxel_light_intensity);

    // Debug: Show cubemap
    //color = cubemap_color;

    color
}

fn get_alpha_test_mask(shading_params: &RasterizerShaderParams, w: Vec4) -> BVec4A {
    let bary1 = shading_params.bary1;
    let bary2 = shading_params.bary2;
    let packet = shading_params.packet;
    let material = shading_params.material;
    let mask = shading_params.mask;

    let du_dv = packet.du_dv;

    let u_vec = packet.u_over_w.interpolate(bary1, bary2) * w;
    let v_vec = packet.v_over_w.interpolate(bary1, bary2) * w;

    let mut out_mask = mask;
    if let Some(diffuse_texture) = &material.base_color_texture {
        let alpha = diffuse_texture.sample4_alpha(u_vec, v_vec, du_dv);
        out_mask &= alpha.cmpge(material.alpha_cutoff_vec);
    }
    out_mask
}
