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
    pub light_dir: Vec3x4,
    pub light_color: Vec3x4,
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
    pub light_dir: Vec3x4,
    pub light_color: Vec3x4,
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
            light_dir: params.light_dir,
            light_color: params.light_color,
            current_color: current_color,
            packet: params.packet,
            material: params.material,
            camera: params.camera,
            scene: params.scene,
        }
    }
}

/// Calculates pseudo-PBR shading and returns a packed color
pub fn pbr_shader<const TRANSLUCENT: bool>(shading_params: PbrShaderParams) -> Vec3x4 {
    let bary1 = shading_params.bary1;
    let bary2 = shading_params.bary2;
    let light_dir = shading_params.light_dir;
    let light_color = shading_params.light_color;
    let packet = shading_params.packet;
    let material = shading_params.material;
    let camera = shading_params.camera;
    let scene = shading_params.scene;

    // Begin attribute interpolation

    let w = 1.0 / packet.one_over_w.interpolate(bary1, bary2);

    let input_normal = packet.normals.interpolate(bary1, bary2).normalize();
    let input_tangent = packet.tangents.interpolate(bary1, bary2).normalize();
    let pos_world = packet.pos_world_over_w.interpolate(bary1, bary2) * w;
    let uv_x = packet.u_over_w.interpolate(bary1, bary2) * w;
    let uv_y = packet.v_over_w.interpolate(bary1, bary2) * w;
    let du_dv = packet.du_dv * w;

    // Apply normal mapping if we have a normal map
    let mut normal_world = input_normal;
    if let Some(normal_map) = &material.normal_texture {
        let mut tangent_space_normal = normal_map.sample4_rgb(uv_x, uv_y, du_dv) * 2.0 - 1.0;
        tangent_space_normal = tangent_space_normal.normalize();
        let binormal = input_normal.cross(input_tangent);
        normal_world = input_tangent * tangent_space_normal.x
            + binormal * tangent_space_normal.y
            + input_normal * tangent_space_normal.z;
    }

    // Compute N.L diffuse lighting
    let n_dot_l = normal_world.dot(light_dir).clamp(Vec4::ZERO, Vec4::ONE);

    // Compute the view direction for each pixel
    let view_dir = Vec3x4::from_vec3a(camera.position) - pos_world;
    let view_normal = view_dir.normalize();
    // Compute half vector
    let half_vector_add = light_dir + view_normal;
    // Normalize half vector
    let half_vector = half_vector_add.normalize();
    // H.N
    let n_dot_h = normal_world.dot(half_vector);
    // Exponentiate
    let n_dot_h_2 = n_dot_h * n_dot_h;
    let n_dot_h_4 = n_dot_h_2 * n_dot_h_2;
    let n_dot_h_8 = n_dot_h_4 * n_dot_h_4;
    let n_dot_h_16 = n_dot_h_8 * n_dot_h_8;
    let n_dot_h_32 = n_dot_h_16 * n_dot_h_16;

    // Get voxel grid lighting if available, for shadows
    let voxel_light_intensity = if let Some(voxel_grid) = &scene.voxel_grid {
        voxel_grid.get_filtered_light_intensity_vec(pos_world)
    } else {
        Vec4::ONE
    };

    let light_intensity = n_dot_l * voxel_light_intensity;

    // Diffuse starts with the base color factor
    let mut diffuse = Vec3x4::from_f32(
        material.base_color_factor.x,
        material.base_color_factor.y,
        material.base_color_factor.z,
    );

    // If we have a base texture, sample it
    if let Some(diffuse_texture) = &material.base_color_texture {
        diffuse *= diffuse_texture.sample4_rgb(uv_x, uv_y, du_dv);
    }

    let mut roughness = Vec4::splat(material.roughness_factor);
    let mut metallic = Vec4::splat(material.metallic_factor);

    // If we have a metallic/roughness texture, sample it
    if let Some(spec_texture) = &material.metallic_roughness_texture {
        let metallic_roughness_mat = spec_texture.sample4_rgb(uv_x, uv_y, du_dv);
        roughness *= metallic_roughness_mat.y;
        metallic *= metallic_roughness_mat.z;
    }

    // Handy inverse roughness and inverse metallic factors
    let shininess = Vec4::ONE - roughness;
    let dielectric = Vec4::ONE - metallic;

    // Compute specular
    let specular_shiny = n_dot_h_32;
    let specular_rough = n_dot_h_2 * Vec4::splat(0.01);
    let n_dot_h_blend = specular_shiny * shininess + specular_rough * roughness;
    let specular = n_dot_h_blend * voxel_light_intensity;

    // Now compute the color, starting with diffuse lighting
    // TODO: This is wrong for metallic
    let mut color = light_color * light_intensity;

    // Add ambient diffuse lighting
    // TODO: Arbitrary ambient. Sample filtered cubemap instead?
    let ambient_top: Vec3x4 = Vec3x4::from_f32(0.1, 0.1, 0.15);
    let ambient_bottom: Vec3x4 = Vec3x4::from_f32(0.1, 0.1, 0.1);
    let ambient_factor_top = normal_world.y.clamp(Vec4::ZERO, Vec4::ONE);
    let ambient_factor_bottom = Vec4::ONE - ambient_factor_top;
    color += ambient_top * ambient_factor_top + ambient_bottom * ambient_factor_bottom;

    // For translucent materials, blend in the current framebuffer color
    if TRANSLUCENT {
        let mut transmission = Vec4::splat(material.transmission);

        // Sample transmission texture if provided
        if let Some(transmission_texture) = &material.transmission_texture {
            let transmission_mat = transmission_texture.sample4_rgb(uv_x, uv_y, du_dv);
            transmission *= transmission_mat.x;
        }

        // Multiply 1-transmission to diffuse lighting
        let inv_transmission = Vec4::ONE - transmission;
        color *= inv_transmission;

        // Add the transmitted color multiplied by the transmission factor
        color += shading_params.current_color * transmission;
    }

    // Multiply by diffuse color
    color *= diffuse;

    // Add dielectric specular
    color += specular * dielectric;

    // Add metallic specular
    color += diffuse * specular * metallic;

    // Sample emissive texture if provided
    if let Some(emissive_texture) = &material.emissive_texture {
        let emissive_mat = emissive_texture.sample4_rgb(uv_x, uv_y, du_dv);
        color += emissive_mat * material.emissive_factor;
    }

    // Sample cubemap (with some totally arbitrary weighting)
    let cube_mip_level = scene.cubemap.mip_level_from_scalar(roughness);
    let reflect = view_normal.reflect(normal_world);
    let cubemap_color = scene
        .cubemap
        .sample_cubemap_rgb(reflect * -1.0, cube_mip_level);
    let shininess_2 = shininess * shininess;
    let shininess_4 = shininess_2 * shininess_2;
    // Add dielectric cubemap contribution
    color += cubemap_color * dielectric * shininess_4 * Vec4::splat(0.2);
    // Add metallic cubemap contribution
    color += cubemap_color * metallic * diffuse * shininess;

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
