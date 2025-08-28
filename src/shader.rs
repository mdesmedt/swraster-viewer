use glam::{BVec4A, UVec4, Vec4};

use crate::{
    math::*,
    rendercamera::RenderCamera,
    renderer::RasterPacket,
    scene::{Material, Scene},
    tilerasterizer::TileRasterizer,
};

pub struct RasterizerShaderParams<'a> {
    pub index_in_tile: usize,
    pub packet_index: usize,
    pub bary0: Vec4,
    pub bary1: Vec4,
    pub bary2: Vec4,
    pub z: Vec4,
    pub w: Vec4,
    pub mask: BVec4A,
    pub depth_from_depth_test: Vec4,
    pub light_dir: Vec3x4,
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
        let bary0 = params.bary0;
        let bary1 = params.bary1;
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
        tile.bary0[params.index_in_tile] =
            Vec4::select(mask, bary0, tile.bary0[params.index_in_tile]);
        tile.bary1[params.index_in_tile] =
            Vec4::select(mask, bary1, tile.bary1[params.index_in_tile]);
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
        tile.color[params.index_in_tile] = UVec4::select(
            bvec4a_to_bvec4(params.mask),
            color,
            tile.color[params.index_in_tile],
        );
    }
}

pub struct PbrShaderParams<'a> {
    pub current_color: UVec4,
    pub bary0: Vec4,
    pub bary1: Vec4,
    pub bary2: Vec4,
    pub mask: BVec4A,
    pub light_dir: Vec3x4,
    pub packet: &'a RasterPacket,
    pub material: &'a Material,
    pub camera: &'a RenderCamera,
    pub scene: &'a Scene,
}

impl<'a> PbrShaderParams<'a> {
    fn from_rasterizer_params(params: &RasterizerShaderParams<'a>, current_color: UVec4) -> Self {
        Self {
            current_color: current_color,
            bary0: params.bary0,
            bary1: params.bary1,
            bary2: params.bary2,
            mask: params.mask,
            light_dir: params.light_dir,
            packet: params.packet,
            material: params.material,
            camera: params.camera,
            scene: params.scene,
        }
    }
}

/// Calculates pseudo-PBR shading and returns a packed color
pub fn pbr_shader<const TRANSLUCENT: bool>(shading_params: PbrShaderParams) -> UVec4 {
    let bary0 = shading_params.bary0;
    let bary1 = shading_params.bary1;
    let bary2 = shading_params.bary2;
    let mask = shading_params.mask;
    let light_dir = shading_params.light_dir;
    let packet = shading_params.packet;
    let material = shading_params.material;
    let camera = shading_params.camera;
    let scene = shading_params.scene;

    // Begin attribute interpolation

    let w = 1.0
        / interpolate_attribute(
            packet.one_over_w[0],
            packet.one_over_w[1],
            packet.one_over_w[2],
            bary0,
            bary1,
            bary2,
        );

    let input_normal = interpolate_attribute_vec3x4(
        packet.normals[0],
        packet.normals[1],
        packet.normals[2],
        bary0,
        bary1,
        bary2,
    )
    .normalize();

    let input_tangent = interpolate_attribute_vec3x4(
        packet.tangents[0],
        packet.tangents[1],
        packet.tangents[2],
        bary0,
        bary1,
        bary2,
    )
    .normalize();

    let pos_world = interpolate_attribute_vec3x4(
        packet.pos_world_over_w[0],
        packet.pos_world_over_w[1],
        packet.pos_world_over_w[2],
        bary0,
        bary1,
        bary2,
    ) * w;

    let uv_x = interpolate_attribute(
        packet.uv_over_w[0].x,
        packet.uv_over_w[1].x,
        packet.uv_over_w[2].x,
        bary0,
        bary1,
        bary2,
    ) * w;
    let uv_y = interpolate_attribute(
        packet.uv_over_w[0].y,
        packet.uv_over_w[1].y,
        packet.uv_over_w[2].y,
        bary0,
        bary1,
        bary2,
    ) * w;

    let du_dv = packet.du_dv * w;

    let mut normal_world = input_normal;

    // Apply normal mapping if we have a normal map
    if let Some(normal_map) = &material.normal_texture {
        let normal_map_mat = normal_map.sample4(uv_x, uv_y, du_dv);
        let mut tangent_space_normal = (Vec3x4::new(
            normal_map_mat.col(0),
            normal_map_mat.col(1),
            normal_map_mat.col(2),
        )) * 2.0
            - 1.0;
        tangent_space_normal = tangent_space_normal.normalize();
        let binormal = input_normal.cross(input_tangent);
        normal_world = input_tangent * tangent_space_normal.x
            + binormal * tangent_space_normal.y
            + input_normal * tangent_space_normal.z;
    }

    // Compute N.L diffuse lighting
    let n_dot_l = normal_world.dot(light_dir);
    let diffuse = n_dot_l.clamp(Vec4::ZERO, Vec4::ONE);

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

    // More ambient light coming from the top, peak intensity of 0.15
    let ambient = (normal_world.y + Vec4::splat(1.5)) * Vec4::splat((0.5 / 1.5) * 0.2);

    // Get voxel grid lighting if available, for shadows
    let voxel_light_intensity = if let Some(voxel_grid) = &scene.voxel_grid {
        voxel_grid.get_filtered_light_intensity_vec(pos_world)
    } else {
        Vec4::ONE
    };

    let light_intensity = diffuse * voxel_light_intensity + ambient;

    // Diffuse starts with the base color factor
    let mut diffuse = Vec3x4::from_f32(
        material.base_color_factor.x,
        material.base_color_factor.y,
        material.base_color_factor.z,
    );

    let mut out_mask = mask;

    // If we have a base texture, sample it
    if let Some(diffuse_texture) = &material.base_color_texture {
        let diffuse_mat = diffuse_texture.sample4(uv_x, uv_y, du_dv);

        // Alpha test
        if material.is_alpha_tested {
            let alpha = diffuse_mat.col(3);
            out_mask &= alpha.cmpge(material.alpha_cutoff_vec);
        }

        // Multiply diffuse color
        // Simple gamma 2.0 instead of 2.2 for perf
        diffuse.x *= diffuse_mat.col(0) * diffuse_mat.col(0);
        diffuse.y *= diffuse_mat.col(1) * diffuse_mat.col(1);
        diffuse.z *= diffuse_mat.col(2) * diffuse_mat.col(2);
    }

    let mut roughness = Vec4::splat(material.roughness_factor);
    let mut metallic = Vec4::splat(material.metallic_factor);

    // If we have a metallic/roughness texture, sample it
    if let Some(spec_texture) = &material.metallic_roughness_texture {
        let metallic_roughness_mat = spec_texture.sample4(uv_x, uv_y, du_dv);
        roughness *= metallic_roughness_mat.col(1);
        metallic *= metallic_roughness_mat.col(2);
    }
    let dielectric = Vec4::ONE - metallic;

    // Compute specular
    let shininess = Vec4::ONE - roughness;
    let specular_shiny = n_dot_h_32;
    let specular_rough = n_dot_h_2 * Vec4::splat(0.01);
    let n_dot_h_blend = specular_shiny * shininess + specular_rough * roughness;
    let specular = n_dot_h_blend * voxel_light_intensity;

    // Now compute the color, starting with irradiance
    // TODO: This is wrong for metallic, but without filtered cubemaps it's tricky to fix
    let mut color = Vec3x4::from_vec4(light_intensity);

    // For translucent materials, blend in the current framebuffer color
    if TRANSLUCENT {
        let mut transmission = Vec4::splat(material.transmission);

        // Sample transmission texture if provided
        if let Some(transmission_texture) = &material.transmission_texture {
            let transmission_mat = transmission_texture.sample4(uv_x, uv_y, du_dv);
            transmission *= transmission_mat.col(0);
        }

        // Multiply 1-transmission to diffuse lighting
        let inv_transmission = Vec4::ONE - transmission;
        color *= inv_transmission;

        // Add the transmitted color multiplied by the transmission factor
        let current_color_packed = shading_params.current_color;
        let current_color = unpack_colors(current_color_packed);
        color += current_color * transmission;
    }

    // Multiply by diffuse color
    color *= diffuse;

    // Add dielectric specular
    color += specular * dielectric;

    // Add metallic specular
    color += diffuse * specular * metallic;

    // Sample cubemap (with some totally arbitrary weighting)
    // TODO: Currently fades out cubemap with roughness because we lack cubemap mipmaps
    let reflect = view_normal.reflect(normal_world);
    let cubemap_mat = scene.cubemap.sample_cubemap(reflect * -1.0);
    let dielectric_strength =
        ((shininess - Vec4::splat(0.6)) * Vec4::splat(0.22)).clamp(Vec4::ZERO, Vec4::ONE);
    let metallic_strength = metallic * shininess;
    let cubemap_color = Vec3x4::new(
        cubemap_mat.col(0) * cubemap_mat.col(0),
        cubemap_mat.col(1) * cubemap_mat.col(1),
        cubemap_mat.col(2) * cubemap_mat.col(2),
    );
    // Add dielectric cubemap contribution
    color += cubemap_color * dielectric_strength;
    // Add metallic cubemap contribution
    color += cubemap_color * diffuse * metallic_strength;

    // Debug: Show world space position
    //color = pos_world;

    // Debug: Show normal
    // color = input_normal;

    // Debug: Show voxel lighting
    // color.x = voxel_light_intensity;
    // color.y = voxel_light_intensity;
    // color.z = voxel_light_intensity;

    // Debug: Show cubemap
    // color.x = cubemap_mat.col(0);
    // color.y = cubemap_mat.col(1);
    // color.z = cubemap_mat.col(2);

    // Clamp final colors before packing
    color = color.clamp(Vec4::ZERO, Vec4::ONE);

    pack_colors(color)
}

fn get_alpha_test_mask(shading_params: &RasterizerShaderParams, w: Vec4) -> BVec4A {
    let bary0 = shading_params.bary0;
    let bary1 = shading_params.bary1;
    let bary2 = shading_params.bary2;
    let packet = shading_params.packet;
    let material = shading_params.material;
    let mask = shading_params.mask;

    let du_dv = packet.du_dv;

    let uv_x = interpolate_attribute(
        packet.uv_over_w[0].x,
        packet.uv_over_w[1].x,
        packet.uv_over_w[2].x,
        bary0,
        bary1,
        bary2,
    ) * w;
    let uv_y = interpolate_attribute(
        packet.uv_over_w[0].y,
        packet.uv_over_w[1].y,
        packet.uv_over_w[2].y,
        bary0,
        bary1,
        bary2,
    ) * w;

    let mut out_mask = mask;
    if let Some(diffuse_texture) = &material.base_color_texture {
        let diffuse_mat = diffuse_texture.sample4(uv_x, uv_y, du_dv);
        let alpha = diffuse_mat.col(3);
        out_mask &= alpha.cmpge(material.alpha_cutoff_vec);
    }

    out_mask
}
