use crate::math::*;
use glam::{USizeVec4, UVec3, Vec3A, Vec4};
use rayon::prelude::*;

/// A 3D voxel grid that stores light intensity values for each voxel
pub struct VoxelGrid {
    width: usize,
    height: usize,
    depth: usize,
    world_min: Vec3A,
    world_max: Vec3A,
    light_intensity: Vec<f32>,
    gi_sh4: Option<Vec<[Vec3A; 4]>>,
    surface_normals: Option<Vec<Vec3A>>,
    surface_albedo: Option<Vec<Vec3A>>,
    occupied_mask: Option<Vec<bool>>,
    active_mask: Option<Vec<bool>>,
}

impl VoxelGrid {
    /// Create a new voxel grid with the specified dimensions and world bounds
    pub fn new(
        width: usize,
        height: usize,
        depth: usize,
        world_min: Vec3A,
        world_max: Vec3A,
    ) -> Self {
        let light_intensity = vec![0.0; width * height * depth];
        Self {
            width,
            height,
            depth,
            world_min,
            world_max,
            light_intensity,
            gi_sh4: None,
            surface_normals: None,
            surface_albedo: None,
            occupied_mask: None,
            active_mask: None,
        }
    }

    /// Returns a parallel iterator over mutable light intensity values with their coordinates
    #[allow(dead_code)]
    pub fn par_iter_mut(&mut self) -> impl ParallelIterator<Item = (UVec3, &mut f32)> {
        let width = self.width;
        let height = self.height;

        self.light_intensity
            .par_iter_mut()
            .enumerate()
            .map(move |(index, intensity)| {
                let z = index / (width * height);
                let remainder = index % (width * height);
                let y = remainder / width;
                let x = remainder % width;

                (UVec3::new(x as u32, y as u32, z as u32), intensity)
            })
    }

    /// Get the world minimum bounds
    pub fn world_min(&self) -> Vec3A {
        self.world_min
    }

    #[allow(dead_code)]
    pub fn world_max(&self) -> Vec3A {
        self.world_max
    }

    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.width, self.height, self.depth)
    }

    pub fn voxel_count(&self) -> usize {
        self.width * self.height * self.depth
    }

    pub fn voxel_index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.width * self.height + y * self.width + x
    }

    #[allow(dead_code)]
    pub fn allocate_gi_sh4(&mut self) {
        self.gi_sh4 = Some(vec![[Vec3A::ZERO; 4]; self.voxel_count()]);
    }

    #[allow(dead_code)]
    pub fn set_gi_sh4(&mut self, x: usize, y: usize, z: usize, coeffs: [Vec3A; 4]) {
        let index = self.voxel_index(x, y, z);
        if let Some(gi_data) = &mut self.gi_sh4 {
            if let Some(slot) = gi_data.get_mut(index) {
                *slot = coeffs;
            }
        }
    }

    pub fn set_all_gi_sh4(&mut self, coeffs: Vec<[Vec3A; 4]>) {
        assert!(coeffs.len() == self.voxel_count());
        self.gi_sh4 = Some(coeffs);
    }

    pub fn has_gi_sh4(&self) -> bool {
        self.gi_sh4.is_some()
    }

    pub fn gi_sh4(&self) -> Option<&Vec<[Vec3A; 4]>> {
        self.gi_sh4.as_ref()
    }

    pub fn set_active_mask(&mut self, mask: Vec<bool>) {
        assert!(mask.len() == self.voxel_count());
        self.active_mask = Some(mask);
    }

    pub fn set_surface_properties(&mut self, normals: Vec<Vec3A>, albedo: Vec<Vec3A>) {
        assert!(normals.len() == self.voxel_count());
        assert!(albedo.len() == self.voxel_count());
        self.surface_normals = Some(normals);
        self.surface_albedo = Some(albedo);
    }

    #[allow(dead_code)]
    pub fn surface_normals(&self) -> Option<&[Vec3A]> {
        self.surface_normals.as_deref()
    }

    #[allow(dead_code)]
    pub fn surface_albedo(&self) -> Option<&[Vec3A]> {
        self.surface_albedo.as_deref()
    }

    pub fn set_occupied_mask(&mut self, mask: Vec<bool>) {
        assert!(mask.len() == self.voxel_count());
        self.occupied_mask = Some(mask);
    }

    #[allow(dead_code)]
    pub fn occupied_mask(&self) -> Option<&[bool]> {
        self.occupied_mask.as_deref()
    }

    pub fn active_mask(&self) -> Option<&[bool]> {
        self.active_mask.as_deref()
    }

    #[allow(dead_code)]
    pub fn is_active_index(&self, index: usize) -> bool {
        self.active_mask
            .as_ref()
            .and_then(|mask| mask.get(index).copied())
            .unwrap_or(true)
    }

    pub fn set_light_intensity_data(&mut self, data: Vec<f32>) {
        assert!(data.len() == self.light_intensity.len());
        self.light_intensity = data;
    }

    /// Get the size of a voxel in world space
    pub fn voxel_size(&self) -> Vec3A {
        let world_size = self.world_max - self.world_min;
        Vec3A::new(
            world_size.x / self.width as f32,
            world_size.y / self.height as f32,
            world_size.z / self.depth as f32,
        )
    }

    /// Get the light intensity at the specified voxel coordinates
    pub fn get_light_intensity(&self, x: usize, y: usize, z: usize) -> f32 {
        let index = self.voxel_index(x, y, z);
        self.light_intensity.get(index).copied().unwrap_or(0.0)
    }

    fn get_gi_coeff(
        gi_data: &[[Vec3A; 4]],
        coeff_index: usize,
        x: usize,
        y: usize,
        z: usize,
        width: usize,
        height: usize,
    ) -> Vec3A {
        let index = z * width * height + y * width + x;
        gi_data
            .get(index)
            .map(|coeffs| coeffs[coeff_index])
            .unwrap_or(Vec3A::ZERO)
    }

    /// Get linearly filtered light intensity for 4 world positions
    pub fn get_filtered_light_intensity_vec(&self, pos_world: Vec3x4) -> Vec4 {
        // Convert world positions to voxel coordinates
        let voxel_size = self.voxel_size();
        let voxel_x_f = (pos_world.x - self.world_min.x) / voxel_size.x;
        let voxel_y_f = (pos_world.y - self.world_min.y) / voxel_size.y;
        let voxel_z_f = (pos_world.z - self.world_min.z) / voxel_size.z;

        // Floor to get voxel coordinates
        let x0_f = voxel_x_f.floor();
        let y0_f = voxel_y_f.floor();
        let z0_f = voxel_z_f.floor();

        // Convert to usize
        let x0_usize = USizeVec4::new(
            x0_f.x as usize,
            x0_f.y as usize,
            x0_f.z as usize,
            x0_f.w as usize,
        );
        let y0_usize = USizeVec4::new(
            y0_f.x as usize,
            y0_f.y as usize,
            y0_f.z as usize,
            y0_f.w as usize,
        );
        let z0_usize = USizeVec4::new(
            z0_f.x as usize,
            z0_f.y as usize,
            z0_f.z as usize,
            z0_f.w as usize,
        );

        let x0 = x0_usize.min(USizeVec4::splat(self.width - 1));
        let y0 = y0_usize.min(USizeVec4::splat(self.height - 1));
        let z0 = z0_usize.min(USizeVec4::splat(self.depth - 1));

        let x1 = (x0 + USizeVec4::ONE).min(USizeVec4::splat(self.width - 1));
        let y1 = (y0 + USizeVec4::ONE).min(USizeVec4::splat(self.height - 1));
        let z1 = (z0 + USizeVec4::ONE).min(USizeVec4::splat(self.depth - 1));

        // Compute fractional weights
        let fx = voxel_x_f - x0_f;
        let fy = voxel_y_f - y0_f;
        let fz = voxel_z_f - z0_f;

        // Helper to gather 8 voxel values
        let sample_grid = |x: USizeVec4, y: USizeVec4, z: USizeVec4| -> Vec4 {
            Vec4::new(
                self.get_light_intensity(x.x, y.x, z.x),
                self.get_light_intensity(x.y, y.y, z.y),
                self.get_light_intensity(x.z, y.z, z.z),
                self.get_light_intensity(x.w, y.w, z.w),
            )
        };

        // Gather 8 surrounding voxels
        let v000 = sample_grid(x0, y0, z0);
        let v001 = sample_grid(x0, y0, z1);
        let v010 = sample_grid(x0, y1, z0);
        let v011 = sample_grid(x0, y1, z1);
        let v100 = sample_grid(x1, y0, z0);
        let v101 = sample_grid(x1, y0, z1);
        let v110 = sample_grid(x1, y1, z0);
        let v111 = sample_grid(x1, y1, z1);

        // Trilinear interpolation
        let v00 = v000 * (Vec4::ONE - fx) + v100 * fx;
        let v01 = v001 * (Vec4::ONE - fx) + v101 * fx;
        let v10 = v010 * (Vec4::ONE - fx) + v110 * fx;
        let v11 = v011 * (Vec4::ONE - fx) + v111 * fx;

        let v0 = v00 * (Vec4::ONE - fy) + v10 * fy;
        let v1 = v01 * (Vec4::ONE - fy) + v11 * fy;

        v0 * (Vec4::ONE - fz) + v1 * fz
    }

    pub fn get_filtered_gi_sh4_vec(&self, pos_world: Vec3x4) -> [Vec3x4; 4] {
        let Some(gi_data) = &self.gi_sh4 else {
            return [Vec3x4::ZERO; 4];
        };

        let voxel_size = self.voxel_size();
        let voxel_x_f = (pos_world.x - self.world_min.x) / voxel_size.x;
        let voxel_y_f = (pos_world.y - self.world_min.y) / voxel_size.y;
        let voxel_z_f = (pos_world.z - self.world_min.z) / voxel_size.z;

        let x0_f = voxel_x_f.floor();
        let y0_f = voxel_y_f.floor();
        let z0_f = voxel_z_f.floor();

        let x0_usize = USizeVec4::new(
            x0_f.x as usize,
            x0_f.y as usize,
            x0_f.z as usize,
            x0_f.w as usize,
        );
        let y0_usize = USizeVec4::new(
            y0_f.x as usize,
            y0_f.y as usize,
            y0_f.z as usize,
            y0_f.w as usize,
        );
        let z0_usize = USizeVec4::new(
            z0_f.x as usize,
            z0_f.y as usize,
            z0_f.z as usize,
            z0_f.w as usize,
        );

        let x0 = x0_usize.min(USizeVec4::splat(self.width - 1));
        let y0 = y0_usize.min(USizeVec4::splat(self.height - 1));
        let z0 = z0_usize.min(USizeVec4::splat(self.depth - 1));

        let x1 = (x0 + USizeVec4::ONE).min(USizeVec4::splat(self.width - 1));
        let y1 = (y0 + USizeVec4::ONE).min(USizeVec4::splat(self.height - 1));
        let z1 = (z0 + USizeVec4::ONE).min(USizeVec4::splat(self.depth - 1));

        let fx = voxel_x_f - x0_f;
        let fy = voxel_y_f - y0_f;
        let fz = voxel_z_f - z0_f;

        let sample_grid = |coeff_idx: usize, x: USizeVec4, y: USizeVec4, z: USizeVec4| -> Vec3x4 {
            let a = Self::get_gi_coeff(gi_data, coeff_idx, x.x, y.x, z.x, self.width, self.height);
            let b = Self::get_gi_coeff(gi_data, coeff_idx, x.y, y.y, z.y, self.width, self.height);
            let c = Self::get_gi_coeff(gi_data, coeff_idx, x.z, y.z, z.z, self.width, self.height);
            let d = Self::get_gi_coeff(gi_data, coeff_idx, x.w, y.w, z.w, self.width, self.height);
            Vec3x4::new(
                Vec4::new(a.x, b.x, c.x, d.x),
                Vec4::new(a.y, b.y, c.y, d.y),
                Vec4::new(a.z, b.z, c.z, d.z),
            )
        };

        let mut out = [Vec3x4::ZERO; 4];
        for (coeff_idx, coeff_out) in out.iter_mut().enumerate() {
            let v000 = sample_grid(coeff_idx, x0, y0, z0);
            let v001 = sample_grid(coeff_idx, x0, y0, z1);
            let v010 = sample_grid(coeff_idx, x0, y1, z0);
            let v011 = sample_grid(coeff_idx, x0, y1, z1);
            let v100 = sample_grid(coeff_idx, x1, y0, z0);
            let v101 = sample_grid(coeff_idx, x1, y0, z1);
            let v110 = sample_grid(coeff_idx, x1, y1, z0);
            let v111 = sample_grid(coeff_idx, x1, y1, z1);

            let v00 = v000 * (Vec4::ONE - fx) + v100 * fx;
            let v01 = v001 * (Vec4::ONE - fx) + v101 * fx;
            let v10 = v010 * (Vec4::ONE - fx) + v110 * fx;
            let v11 = v011 * (Vec4::ONE - fx) + v111 * fx;
            let v0 = v00 * (Vec4::ONE - fy) + v10 * fy;
            let v1 = v01 * (Vec4::ONE - fy) + v11 * fy;
            *coeff_out = v0 * (Vec4::ONE - fz) + v1 * fz;
        }
        out
    }

    /// Blur light intensity for a voxel with a 3x3x3 filter
    fn blur_intensity(&self, x: usize, y: usize, z: usize) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;

        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;

                    // Check bounds
                    if nx >= 0
                        && nx < self.width as i32
                        && ny >= 0
                        && ny < self.height as i32
                        && nz >= 0
                        && nz < self.depth as i32
                    {
                        sum += self.get_light_intensity(nx as usize, ny as usize, nz as usize);
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            (sum / count as f32).powf(2.0)
        } else {
            0.0
        }
    }

    /// Simple 3x3x3 blur filter to smooth light intensity values
    pub fn blur_grid(&mut self) {
        let mut blurred = vec![0.0; self.light_intensity.len()];

        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    let intensity = self.blur_intensity(x, y, z);
                    let index = self.voxel_index(x, y, z);
                    blurred[index] = intensity;
                }
            }
        }

        self.light_intensity = blurred;
    }

    /// Simple 3x3x3 blur filter to smooth GI SH4 coefficients.
    pub fn blur_gi_sh4(&mut self) {
        let Some(gi_data) = self.gi_sh4.as_ref() else {
            return;
        };

        let source = gi_data.clone();
        let mut blurred = vec![[Vec3A::ZERO; 4]; source.len()];

        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    let index = self.voxel_index(x, y, z);
                    let mut sum = [Vec3A::ZERO; 4];
                    let mut count = 0.0f32;

                    for dz in -1..=1 {
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;
                                let nz = z as i32 + dz;

                                if nx < 0
                                    || nx >= self.width as i32
                                    || ny < 0
                                    || ny >= self.height as i32
                                    || nz < 0
                                    || nz >= self.depth as i32
                                {
                                    continue;
                                }

                                let nindex =
                                    self.voxel_index(nx as usize, ny as usize, nz as usize);
                                for c in 0..4 {
                                    sum[c] += source[nindex][c];
                                }
                                count += 1.0;
                            }
                        }
                    }

                    if count > 0.0 {
                        for c in 0..4 {
                            blurred[index][c] = sum[c] / count;
                        }
                    } else {
                        blurred[index] = source[index];
                    }
                }
            }
        }

        self.gi_sh4 = Some(blurred);
    }
}
