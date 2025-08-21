use glam::{UVec3, Vec3, Vec4};
use rayon::prelude::*;

/// A 3D voxel grid that stores light intensity values for each voxel
pub struct VoxelGrid {
    width: usize,
    height: usize,
    depth: usize,
    world_min: Vec3,
    world_max: Vec3,
    light_intensity: Vec<f32>,
}

impl VoxelGrid {
    /// Create a new voxel grid with the specified dimensions and world bounds
    pub fn new(
        width: usize,
        height: usize,
        depth: usize,
        world_min: Vec3,
        world_max: Vec3,
    ) -> Self {
        let light_intensity = vec![0.0; width * height * depth];
        Self {
            width,
            height,
            depth,
            world_min,
            world_max,
            light_intensity,
        }
    }

    /// Returns a parallel iterator over mutable light intensity values with their coordinates
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
    pub fn world_min(&self) -> Vec3 {
        self.world_min
    }

    /// Get the size of a voxel in world space
    pub fn voxel_size(&self) -> Vec3 {
        let world_size = self.world_max - self.world_min;
        Vec3::new(
            world_size.x / self.width as f32,
            world_size.y / self.height as f32,
            world_size.z / self.depth as f32,
        )
    }

    /// Get the light intensity at the specified voxel coordinates
    pub fn get_light_intensity(&self, x: usize, y: usize, z: usize) -> f32 {
        let index = z * self.width * self.height + y * self.width + x;
        self.light_intensity[index]
    }

    /// Get light intensity for 4 world positions, returning a Vec4
    /// If positions are out of bounds, returns 1.0 for those components
    pub fn _get_light_intensity_vec(
        &self,
        pos_world_x: Vec4,
        pos_world_y: Vec4,
        pos_world_z: Vec4,
    ) -> Vec4 {
        let mut result = Vec4::ZERO;

        // Convert world position to voxel coordinates (relative to world_min)
        let local_x = pos_world_x - self.world_min.x;
        let local_y = pos_world_y - self.world_min.y;
        let local_z = pos_world_z - self.world_min.z;

        let voxel_size = self.voxel_size();
        let voxel_x_f = local_x / voxel_size.x;
        let voxel_y_f = local_y / voxel_size.y;
        let voxel_z_f = local_z / voxel_size.z;

        let voxel_x_arr = voxel_x_f.to_array();
        let voxel_y_arr = voxel_y_f.to_array();
        let voxel_z_arr = voxel_z_f.to_array();

        for i in 0..4 {
            let voxel_x = voxel_x_arr[i] as usize;
            let voxel_y = voxel_y_arr[i] as usize;
            let voxel_z = voxel_z_arr[i] as usize;

            // Check bounds and get intensity
            if voxel_x < self.width && voxel_y < self.height && voxel_z < self.depth {
                result[i] = self.get_light_intensity(voxel_x, voxel_y, voxel_z);
            } else {
                result[i] = 1.0; // Out of bounds = lit
            }
        }

        result
    }

    /// Get linearly filtered light intensity for 4 world positions
    pub fn get_filtered_light_intensity_vec(
        &self,
        pos_world_x: Vec4,
        pos_world_y: Vec4,
        pos_world_z: Vec4,
    ) -> Vec4 {
        // Convert world positions to voxel coordinates
        let local_x = pos_world_x - self.world_min.x;
        let local_y = pos_world_y - self.world_min.y;
        let local_z = pos_world_z - self.world_min.z;

        let voxel_size = self.voxel_size();
        let voxel_x_f = local_x / voxel_size.x;
        let voxel_y_f = local_y / voxel_size.y;
        let voxel_z_f = local_z / voxel_size.z;

        let mut result = Vec4::ZERO;

        // TODO: Vectorize this better
        for i in 0..4 {
            let x = voxel_x_f[i];
            let y = voxel_y_f[i];
            let z = voxel_z_f[i];

            // Get the 8 surrounding voxel coordinates
            let x0 = x.floor() as usize;
            let y0 = y.floor() as usize;
            let z0 = z.floor() as usize;
            let x1 = (x0 + 1).min(self.width - 1);
            let y1 = (y0 + 1).min(self.height - 1);
            let z1 = (z0 + 1).min(self.depth - 1);

            if x0 < self.width && y0 < self.height && z0 < self.depth {
                // Compute weights
                let fx = x - x0 as f32;
                let fy = y - y0 as f32;
                let fz = z - z0 as f32;

                // Get the 8 voxel values
                let v000 = self.get_light_intensity(x0, y0, z0);
                let v001 = self.get_light_intensity(x0, y0, z1);
                let v010 = self.get_light_intensity(x0, y1, z0);
                let v011 = self.get_light_intensity(x0, y1, z1);
                let v100 = self.get_light_intensity(x1, y0, z0);
                let v101 = self.get_light_intensity(x1, y0, z1);
                let v110 = self.get_light_intensity(x1, y1, z0);
                let v111 = self.get_light_intensity(x1, y1, z1);

                // Perform trilinear interpolation
                let v00 = v000 * (1.0 - fx) + v100 * fx;
                let v01 = v001 * (1.0 - fx) + v101 * fx;
                let v10 = v010 * (1.0 - fx) + v110 * fx;
                let v11 = v011 * (1.0 - fx) + v111 * fx;

                let v0 = v00 * (1.0 - fy) + v10 * fy;
                let v1 = v01 * (1.0 - fy) + v11 * fy;

                let interpolated = v0 * (1.0 - fz) + v1 * fz;
                result[i] = interpolated;
            } else {
                result[i] = 1.0; // Out of bounds = lit
            }
        }

        result
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
                    let index = z * self.width * self.height + y * self.width + x;
                    blurred[index] = intensity;
                }
            }
        }

        self.light_intensity = blurred;
    }
}
