use glam::{Mat4, Quat, Vec3};
use gltf::camera::Projection;
use std::f32::consts::PI;

pub struct RenderCamera {
    position: Vec3,
    fov: f32,
    width: f32,
    height: f32,
    near: f32,
    far: f32,
    yaw: f32,   // Rotation around Y axis
    pitch: f32, // Rotation around X axis
    roll: f32,  // Rotation around Z axis
}

impl RenderCamera {
    pub fn new(position: Vec3, look_at: Vec3, fov: f32, width: f32, height: f32) -> Self {
        // Calculate the forward direction (from position to look_at)
        let forward = (look_at - position).normalize();

        // Calculate yaw and pitch from forward direction
        let yaw: f32 = forward.x.atan2(-forward.z);
        let pitch = forward.y.asin();

        Self {
            position,
            fov,
            width,
            height,
            near: 0.1,
            far: 100.0,
            yaw,
            pitch,
            roll: 0.0,
        }
    }

    pub fn from_gltf(camera: &gltf::Camera, width: f32, height: f32) -> Option<Self> {
        match camera.projection() {
            Projection::Perspective(perspective) => {
                let fov = perspective.yfov();
                let position = Vec3::new(0.0, 0.0, 5.0); // Default position, should be set by node transform
                let look_at = Vec3::new(0.0, 0.0, 0.0); // Default look at, should be set by node transform

                Some(Self::new(position, look_at, fov, width, height))
            }
            Projection::Orthographic(_) => None, // We don't support orthographic cameras yet
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        // Get rotation matrix from quaternion
        let rotation_matrix: Mat4 = Mat4::from_quat(self.get_rotation());

        // Create translation matrix
        let translation_matrix = Mat4::from_translation(-self.position);

        rotation_matrix * translation_matrix
    }

    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.width / self.height, self.near, self.far)
    }

    pub fn move_relative(&mut self, direction: Vec3, magnitude: f32) {
        // Get rotation quaternion
        let rotation = self.get_rotation();

        // Convert direction from camera space to world space
        let world_dir = rotation.conjugate() * direction;

        // Apply movement
        self.position += world_dir * magnitude;
    }

    pub fn rotate_mouse(&mut self, dx: f32, dy: f32) {
        // Update yaw and pitch from mouse movement
        self.yaw += dx * 0.01;
        self.pitch = (self.pitch + dy * 0.01).clamp(-PI / 2.0 + 0.1, PI / 2.0 - 0.1);
    }

    pub fn get_rotation(&self) -> Quat {
        // Create quaternions for each axis rotation
        let q_yaw = Quat::from_rotation_y(self.yaw);
        let q_pitch = Quat::from_rotation_x(self.pitch);
        let q_roll = Quat::from_rotation_z(self.roll);

        // Combine rotations in the same order as our matrix multiplication
        // roll * pitch * yaw
        q_roll * q_pitch * q_yaw
    }
}
