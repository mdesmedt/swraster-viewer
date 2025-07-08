use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use gltf::camera::Projection;
use std::f32::consts::PI;

pub struct RenderCamera {
    pub position: Vec3,
    fov: f32,
    width: f32,
    height: f32,
    near: f32,
    far: f32,
    yaw: f32,   // Rotation around Y axis
    pitch: f32, // Rotation around X axis
    roll: f32,  // Rotation around Z axis
    // Cached matrices and planes
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
    pub view_project_matrix: Mat4,
    pub inverse_view_project_matrix: Mat4,
    pub view_clip_planes: [Vec4; 6],
    pub view_normal: Vec3,
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
            near: 0.01,
            far: 200.0,
            yaw,
            pitch,
            roll: 0.0,
            view_matrix: Mat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
            view_project_matrix: Mat4::IDENTITY,
            inverse_view_project_matrix: Mat4::IDENTITY,
            view_clip_planes: [Vec4::ZERO; 6],
            view_normal: Vec3::ZERO,
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

    pub fn update_matrices(&mut self) {
        self.view_matrix = self.compute_view_matrix();
        self.projection_matrix = self.compute_projection_matrix();
        self.view_project_matrix = self.projection_matrix * self.view_matrix;
        self.inverse_view_project_matrix = self.view_project_matrix.inverse();
        self.view_clip_planes = self.compute_view_clip_planes();
        self.view_normal = self.get_rotation().conjugate() * Vec3::new(0.0, 0.0, 1.0);
    }

    fn compute_view_matrix(&self) -> Mat4 {
        // Get rotation matrix from quaternion
        let rotation_matrix: Mat4 = Mat4::from_quat(self.get_rotation());

        // Create translation matrix
        let translation_matrix = Mat4::from_translation(-self.position);

        rotation_matrix * translation_matrix
    }

    fn compute_projection_matrix(&self) -> Mat4 {
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

    pub fn rotate_mouse(&mut self, rotation: Vec2) {
        // Update yaw and pitch from mouse movement
        self.yaw += rotation.x * 0.01;
        self.pitch = (self.pitch + rotation.y * 0.01).clamp(-PI / 2.0 + 0.1, PI / 2.0 - 0.1);
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

    fn compute_view_clip_planes(&self) -> [Vec4; 6] {
        let aspect_ratio = self.width / self.height;
        let tan_y = (self.fov * 0.5).tan();
        let tan_x = tan_y * aspect_ratio;

        let near = Vec4::new(0.0, 0.0, -1.0, self.near);
        let far = Vec4::new(0.0, 0.0, 1.0, self.far);
        let right = Vec3::new(1.0, 0.0, -tan_x).normalize().extend(0.0);
        let left = Vec3::new(-1.0, 0.0, -tan_x).normalize().extend(0.0);
        let top = Vec3::new(0.0, 1.0, -tan_y).normalize().extend(0.0);
        let bottom = Vec3::new(0.0, -1.0, -tan_y).normalize().extend(0.0);

        [near, far, right, left, top, bottom]
    }
}
