use clap::Parser;
use glam::{Vec2, Vec3, Vec3A};
use gltf::Gltf;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use pixels::{Pixels, SurfaceTexture};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

mod bumpqueue;
mod math;
mod raytracer;
mod rendercamera;
mod renderer;
mod texture;
mod tilerasterizer;
mod voxelgrid;

// TODO: Remove dead code in the GLTF scene boilerplate module. This is just to stop rustc from complaining.
#[allow(dead_code)]
mod scene;

use raytracer::RayTracer;
use rendercamera::RenderCamera;
use renderer::{RenderBuffer, Renderer};
use scene::Scene;
use texture::TextureCache;
use voxelgrid::VoxelGrid;

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const CAMERA_SPEED: f32 = 0.15; // Speed relative to scene size per second
const CAMERA_SPEED_FAST: f32 = 0.5; // Speed when shift is pressed
const KEY_ROTATION_SPEED: f32 = 75.0;
const KEY_ROTATION_SPEED_FAST: f32 = 150.0;

#[derive(Parser, Clone)]
#[command(name = "swrast")]
#[command(about = "Software rasterizer viewer for GLTF files")]
struct Settings {
    /// GLTF file to load
    #[arg(default_value = "glTF-Sample-Assets/Models/FlightHelmet/glTF/FlightHelmet.gltf")]
    file: String,

    /// Disable shadow computation
    #[arg(long)]
    no_shadow: bool,

    /// Disable vsync
    #[arg(long)]
    disable_vsync: bool,
}

struct RenderState {
    scene: Scene,
    camera: RenderCamera,
}

enum LoadingState {
    Loading,
    Loaded(RenderState),
    Error(String),
}

// Load the GLTF document and buffers, but skip loading images.
fn import_gltf(
    Gltf { document, blob }: Gltf,
    base: Option<&Path>,
) -> Result<(gltf::Document, Vec<gltf::buffer::Data>), gltf::Error> {
    let buffer_data = gltf::import_buffers(&document, base, blob)?;
    Ok((document, buffer_data))
}

fn load_gltf(path: &Path) -> Result<(gltf::Document, Vec<gltf::buffer::Data>), gltf::Error> {
    let base = path.parent().unwrap_or_else(|| Path::new("./"));
    let file = fs::File::open(path).map_err(gltf::Error::Io)?;
    let reader = io::BufReader::new(file);
    import_gltf(Gltf::from_reader(reader)?, Some(base))
}

fn load_scene(gltf_path: &Path, loading_state: &Arc<Mutex<LoadingState>>, settings: &Settings) {
    println!("Loading: {}", gltf_path.display());

    // Load the GLTF file
    let (document, buffers) = match load_gltf(&gltf_path) {
        Ok(result) => result,
        Err(e) => {
            // Something failed during loading
            if let Ok(mut state) = loading_state.lock() {
                *state = LoadingState::Error(format!("Failed to load GLTF file: {}", e));
            }
            return;
        }
    };

    // Get the default scene or first scene
    let gltf_scene = match document
        .default_scene()
        .or_else(|| document.scenes().next())
    {
        Some(scene) => scene,
        None => {
            if let Ok(mut state) = loading_state.lock() {
                *state = LoadingState::Error("No scenes found in GLTF file".to_string());
            }
            return;
        }
    };

    // Extract base directory from GLTF file path for texture loading
    let base_dir = gltf_path
        .parent()
        .and_then(|p| p.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| ".".to_string());

    // Create texture cache for the scene
    let mut texture_cache = TextureCache::new(base_dir);
    let mut scene = match Scene::from_gltf(&document, &gltf_scene, &buffers, &mut texture_cache) {
        Ok(scene) => scene,
        Err(e) => {
            if let Ok(mut state) = loading_state.lock() {
                *state = LoadingState::Error(format!("Failed to create scene from GLTF: {}", e));
            }
            return;
        }
    };

    // Print scene statistics
    let total_primitives: usize = scene
        .nodes
        .iter()
        .map(|node| {
            node.mesh_index
                .map(|mesh_index| scene.meshes[mesh_index].primitives.len())
                .unwrap_or(0)
        })
        .sum();
    let total_triangles: usize = scene
        .nodes
        .iter()
        .map(|node| {
            node.mesh_index
                .map(|mesh_index| {
                    scene.meshes[mesh_index]
                        .primitives
                        .iter()
                        .map(|primitive| primitive.indices.len() / 3)
                        .sum::<usize>()
                })
                .unwrap_or(0)
        })
        .sum();

    // Calculate texture statistics
    let unique_textures = texture_cache.unique_texture_count();
    let total_texture_data_mib =
        (texture_cache.total_texture_data_size() as f64) / (1024.0 * 1024.0);

    println!("Scene Statistics:");
    println!("  Nodes: {}", scene.nodes.len());
    println!("  Meshes: {}", scene.meshes.len());
    println!("  Primitives: {}", total_primitives);
    println!("  Total Triangles: {}", total_triangles);
    println!("  Unique Textures: {}", unique_textures);
    println!("  Total Texture Data: {:.2} MiB", total_texture_data_mib);

    // Create a default camera if none exists in the file
    let camera = if let Some(camera) = document.cameras().next() {
        match RenderCamera::from_gltf(&camera, WIDTH as f32, HEIGHT as f32) {
            Some(camera) => camera,
            None => {
                if let Ok(mut state) = loading_state.lock() {
                    *state = LoadingState::Error(
                        "Failed to create camera from GLTF: unsupported camera type".to_string(),
                    );
                }
                return;
            }
        }
    } else {
        // Position camera at a distance proportional to scene size
        let camera_distance = scene.bounds.diagonal;
        let camera_pos = scene.bounds.center + Vec3A::new(0.0, 0.0, camera_distance);

        RenderCamera::new(
            Vec3::from(camera_pos),
            Vec3::from(scene.bounds.center),
            std::f32::consts::PI / 4.0,
            WIDTH as f32,
            HEIGHT as f32,
        )
    };

    // Create and populate a voxel grid if shadows are enabled
    if !settings.no_shadow {
        // Create raytracer from the scene
        println!("Creating raytracer");
        let raytracer = RayTracer::new(&scene);

        // Use 64x64x64 voxels
        const GRID_SIZE: usize = 128;

        let mut voxel_grid = VoxelGrid::new(
            GRID_SIZE,
            GRID_SIZE,
            GRID_SIZE,
            Vec3::from(scene.bounds.min),
            Vec3::from(scene.bounds.max),
        );

        // Fill voxel grid with light intensity based on visibility of the voxel center to the light
        println!(
            "Computing light intensity for {} voxels",
            GRID_SIZE * GRID_SIZE * GRID_SIZE
        );
        let light_direction = scene.light.direction;
        let ray_dir = light_direction.normalize();
        let voxel_size = voxel_grid.voxel_size();
        let center_min = voxel_grid.world_min() + voxel_size * 0.5;

        voxel_grid.par_iter_mut().for_each(|(coords, intensity)| {
            // Compute ray origin with some bias to avoid self-shadowing
            let voxel_center = center_min + coords.as_vec3() * voxel_size;
            let ray_origin = voxel_center + voxel_size * ray_dir * 3.0;

            // Trace ray towards the light
            *intensity = if raytracer.ray_intersect(ray_origin, ray_dir) {
                0.0 // Light is blocked by scene geometry
            } else {
                1.0 // Light reaches this voxel
            };
        });

        // Apply simple blur to smooth the voxel grid
        voxel_grid.blur_grid();

        println!("Voxel grid ready");
        scene.voxel_grid = Some(voxel_grid);
    } else {
        scene.voxel_grid = None;
    }

    // Create and store the render state
    let render_state = RenderState { scene, camera };
    if let Ok(mut state) = loading_state.lock() {
        *state = LoadingState::Loaded(render_state);
    }
}

struct InputState {
    keys_down: HashSet<KeyCode>,
    mouse_down: bool,
    mouse_position: Vec2,
    mouse_delta: Vec2,
}

impl InputState {
    fn is_key_down(&self, key_code: KeyCode) -> bool {
        return self.keys_down.contains(&key_code);
    }
}

struct App {
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'static>>,
    loading_state: Arc<Mutex<LoadingState>>,
    filename: String,
    start_time: Instant,
    renderer: Renderer,
    frame_count: u32,
    last_fps_update: Instant,
    last_frame_time: Instant,
    input_state: InputState,
    settings: Settings,
}

impl App {
    fn new() -> Self {
        // Parse command line arguments
        let settings = Settings::parse();

        // Get the GLTF file path from settings
        let gltf_path = Path::new(&settings.file);

        // Extract filename
        let filename = gltf_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");

        // Save the start time
        let start_time = Instant::now();

        // Create the renderer
        let renderer = Renderer::new(WIDTH as i32, HEIGHT as i32);

        // Create shared state for background loading
        let loading_state = Arc::new(Mutex::new(LoadingState::Loading));
        let loading_state_clone = loading_state.clone();

        // Start background thread for scene loading
        let gltf_path = gltf_path.to_path_buf(); // Clone the path for the thread
        let settings_clone = settings.clone();
        thread::spawn(move || {
            load_scene(&gltf_path, &loading_state_clone, &settings_clone);
        });

        // FPS measurement variables
        let frame_count = 0;
        let last_fps_update = Instant::now();
        let last_frame_time = Instant::now();

        let input_state = InputState {
            keys_down: HashSet::new(),
            mouse_down: false,
            mouse_position: Vec2::ZERO,
            mouse_delta: Vec2::ZERO,
        };

        Self {
            window: None,
            pixels: None,
            renderer,
            loading_state,
            filename: filename.to_string(),
            start_time,
            frame_count,
            last_fps_update,
            last_frame_time,
            input_state,
            settings,
        }
    }

    fn update_keyboard(&mut self, _device_id: DeviceId, event: KeyEvent) {
        if event.repeat {
            return;
        }
        if let Some(key_code) = match event.physical_key {
            PhysicalKey::Code(code) => Some(code),
            PhysicalKey::Unidentified(_) => None,
        } {
            if event.state == ElementState::Pressed {
                self.input_state.keys_down.insert(key_code);
            } else {
                self.input_state.keys_down.remove(&key_code);
            }
        }
    }

    fn update_mouse_button(
        &mut self,
        _device_id: DeviceId,
        state: ElementState,
        _button: MouseButton,
    ) {
        if state == ElementState::Pressed {
            self.input_state.mouse_down = true;
            self.input_state.mouse_delta = Vec2::ZERO;
        } else {
            self.input_state.mouse_down = false;
        }
    }

    fn update_mouse_position(&mut self, position: (f64, f64)) {
        self.input_state.mouse_position = Vec2::new(position.0 as f32, position.1 as f32);
    }

    fn update_mouse_motion(&mut self, delta: (f64, f64)) {
        self.input_state.mouse_delta += Vec2::new(delta.0 as f32, delta.1 as f32);
    }

    fn render_frame(&mut self, event_loop: &ActiveEventLoop) {
        // Compute delta time
        let current_time = Instant::now();
        let delta_time = current_time
            .duration_since(self.last_frame_time)
            .as_secs_f32();
        self.last_frame_time = current_time;

        if self.input_state.is_key_down(KeyCode::Escape) {
            event_loop.exit();
            return;
        }

        // Try to get the render state
        let mut loading_state_guard = self.loading_state.lock().unwrap();
        let mut loading_complete = false;

        match &mut *loading_state_guard {
            LoadingState::Loaded(render_state) => {
                // Scene is loaded, handle input and render
                let camera = &mut render_state.camera;

                // Handle keyboard input for camera movement
                let mut move_dir = Vec3::ZERO;

                if self.input_state.is_key_down(KeyCode::KeyW) {
                    move_dir.z -= 1.0; // Forward (negative Z in camera space)
                }
                if self.input_state.is_key_down(KeyCode::KeyS) {
                    move_dir.z += 1.0; // Backward (positive Z in camera space)
                }
                if self.input_state.is_key_down(KeyCode::KeyA) {
                    move_dir.x -= 1.0; // Left (negative X in camera space)
                }
                if self.input_state.is_key_down(KeyCode::KeyD) {
                    move_dir.x += 1.0; // Right (positive X in camera space)
                }
                if self.input_state.is_key_down(KeyCode::KeyE) {
                    move_dir.y += 1.0; // Up (positive Y in camera space)
                }
                if self.input_state.is_key_down(KeyCode::KeyQ) {
                    move_dir.y -= 1.0; // Down (negative Y in camera space)
                }

                // Handle camera rotation using the arrow keys
                let key_rotation_speed = if self.input_state.is_key_down(KeyCode::ShiftLeft)
                    || self.input_state.is_key_down(KeyCode::ShiftRight)
                {
                    KEY_ROTATION_SPEED_FAST * delta_time
                } else {
                    KEY_ROTATION_SPEED * delta_time
                };
                if self.input_state.is_key_down(KeyCode::ArrowLeft) {
                    camera.rotate_mouse(Vec2::new(-key_rotation_speed, 0.0));
                }
                if self.input_state.is_key_down(KeyCode::ArrowRight) {
                    camera.rotate_mouse(Vec2::new(key_rotation_speed, 0.0));
                }
                if self.input_state.is_key_down(KeyCode::ArrowUp) {
                    camera.rotate_mouse(Vec2::new(0.0, -key_rotation_speed));
                }
                if self.input_state.is_key_down(KeyCode::ArrowDown) {
                    camera.rotate_mouse(Vec2::new(0.0, key_rotation_speed));
                }

                let scene_size = render_state.scene.bounds.diagonal;

                // Normalize movement direction if moving diagonally
                if move_dir.length_squared() > 0.0 {
                    move_dir = move_dir.normalize();
                    // Use faster speed when shift is pressed
                    let speed = if self.input_state.is_key_down(KeyCode::ShiftLeft)
                        || self.input_state.is_key_down(KeyCode::ShiftRight)
                    {
                        CAMERA_SPEED_FAST * scene_size
                    } else {
                        CAMERA_SPEED * scene_size
                    };
                    camera.move_relative(move_dir, speed * delta_time);
                }

                // Handle mouse input for camera rotation
                if self.input_state.mouse_down {
                    const ROTATION_FACTOR: Vec2 = Vec2::new(0.4, 0.3);
                    let delta = self.input_state.mouse_delta;
                    camera.rotate_mouse(delta * ROTATION_FACTOR);
                    self.input_state.mouse_delta = Vec2::ZERO;
                }

                // Update camera matrices
                camera.update_matrices();

                // Render the scene
                self.renderer.render_scene(&render_state.scene, camera);

                loading_complete = true;
            }
            LoadingState::Loading => {
                // Do nothing
            }
            LoadingState::Error(error_msg) => {
                // Scene failed to load, exit
                eprintln!("Scene loading failed: {}", error_msg);
                event_loop.exit();
                return;
            }
        }

        // Grab the buffer from pixels
        let pixels = self.pixels.as_mut().unwrap();
        let buffer = pixels.frame_mut();
        let mut render_buffer = RenderBuffer::new(WIDTH, HEIGHT, buffer);

        if loading_complete {
            // Blit the rendered tiles to the buffer
            self.renderer.blit_to_buffer(&mut render_buffer);
        } else {
            // Still loading, draw quaint little loading indicator
            render_buffer.clear();
            let center_x = WIDTH as i32 / 2;
            let center_y = HEIGHT as i32 / 2;
            let radius = 50;
            let time = current_time.duration_since(self.start_time).as_secs_f64();
            let angle = time * 2.0 * std::f64::consts::PI; // One rotation per second
            const NUM_POINTS: usize = 32;
            const ANGLE_STEP: f64 = (std::f64::consts::PI * 0.5) / NUM_POINTS as f64;
            for i in 0..NUM_POINTS {
                let point_angle = (i as f64 * ANGLE_STEP) + angle;
                let x = center_x as i32 + (point_angle.cos() * radius as f64) as i32;
                let y = center_y as i32 + (point_angle.sin() * radius as f64) as i32;
                let px = x as usize;
                let py = y as usize;
                let intensity = i as f64 / NUM_POINTS as f64;
                let intensity_int = (intensity * 255.0) as u8;
                render_buffer.set_pixel(px, py, 0, intensity_int);
                render_buffer.set_pixel(px, py, 1, intensity_int);
                render_buffer.set_pixel(px, py, 2, intensity_int);
                render_buffer.set_pixel(px, py, 3, 0xFF);
            }
        }

        if let Err(err) = pixels.render() {
            eprintln!("Pixels render error: {}", err);
            event_loop.exit();
            return;
        }

        // Update FPS counter
        self.frame_count += 1;
        let elapsed = current_time.duration_since(self.last_fps_update);
        if elapsed.as_secs_f64() >= 1.0 {
            let fps = self.frame_count as f64 / elapsed.as_secs_f64();
            self.window
                .as_ref()
                .unwrap()
                .set_title(&format!("GLTF Viewer - {} - {:.1} FPS", self.filename, fps,));
            self.frame_count = 0;
            self.last_fps_update = current_time;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create window
        let window_size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        let window_attributes = Window::default_attributes()
            .with_inner_size(window_size)
            .with_title(format!("GLTF Viewer - {}", self.filename));
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        // Init pixels
        let window_size = window.inner_size();
        let surface_texture =
            SurfaceTexture::new(window_size.width, window_size.height, window.clone());
        let mut pixels = Pixels::new(WIDTH as u32, HEIGHT as u32, surface_texture).unwrap();

        if self.settings.disable_vsync {
            pixels.enable_vsync(false);
        }

        // Set scaling mode
        pixels.set_scaling_mode(pixels::ScalingMode::Fill);

        // Clear alpha to FF once, never touch it again
        // TODO: Might be faster to just write alpha, possibly with uint64_t or so?
        pixels.frame_mut().chunks_exact_mut(4).for_each(|pixel| {
            pixel[3] = 0xff;
        });

        self.window = Some(window);
        self.pixels = Some(pixels);

        // Request initial redraw
        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                device_id,
                event,
                is_synthetic,
            } => {
                if is_synthetic {
                    return;
                }
                self.update_keyboard(device_id, event);
            }
            WindowEvent::MouseInput {
                device_id,
                state,
                button,
            } => {
                self.update_mouse_button(device_id, state, button);
            }
            WindowEvent::CursorMoved {
                device_id: _,
                position,
            } => {
                self.update_mouse_position((position.x, position.y));
            }
            WindowEvent::RedrawRequested => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.
                self.render_frame(event_loop);
            }
            WindowEvent::Resized(physical_size) => {
                self.pixels
                    .as_mut()
                    .unwrap()
                    .resize_surface(physical_size.width, physical_size.height)
                    .unwrap();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let inner_size = self.window.as_ref().unwrap().inner_size();
                self.pixels
                    .as_mut()
                    .unwrap()
                    .resize_surface(inner_size.width, inner_size.height)
                    .unwrap();
            }
            _ => (),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.update_mouse_motion(delta);
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // After processing all other events, request a redraw
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
