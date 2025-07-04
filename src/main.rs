use glam::{Vec3, Vec3A};
use gltf::Gltf;
use minifb::{Key, MouseButton, Window, WindowOptions};
use std::env;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

mod bumpqueue;
mod math;
mod rendercamera;
mod renderer;
mod texture;
mod tilerasterizer;

// TODO: Remove dead code in the GLTF scene boilerplate module. This is just to stop rustc from complaining.
#[allow(dead_code)]
mod scene;

use rendercamera::RenderCamera;
use renderer::{RenderBuffer, Renderer};
use scene::Scene;
use texture::TextureCache;

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const CAMERA_SPEED: f32 = 0.15; // Speed relative to scene size per second
const CAMERA_SPEED_FAST: f32 = 0.6; // Speed when shift is pressed

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

fn load_scene(gltf_path: &Path, render_state_loading: &Arc<Mutex<LoadingState>>) {
    // Load the GLTF file
    let (document, buffers) = match load_gltf(&gltf_path) {
        Ok(result) => result,
        Err(e) => {
            // Something failed during loading
            if let Ok(mut state) = render_state_loading.lock() {
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
            if let Ok(mut state) = render_state_loading.lock() {
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
    let scene = match Scene::from_gltf(&document, &gltf_scene, &buffers, &mut texture_cache) {
        Ok(scene) => scene,
        Err(e) => {
            if let Ok(mut state) = render_state_loading.lock() {
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
                if let Ok(mut state) = render_state_loading.lock() {
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

    // Create and store the render state
    let render_state = RenderState { scene, camera };
    if let Ok(mut state) = render_state_loading.lock() {
        *state = LoadingState::Loaded(render_state);
    }
}

fn main() {
    // Get the GLTF file path from command line arguments or use default
    let args: Vec<String> = env::args().collect();
    let gltf_path = if args.len() == 2 {
        Path::new(&args[1])
    } else {
        println!("Attempting to load glTF-Sample-Assets model: FlightHelmet");
        Path::new("glTF-Sample-Assets/Models/FlightHelmet/glTF/FlightHelmet.gltf")
    };

    // Extract filename
    let filename = gltf_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");

    // Save the start time
    let start_time = Instant::now();

    // Create the renderer
    let mut renderer = Renderer::new(WIDTH as i32, HEIGHT as i32);

    // Create the window and buffer
    let mut buffer = RenderBuffer::new(WIDTH, HEIGHT);
    let mut window = Window::new(
        &format!("GLTF Viewer - Loading: {}", filename),
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    window.set_target_fps(60);

    // Create shared state for background loading
    let loading_state = Arc::new(Mutex::new(LoadingState::Loading));
    let loading_state_clone = loading_state.clone();

    // Start background thread for scene loading
    let gltf_path = gltf_path.to_path_buf(); // Clone the path for the thread
    thread::spawn(move || {
        load_scene(&gltf_path, &loading_state_clone);
    });

    // Mouse state for rotation
    let mut is_dragging = false;
    let mut last_mouse_pos = (0.0, 0.0);

    // FPS measurement variables
    let mut frame_count = 0;
    let mut last_fps_update = Instant::now();
    let mut last_frame_time = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Compute delta time
        let current_time = Instant::now();
        let delta_time = current_time.duration_since(last_frame_time).as_secs_f32();
        last_frame_time = current_time;

        // Try to get the render state
        let mut render_state_guard = loading_state.lock().unwrap();

        match &mut *render_state_guard {
            LoadingState::Loaded(render_state) => {
                // Scene is loaded, handle input and render
                let camera = &mut render_state.camera;

                // Handle keyboard input for camera movement
                let mut move_dir = Vec3::ZERO;

                if window.is_key_down(Key::W) {
                    move_dir.z -= 1.0; // Forward (negative Z in camera space)
                }
                if window.is_key_down(Key::S) {
                    move_dir.z += 1.0; // Backward (positive Z in camera space)
                }
                if window.is_key_down(Key::A) {
                    move_dir.x -= 1.0; // Left (negative X in camera space)
                }
                if window.is_key_down(Key::D) {
                    move_dir.x += 1.0; // Right (positive X in camera space)
                }
                if window.is_key_down(Key::E) {
                    move_dir.y += 1.0; // Up (positive Y in camera space)
                }
                if window.is_key_down(Key::Q) {
                    move_dir.y -= 1.0; // Down (negative Y in camera space)
                }

                let scene_size = render_state.scene.bounds.diagonal;

                // Normalize movement direction if moving diagonally
                if move_dir.length_squared() > 0.0 {
                    move_dir = move_dir.normalize();
                    // Use faster speed when shift is pressed
                    let speed = if window.is_key_down(Key::LeftShift)
                        || window.is_key_down(Key::RightShift)
                    {
                        CAMERA_SPEED_FAST * scene_size
                    } else {
                        CAMERA_SPEED * scene_size
                    };
                    camera.move_relative(move_dir, speed * delta_time);
                }

                // Handle mouse input for camera rotation
                if window.get_mouse_down(MouseButton::Left) {
                    if let Some((x, y)) = window.get_mouse_pos(minifb::MouseMode::Discard) {
                        if !is_dragging {
                            is_dragging = true;
                        } else {
                            let dx = x - last_mouse_pos.0;
                            let dy = y - last_mouse_pos.1;
                            camera.rotate_mouse(dx, dy);
                        }
                        last_mouse_pos = (x, y);
                    }
                } else {
                    is_dragging = false;
                }

                // Update camera matrices
                camera.update_matrices();

                // Render the scene
                renderer.render_scene(&render_state.scene, camera, &mut buffer);
            }
            LoadingState::Loading => {
                // Scene is still loading, show loading screen
                buffer.clear();

                // Quaint little loading indicator
                let center_x = WIDTH as i32 / 2;
                let center_y = HEIGHT as i32 / 2;
                let radius = 50;
                let time = current_time.duration_since(start_time).as_secs_f64();
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
                    let intensity_int = (intensity * 255.0) as u32;
                    let color = intensity_int << 24
                        | intensity_int << 16
                        | intensity_int << 8
                        | intensity_int;
                    buffer.set_pixel(px, py, color);
                }
            }
            LoadingState::Error(error_msg) => {
                // Scene failed to load, exit
                eprintln!("Scene loading failed: {}", error_msg);
                break;
            }
        }

        // Drop the guard before updating the window
        drop(render_state_guard);

        // Update the window
        window
            .update_with_buffer(buffer.pixels(), WIDTH, HEIGHT)
            .unwrap();

        // Update FPS counter
        frame_count += 1;
        let elapsed = current_time.duration_since(last_fps_update);
        if elapsed.as_secs_f64() >= 1.0 {
            let fps = frame_count as f64 / elapsed.as_secs_f64();
            window.set_title(&format!("GLTF Viewer - {} - {:.1} FPS", filename, fps,));
            frame_count = 0;
            last_fps_update = current_time;
        }
    }
}
