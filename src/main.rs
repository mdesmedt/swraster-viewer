use glam::{Vec3, Vec3A};
use minifb::{Key, MouseButton, Window, WindowOptions};
use std::env;
use std::path::Path;
use std::time::Instant;

mod rendercamera;
mod renderer;

// TODO: Remove dead code in the GLTF scene boilerplate module. This is just to stop rustc from complaining.
#[allow(dead_code)]
mod scene;

use rendercamera::RenderCamera;
use renderer::{RenderBuffer, Renderer};
use scene::{compute_scene_bounds, Scene};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const CAMERA_SPEED: f32 = 2.0; // Units per second
const CAMERA_SPEED_FAST: f32 = 20.0; // Units per second when shift is pressed

fn main() {
    // Get the GLTF file path from command line arguments or use default
    let args: Vec<String> = env::args().collect();
    let gltf_path = if args.len() == 2 {
        Path::new(&args[1])
    } else {
        println!("Attempting to load glTF-Sample-Assets model: FlightHelmet");
        Path::new("glTF-Sample-Assets/Models/FlightHelmet/glTF/FlightHelmet.gltf")
    };

    // Load the GLTF file
    let (document, buffers, _) = gltf::import(gltf_path).expect("Failed to load GLTF file");

    // Get the default scene or first scene
    let gltf_scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .expect("No scenes found in GLTF file");

    let scene = Scene::from_gltf(&document, &gltf_scene, &buffers)
        .expect("Failed to create scene from GLTF");

    // Print scene statistics
    let total_triangles: usize = scene
        .meshes
        .iter()
        .map(|mesh| {
            mesh.primitives
                .iter()
                .map(|primitive| primitive.indices.len() / 3)
                .sum::<usize>()
        })
        .sum();
    println!("Scene Statistics:");
    println!("  Nodes: {}", scene.nodes.len());
    println!("  Meshes: {}", scene.meshes.len());
    println!("  Total Triangles: {}", total_triangles);

    // Create the renderer
    let mut renderer = Renderer::new(WIDTH as i32, HEIGHT as i32);

    // Create the window and buffer
    let mut buffer = RenderBuffer::new(WIDTH, HEIGHT);
    let mut window = Window::new(
        "GLTF Viewer - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    window.set_target_fps(60);

    // Compute scene bounds and set up camera to look at center
    let bounds = compute_scene_bounds(&scene);
    let scene_center = (bounds.min + bounds.max) * 0.5;
    let scene_size = (bounds.max - bounds.min).length();

    // Create a default camera if none exists in the file
    let mut camera = if let Some(camera) = document.cameras().next() {
        RenderCamera::from_gltf(&camera, WIDTH as f32, HEIGHT as f32)
            .expect("Failed to create camera from GLTF")
    } else {
        // Position camera at a distance proportional to scene size
        let camera_distance = scene_size * 2.0;
        let camera_pos = scene_center + Vec3A::new(0.0, 0.0, camera_distance);

        RenderCamera::new(
            Vec3::from(camera_pos),
            Vec3::from(scene_center),
            std::f32::consts::PI / 4.0,
            WIDTH as f32,
            HEIGHT as f32,
        )
    };

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

        // Normalize movement direction if moving diagonally
        if move_dir.length_squared() > 0.0 {
            move_dir = move_dir.normalize();
            // Use faster speed when shift is pressed
            let speed = if window.is_key_down(Key::LeftShift) || window.is_key_down(Key::RightShift)
            {
                CAMERA_SPEED_FAST
            } else {
                CAMERA_SPEED
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

        // Render the scene
        renderer.render_scene(&scene, &camera, &mut buffer);

        // Update the window
        window
            .update_with_buffer(buffer.pixels(), WIDTH, HEIGHT)
            .unwrap();

        // Update FPS counter
        frame_count += 1;
        let elapsed = current_time.duration_since(last_fps_update);
        if elapsed.as_secs_f64() >= 1.0 {
            let fps = frame_count as f64 / elapsed.as_secs_f64();
            window.set_title(&format!(
                "GLTF Viewer - {:.1} FPS ({:.1}ms) - WSAD to move, Mouse to look - ESC to exit",
                fps,
                1000.0 / fps
            ));
            frame_count = 0;
            last_fps_update = current_time;
        }
    }
}
