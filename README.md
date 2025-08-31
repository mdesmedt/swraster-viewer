# swraster-viewer
Software rasterizer which views GLTF files, written in Rust.

![Screenshot](https://github.com/user-attachments/assets/3ed5cf7a-52ef-4cdc-ac62-43f0a752952e)

The project has been tested with the assets from the glTF-Sample-Assets repository: https://github.com/KhronosGroup/glTF-Sample-Assets

# Quick Start
Clone the glTF-Sample-Assets repository into the project folder and run the project:
```
git clone https://github.com/KhronosGroup/glTF-Sample-Assets.git
cargo run --release glTF-Sample-Assets/Models/ChronographWatch/glTF/ChronographWatch.gltf
```

# About
With this project I tried to make a relatively simple, relatively performant software rasterizer. Just for fun.

The general architecture of the rasterizer is:
- Load a glTF scene with the `gltf` crate.
- Iterate over nodes/meshes/primitives in parallel with `rayon`.
- Project each triangle into clip space, clip against the frustum, and bin the resulting triangles into screen space tiles using a segmented queue.
- Every rasterizer tile runs in a thread and uses SIMD (`glam`) to rasterize a 4-pixel quad at a time, writing into a visibility buffer.
- Each tile shades its visibility buffer, with up to 4 active SIMD lanes per quad. Shading uses GLTF pbrMetallicRoughness inputs but is currently very approximated and hacked.
- After all triangles have been processed the tiles are copied into a linear color buffer which is rendered to a window with `pixels` and `winit`.

Thanks to Fabian “ryg” Giesen for his helpful software rasterizer tutorial series: https://fgiesen.wordpress.com/2013/02/17/optimizing-sw-occlusion-culling-index/
