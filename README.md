# swraster-viewer
Simple software rasterizer which views GLTF files, written in Rust.

![Screenshot](https://github.com/user-attachments/assets/7ab18e2f-8c85-4ef7-a2f4-2e061f5f7840)

The project has been tested with the assets from the glTF-Sample-Assets repository: https://github.com/KhronosGroup/glTF-Sample-Assets

# Quick Start
Clone the glTF-Sample-Assets repository into the project folder and run the project:
```
git clone https://github.com/KhronosGroup/glTF-Sample-Assets.git
cargo run --release
```

To view another glTF view just specify it on the command line:
```
cargo run --release glTF-Sample-Assets/Models/Sponza/glTF/Sponza.gltf
```

# About
With this project I tried to make a relatively simple, relatively performant software rasterizer. Just for fun.

The general architecture of the rasterizer is:
- Load a glTF scene with the `gltf` crate.
- Iterate over nodes/meshes/primitives in parallel with `rayon`.
- Project each triangle into clip space, clip against the frustum, and bin the resulting triangles into screen space tiles, using a custom queuing algorithming.
- Every screen space tile runs in a thread and uses SIMD (`glam`) to edge test, shade and depth test 4 pixels at a time.
- After all triangles have been processed the tiles are copied into a linear color buffer which is viewed with `minifb`.

Thanks to Fabian “ryg” Giesen for his helpful software rasterizer tutorial series: https://fgiesen.wordpress.com/2013/02/17/optimizing-sw-occlusion-culling-index/
