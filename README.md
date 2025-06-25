# swraster-viewer
Simple software rasterizer which views GLTF files, written in Rust.

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
- Project each triangle into clip space, clip against the frustum, and bin the resulting triangles into screen space 128x128 pixel tiles, using `crossbeam` channels.
- Every screen space tile has an independent rastizer running in a thread, using SIMD to edge test, shade and depth test 4 pixels at a time with `glam`.
- After all triangles have been processed the tiles are copied into a linear color buffer which is viewed with `minifb`.

Thanks to Fabian “ryg” Giesen for his helpful software rasterizer tutorial series: https://fgiesen.wordpress.com/2013/02/17/optimizing-sw-occlusion-culling-index/
