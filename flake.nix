{
  description = "swraster-viewer - software rasterizer glTF viewer";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAll (
        pkgs: with pkgs; {
          default =
            let
              libs = [
                wayland
                glfw
                libxkbcommon
                libGL
                vulkan-headers
                vulkan-loader
                vulkan-tools
                vulkan-tools-lunarg
                vulkan-extension-layer
                vulkan-validation-layers
              ];
            in
            mkShell {
              packages = [
                cargo
                rustc
                rust-analyzer
                clippy
                rustfmt
                pkg-config
                mold
              ] ++ libs;

              RUSTFLAGS = "-C link-args=-Wl,-fuse-ld=mold,-rpath,${lib.makeLibraryPath libs}";
            };
        }
      );
    };
}
