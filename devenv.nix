{ pkgs, lib, ... }:

let
  libs = with pkgs; [
    pkg-config
    mold
    wayland
    glfw-wayland
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
{
  languages.rust = {
    enable = true;
    components = [
      "rustc"
      "cargo"
      "clippy"
      "rustfmt"
      "rust-analyzer"
    ];
  };

  packages = libs;

  env.RUSTFLAGS = lib.mkForce "-C link-args=-Wl,-fuse-ld=mold,-rpath,${lib.makeLibraryPath libs}";

  git-hooks.hooks = {
    rustfmt.enable = true;
    clippy.enable = false;
  };
}
