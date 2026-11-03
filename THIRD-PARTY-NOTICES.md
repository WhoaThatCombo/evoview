# Third-party notices

evoview links 219 Rust crates. Every one is used unmodified, under the licence
shown. Nothing here is redistributed in source form; these are the obligations that
travel with the compiled binary.

The Apache-2.0 entries are why this file exists: Apache-2.0 section 4(d) requires the
attribution notice to be carried with any distribution, source or binary.

| licence | crates |
|---|---|
| MIT OR Apache-2.0 | `ahash`, `android-activity`, `android_system_properties`, `arrayvec`, `as-raw-xcb-connection`, `ash`, `bitflags`, `bumpalo`, `cc`, `cfg-if`, `core-foundation`, `core-foundation-sys`, `core-graphics`, `core-graphics-types`, `crc32fast`, `crossbeam-utils`, `d3d12`, `document-features`, `errno`, `fdeflate`, `find-msvc-tools`, `flate2`, `futures-core`, `futures-task`, `futures-util`, `getrandom`, `glam`, `gpu-alloc`, `gpu-alloc-types`, `gpu-allocator`, `gpu-descriptor`, `gpu-descriptor-types`, `hashbrown`, `hermit-abi`, `itoa`, `jni-sys`, `jni-sys-macros`, `jobserver`, `js-sys`, `libc`, `litrs`, `lock_api`, `log`, `memmap2`, `metal`, `naga`, `ndk`, `ndk-context`, `ndk-sys`, `once_cell`, `parking_lot`, `parking_lot_core`, `paste`, `percent-encoding`, `pkg-config`, `png`, `presser`, `proc-macro-crate`, `proc-macro2`, `profiling`, `quote`, `range-alloc`, `renderdoc-sys`, `rustversion`, `scopeguard`, `serde`, `serde_core`, `serde_derive`, `serde_json`, `shlex`, `smallvec`, `smol_str`, `static_assertions`, `syn`, `thiserror`, `thiserror-impl`, `toml_datetime`, `toml_edit`, `toml_parser`, `ttf-parser`, `unicode-segmentation`, `unicode-width`, `unicode-xid`, `wasm-bindgen`, `wasm-bindgen-futures`, `wasm-bindgen-macro`, `wasm-bindgen-macro-support`, `wasm-bindgen-shared`, `web-sys`, `web-time`, `wgpu`, `wgpu-core`, `wgpu-hal`, `wgpu-types`, `widestring`, `windows`, `windows-core`, `windows-link`, `windows-sys`, `windows-targets`, `windows_aarch64_gnullvm`, `windows_aarch64_msvc`, `windows_i686_gnu`, `windows_i686_gnullvm`, `windows_i686_msvc`, `windows_x86_64_gnu`, `windows_x86_64_gnullvm`, `windows_x86_64_msvc`, `x11rb`, `x11rb-protocol` |
| MIT | `android-properties`, `block`, `block-sys`, `block2`, `bytes`, `calloop`, `calloop-wayland-source`, `cfg_aliases`, `com`, `com_macros`, `com_macros_support`, `combine`, `dispatch`, `dlib`, `hassle-rs`, `icrate`, `libredox`, `malloc_buf`, `objc`, `objc-sys`, `objc2`, `objc2-encode`, `orbclient`, `quick-xml`, `redox_syscall`, `sctk-adwaita`, `simd-adler32`, `slab`, `smithay-client-toolkit`, `strict-num`, `wayland-backend`, `wayland-client`, `wayland-csd-frame`, `wayland-cursor`, `wayland-protocols`, `wayland-protocols-plasma`, `wayland-protocols-wlr`, `wayland-scanner`, `wayland-sys`, `winnow`, `x11-dl`, `xcursor`, `xkbcommon-dl`, `xml-rs`, `zmij` |
| MIT/Apache-2.0 | `bit-set`, `bit-vec`, `bitflags`, `downcast-rs`, `foreign-types`, `foreign-types-macros`, `foreign-types-shared`, `jni`, `khronos-egl`, `plain`, `scoped-tls`, `version_check`, `winapi`, `winapi-i686-pc-windows-gnu`, `winapi-x86_64-pc-windows-gnu` |
| Apache-2.0 | `ab_glyph`, `ab_glyph_rasterizer`, `codespan-reporting`, `gethostname`, `gl_generator`, `glutin_wgl_sys`, `khronos_api`, `owned_ttf_parser`, `spirv`, `winit` |
| Apache-2.0 OR MIT | `atomic-waker`, `concurrent-queue`, `equivalent`, `indexmap`, `pin-project-lite`, `polling` |
| Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | `linux-raw-sys`, `rustix`, `wasip2`, `wit-bindgen` |
| MIT OR Apache-2.0 OR Zlib | `cursor-icon`, `glow`, `raw-window-handle`, `xkeysym` |
| Apache-2.0/MIT | `cesu8`, `pollster`, `rustc-hash` |
| Unlicense OR MIT | `memchr`, `termcolor`, `winapi-util` |
| BSD-2-Clause OR Apache-2.0 OR MIT | `zerocopy`, `zerocopy-derive` |
| BSD-3-Clause | `tiny-skia`, `tiny-skia-path` |
| BSD-3-Clause OR MIT OR Apache-2.0 | `num_enum`, `num_enum_derive` |
| Unlicense/MIT | `same-file`, `walkdir` |
| Zlib | `foldhash`, `slotmap` |
| Zlib OR Apache-2.0 OR MIT | `bytemuck`, `bytemuck_derive` |
| (MIT OR Apache-2.0) AND Unicode-3.0 | `unicode-ident` |
| 0BSD OR MIT OR Apache-2.0 | `adler2` |
| BSD-2-Clause | `arrayref` |
| CC0-1.0 | `hexf-parse` |
| ISC | `libloading` |
| MIT OR Apache-2.0 OR LGPL-2.1-or-later | `r-efi` |
| MIT OR Zlib OR Apache-2.0 | `miniz_oxide` |

Regenerate with:

```
cargo metadata --format-version 1 --all-features
```

## Techniques

- Screen-space reflections (`src/ssr.rs`, `src/ssr.wgsl`) follow AMD FidelityFX
  Stochastic Screen Space Reflections (FidelityFX SDK, MIT licence,
  <https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK>): one GGX-VNDF ray per
  2x2 quad, a hierarchical min-depth march and its hit validation, written for wgpu.
- Colour grading (`src/grade.rs`) follows Hajime Uchimura's GT tone curve (CEDEC 2017)
  and Timothy Lottes' crosstalk "path to white" (GDC 2016).

## Ported source

- Open Dynamics Engine 0.15.2 (`src/ode/`): a Rust translation of ODE's maths,
  bodies, joints, QuickStep stepper, spaces and colliders, and of the parts of
  OPCODE 1.3 (Copyright (C) 2001 Pierre Terdiman; distributed with ODE under ODE's
  licence terms) that ODE's trimesh colliders use, used under ODE's BSD-style licence (ODE is
  dual-licensed LGPL / BSD; the BSD terms are the ones taken here):

```text
This is the BSD-style license for the Open Dynamics Engine
----------------------------------------------------------

Open Dynamics Engine
Copyright (c) 2001-2007, Russell L. Smith.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

Neither the names of ODE's copyright owner nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Runtime libraries (not shipped)

- AMD FSR (`amd_fidelityfx_loader_vk.dll`, `amd_fidelityfx_upscaler_vk.dll`) and NVIDIA
  DLSS (`nvngx_dlss.dll`) are the vendors' own signed binaries, which the player adds;
  DLSS runs through the NGX runtime in NVIDIA's driver. The game's FMOD libraries are
  loaded from the player's install. None of these are part of evoview.
