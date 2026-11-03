//! Headless render to a PNG.
//!
//! A framerate counter says the pipeline runs; it says nothing about whether
//! the car looks like a car. This renders one frame with no window so the
//! result can be inspected directly - useful for checking geometry, winding
//! and shading without a human at the keyboard, and for regression shots.
//!
//! `Still` keeps the device and scene so one load can render many frames
//! (different effect states, cameras): what `--check` (check.rs) needs.

use crate::asset::Car;
use crate::camera::Orbit;
use crate::fxstate::FxState;
use crate::material::{Blend, Materials};
use crate::render::{build_scene, ground_quad, uniforms_mode, uniforms_shadowed,
                    light_matrix, robust_bounds, GpuMesh, Pipelines, Uniforms};
use std::path::Path;
use wgpu::util::DeviceExt;

pub struct Still {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub car: Car,
    pub materials: Materials,
    pub meshes: Vec<GpuMesh>,
    pipes: Pipelines,
    track: Option<crate::track::TrackRender>,
    pub fxg: crate::fxgpu::FxGpu,
    msaa: wgpu::TextureView,
    depth: wgpu::TextureView,
    resolved: wgpu::Texture,
    rv: wgpu::TextureView,
    readback: wgpu::Buffer,
    padded: u32,
    pub width: u32,
    pub height: u32,
    pub outdoor: bool,
    shadows: bool,
    /// per-layer render views of the shadow map (0 near / hero, 1 far)
    shadow_layers: Vec<wgpu::TextureView>,
    /// whether the far cascade is drawn (tracks)
    far: bool,
    uniform_buf: wgpu::Buffer,
    bone_buf: wgpu::Buffer,
    /// Some(dt): consecutive frames of one sequence - particles step by dt
    /// instead of being re-simulated for a still.
    pub seq_dt: Option<f32>,
    pub cam_u: Uniforms,
    pub vp: glam::Mat4,
    pub eye: glam::Vec3,
    pub lo: glam::Vec3,
    pub hi: glam::Vec3,
}

pub async fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("no Vulkan adapter");
    println!("GPU: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::TEXTURE_COMPRESSION_BC,
                required_limits: crate::render::limits(&adapter),
                memory_hints: Default::default(),
            },
            None,
        )
        .await
        .expect("device (does this GPU support BC textures?)")
}

impl Still {
    #[allow(clippy::too_many_arguments)]
    /// `view` overrides the orbit camera entirely with a (view_proj, eye) pair -
    /// what the replay follow-cameras produce. They cannot be expressed as
    /// yaw/pitch/zoom because zoom is relative to a framing distance only known
    /// once the scene is uploaded.
    pub async fn new(car: Car, width: u32, height: u32, yaw: f32, pitch: f32, zoom: f32,
                     at: Option<glam::Vec3>, pose: Option<Vec<[f32; 16]>>, outdoor: bool,
                     view: Option<(glam::Mat4, glam::Vec3)>) -> Still {
        let (device, queue) = device().await;
        let materials = Materials::build(&device, &queue, &car);
        let (mut meshes, mut lo, mut hi, tris) = if crate::grid::wanted() {
            crate::grid::build(&device, &car, &materials)
        } else {
            build_scene(&device, &car, Some(&materials))
        };
        // ⚠ Not in grid mode: this replaces the bounds with the TRACK's real
        // extent, which is kilometres wide, and the sheet ends up a speck at the
        // origin - the frame comes back as nothing but sky.
        if outdoor && !crate::grid::wanted() {
            if let Some((a, b)) = robust_bounds(&car) {
                lo = a;
                hi = b;
            }
        }
        println!("{} mesh(es), {} triangles", meshes.len(), tris);
        let mut cam = Orbit::framing(lo, hi);
        cam.yaw = yaw;
        cam.pitch = pitch;
        cam.distance /= zoom.max(0.01);
        if let Some(t) = at {
            cam.target = t;
        }
        println!(
            "camera: {:.1} m from ({:.1}, {:.1}, {:.1})",
            cam.distance, cam.target.x, cam.target.y, cam.target.z);

        // Same rule as the interactive view: no floor between camera and car.
        // Tracks are outdoors — a studio sweep under a circuit is nonsense.
        let eye_y = view.map_or_else(|| cam.eye().y, |v| v.1.y);
        if !outdoor && !crate::grid::wanted() && eye_y > lo.y + 1e-3
            && std::env::var("EVOVIEW_NO_GROUND").is_err() {
            let mut g = ground_quad(&device, lo, hi);
            g.draws = vec![(0, 6, materials.ground)];
            meshes.push(g);
        }

        // sRGB so the shader's gamma step lands the same as on screen
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let make = |sample_count, fmt, usage, label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage,
                view_formats: &[],
            })
        };
        let msaa = make(4, format, wgpu::TextureUsages::RENDER_ATTACHMENT, "msaa");
        let depth = make(4, wgpu::TextureFormat::Depth32Float,
                         wgpu::TextureUsages::RENDER_ATTACHMENT, "depth");
        let resolved = make(1, format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, "resolved");

        // Shadows only outdoors: the studio rig is area lights, which this map
        // cannot represent, and the car already has its contact shadow.
        let shadows = outdoor && std::env::var("EVOVIEW_NO_SHADOW").is_err();
        let vp = view.map_or_else(|| cam.view_proj(width as f32 / height as f32), |v| v.0);
        let eye = view.map_or_else(|| cam.eye(), |v| v.1);
        // Aim the map where the camera is looking, not at the centre of a
        // kilometres-wide circuit.
        let focus = if outdoor && view.is_some() { crate::render::view_focus(vp, eye) } else { cam.target };
        // Cars get the game's hero shadow instead: a map fitted to the car,
        // rendered once (the pose is fixed for a still).
        let hero = !outdoor && std::env::var("EVOVIEW_NO_SHADOW").is_err()
            && std::env::var("EVOVIEW_NO_HERO").is_err();
        let cam_u = if shadows {
            uniforms_shadowed(vp, eye, lo, hi, outdoor, true, focus)
        } else if hero {
            crate::render::uniforms_car_shadowed(uniforms_mode(vp, eye, lo, hi, outdoor), lo, hi)
        } else {
            uniforms_mode(vp, eye, lo, hi, outdoor)
        };
        let shadows = shadows || hero;
        let clouds = crate::clouds::CloudGpu::new(&device, &queue, &car);
        let mut cam_u = cam_u;
        if outdoor {
            cam_u.clouds = clouds.block();
        }
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera"),
            contents: bytemuck::bytes_of(&cam_u),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // The shadow pass renders with the LIGHT's matrix in `view_proj`.
        let mut light_u = cam_u;
        light_u.view_proj = if hero {
            crate::render::car_light_matrix(lo, hi).0.to_cols_array_2d()
        } else {
            light_matrix(focus).to_cols_array_2d()
        };
        let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("light camera"),
            contents: bytemuck::bytes_of(&light_u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let (shadow_view, shadow_samp, shadow_layers) = crate::render::shadow_target_dim(
            &device, if hero { crate::render::hero_dim() } else { crate::render::SHADOW_DIM });
        // Rest pose unless the caller posed the skeleton, which is what makes an
        // animation checkable from a screenshot instead of only by eye.
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut bones: Vec<[f32; 16]> = pose.unwrap_or_default();
        bones.resize(crate::render::MAX_BONES, identity);
        let bone_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bones"),
            contents: bytemuck::cast_slice(&bones),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pipes = crate::render::build_pipeline(
            &device, format, &uniform_buf, 4, &materials.layout, &bone_buf,
            &materials.env_view, &materials.env_samp, &materials.brdf_view, &materials.lut_view,
            &shadow_view, &shadow_samp, &light_buf, &clouds, None, [None; 3], None, &crate::rain::streak_view(&device, &queue, None));
        let far = shadows && !hero && outdoor && crate::render::cascades();
        if far {
            let mut lf = cam_u;
            lf.view_proj = crate::render::light_matrix_far(focus).to_cols_array_2d();
            queue.write_buffer(&pipes.light_buf_far, 0, bytemuck::bytes_of(&lf));
        }
        let track = crate::track::TrackRender::new(
            &device, &queue, &car, format, 4, &shadow_view, &shadow_samp, &materials.lut_view, &materials,
            &clouds);
        if let Some(t) = track.as_ref() {
            t.update(&queue, vp, eye, true, &cam_u);
        }
        let fxg = crate::fxgpu::FxGpu::new(&device, &queue, &car, &materials, format, 4, &uniform_buf);

        // Readback rows must be padded to 256 bytes - a wgpu requirement that
        // silently corrupts the image if ignored.
        let unpadded = width * 4;
        let pad = (256 - unpadded % 256) % 256;
        let padded = unpadded + pad;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mv = msaa.create_view(&Default::default());
        let rv = resolved.create_view(&Default::default());
        let dv = depth.create_view(&Default::default());
        Still {
            device, queue, car, materials, meshes, pipes, track, fxg, msaa: mv, depth: dv,
            resolved, rv, readback, padded, width, height, outdoor, shadows, shadow_layers, far,
            uniform_buf, bone_buf, seq_dt: None, cam_u, vp, eye, lo, hi,
        }
    }

    /// Pose the skeleton (skin matrices, as drive / demo / check produce
    /// them) for the next `render`. None = rest pose.
    pub fn set_pose(&self, pose: Option<&[[f32; 16]]>) {
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut b: Vec<[f32; 16]> = pose.map(|p| p.to_vec()).unwrap_or_default();
        b.resize(crate::render::MAX_BONES, identity);
        self.queue.write_buffer(&self.bone_buf, 0, bytemuck::cast_slice(&b));
    }

    /// Render one frame with effect state `st`; RGBA8 rows, top first.
    pub fn render(&mut self, st: &FxState) -> Vec<u8> {
        let mut u = self.cam_u;
        u.view_proj = self.vp.to_cols_array_2d();
        u.eye = [self.eye.x, self.eye.y, self.eye.z, 1.0];
        u.fx = crate::lights::fx_block(self.car.fx.as_deref(), st, self.outdoor);
        if std::env::var("EVOVIEW_FX_LOG").is_ok() {
            if let Some(l) = self.car.fx.as_ref().and_then(|f| f.led_sys.as_ref()) {
                let mut k: Vec<_> = l.groups.keys().collect();
                k.sort();
                for f in k {
                    for (i, g) in l.groups[f].iter().enumerate() {
                        for x in g {
                            println!("led group {f}[{i}]: led {} thr {} on {:?} v {}", x.led, x.threshold, x.color_on, x.value_on);
                        }
                    }
                }
            }
            for (i, v) in u.fx.iter().enumerate() {
                if v.iter().any(|x| *x != 0.0) {
                    println!("fx[{i}] = {v:?}");
                }
            }
        }
        self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
        let (device, queue) = (&self.device, &self.queue);
        let materials = &self.materials;
        let meshes = &self.meshes;
        let pipes = &self.pipes;
        let track = &self.track;
        let mut enc = device.create_command_encoder(&Default::default());
        let eye3 = [self.eye.x, self.eye.y, self.eye.z];
        match self.seq_dt {
            Some(dt) => self.fxg.prepare_frame(device, queue, &mut enc, self.car.fx.as_deref(), st, dt, eye3),
            None => self.fxg.prepare_still(device, queue, &mut enc, &self.car, st, eye3),
        }
        let (pipeline, blended, bind_group, bone_bg) =
            (&pipes.pipeline, &pipes.blended, &pipes.bind_group, &pipes.bone_bg);
        let layers = if !self.shadows { 0 } else if self.far { 2 } else { 1 };
        for layer in 0..layers {
            // Depth from the sun. Opaque and cutout only - glass and foliage cards
            // that are alpha-BLENDED must not stamp a solid silhouette on the
            // ground, or every window throws a black rectangle.
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(if layer == 0 { "shadow" } else { "shadow (far)" }),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_layers[layer],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&pipes.shadow);
            pass.set_bind_group(0, if layer == 0 { &pipes.light_bind_group } else { &pipes.light_bind_group_far }, &[]);
            pass.set_bind_group(2, bone_bg, &[]);
            for m in meshes {
                pass.set_vertex_buffer(0, m.vertices.slice(..));
                pass.set_vertex_buffer(1, m.instances.slice(..));
                pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                for &(start, count, mi) in &m.draws {
                    if materials.blend.get(mi).copied().unwrap_or(Blend::Opaque).transparent()
                        || materials.casts_no_shadow(mi)
                        // track cutouts: alpha-tested below
                        || track.as_ref().map_or(false, |t| t.casts_cutout(mi)) {
                        continue;
                    }
                    pass.set_bind_group(1, &materials.groups[mi], &[]);
                    pass.draw_indexed(start..start + count, 0, 0..m.instance_count);
                }
            }
            if let Some(t) = track.as_ref() {
                t.draw_shadow_cutouts(&mut pass, layer, meshes, &|gi| {
                    (&meshes[gi].instances, meshes[gi].instance_count)
                });
            }
        }
        {
            let clear = if self.outdoor {
                wgpu::Color { r: 0.42, g: 0.58, b: 0.82, a: 1.0 }
            } else {
                // EVOVIEW_SHOT_ALPHA: transparent background, so the PNG's
                // alpha is the car's silhouette (lookcmp camera fitting)
                let c = crate::grade::studio_clear_scaled(u.fx[0][0]);
                wgpu::Color { r: c[0], g: c[1], b: c[2],
                    a: if std::env::var("EVOVIEW_SHOT_ALPHA").is_ok() { 0.0 } else { 1.0 } }
            };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shot"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa,
                    resolve_target: Some(&self.rv),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_bind_group(2, bone_bg, &[]);
            // Opaque first, then everything transparent. Within the transparent
            // pass depth writes are off, so glass never hides the lamp internals
            // or the interior behind it.
            for phase in [Blend::Opaque, Blend::Cutout, Blend::Alpha, Blend::Glass].iter() {
                if let Some(t) = track.as_ref() {
                    if *phase == Blend::Alpha {
                        t.draw_sky(&mut pass);
                    }
                    t.draw_phase(&mut pass, meshes, *phase, &|_, n| n);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.set_bind_group(2, bone_bg, &[]);
                }
                pass.set_pipeline(if *phase == Blend::Alpha { blended } else { pipeline });
                for m in meshes {
                    if track.as_ref().map_or(false, |t| t.owns_mesh(m)) {
                        continue;
                    }
                    pass.set_vertex_buffer(0, m.vertices.slice(..));
                    pass.set_vertex_buffer(1, m.instances.slice(..));
                    pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                    for &(start, count, mi) in &m.draws {
                        if m.extra.is_some() && track.as_ref().map_or(false, |t| t.owns(mi)) {
                            continue;
                        }
                        if materials.blend.get(mi).copied().unwrap_or(Blend::Opaque) != *phase {
                            continue;
                        }
                        pass.set_bind_group(1, &materials.groups[mi], &[]);
                        if *phase == Blend::Glass {
                            pass.set_pipeline(&pipes.glass_tint);
                            pass.draw_indexed(start..start + count, 0, 0..m.instance_count);
                            pass.set_pipeline(&pipes.glass_add);
                        }
                        pass.draw_indexed(start..start + count, 0, 0..m.instance_count);
                    }
                }
            }
            self.fxg.draw_particles(&mut pass);
        }
        self.fxg.bloom(device, queue, &mut enc, st.bloom, meshes, materials, pipes, &self.depth, &self.rv,
                       self.width, self.height);
        let size = wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 };
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.resolved,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded),
                    rows_per_image: Some(self.height),
                },
            },
            size,
        );
        queue.submit([enc.finish()]);

        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let unpadded = self.width * 4;
        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((unpadded * self.height) as usize);
        for row in 0..self.height {
            let s = (row * self.padded) as usize;
            pixels.extend_from_slice(&data[s..s + unpadded as usize]);
        }
        drop(data);
        self.readback.unmap();
        pixels
    }
}

pub fn write_png(out: &Path, width: u32, height: u32, pixels: &[u8]) {
    // A bad output path is a caller mistake, not a bug worth a panic and a
    // backtrace - say which path failed and why.
    if let Some(dir) = out.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let file = match std::fs::File::create(out) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot write {}: {e}", out.display());
            std::process::exit(2);
        }
    };
    let mut e = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    e.set_color(png::ColorType::Rgba);
    e.set_depth(png::BitDepth::Eight);
    e.write_header().unwrap().write_image_data(pixels).unwrap();
    println!("wrote {}", out.display());
}

/// Several posed frames of one scene, uploaded once: a demo frame sequence or
/// a contact sheet of key poses costs one load instead of one per image.
/// Each frame may carry its own effect state (None = the env vars').
#[allow(clippy::too_many_arguments)]
pub async fn shot_frames(car: Car, frames: Vec<(Option<Vec<[f32; 16]>>, std::path::PathBuf)>,
                         width: u32, height: u32, yaw: f32,
                         pitch: f32, zoom: f32, at: Option<glam::Vec3>, outdoor: bool,
                         view: Option<(glam::Mat4, glam::Vec3)>) {
    let fx: Vec<Option<FxState>> = vec![None; frames.len()];
    shot_frames_fx(car, frames, fx, None, width, height, yaw, pitch, zoom, at, outdoor, view).await
}

#[allow(clippy::too_many_arguments)]
pub async fn shot_frames_fx(car: Car, frames: Vec<(Option<Vec<[f32; 16]>>, std::path::PathBuf)>,
                            fx: Vec<Option<FxState>>,
                            seq_dt: Option<f32>,
                            width: u32, height: u32, yaw: f32,
                            pitch: f32, zoom: f32, at: Option<glam::Vec3>, outdoor: bool,
                            view: Option<(glam::Mat4, glam::Vec3)>) {
    let pose = frames.first().and_then(|f| f.0.clone());
    let mut still = Still::new(car, width, height, yaw, pitch, zoom, at, pose, outdoor, view).await;
    // a frame SEQUENCE keeps its particles alive from frame to frame
    still.seq_dt = seq_dt;
    let env = FxState::from_env();
    for (i, (p, out)) in frames.iter().enumerate() {
        if i > 0 {
            still.set_pose(p.as_deref());
        }
        let st = fx.get(i).copied().flatten().unwrap_or(env);
        let px = still.render(&st);
        write_png(out, width, height, &px);
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn shot(car: Car, out: &Path, width: u32, height: u32, yaw: f32,
                  pitch: f32, zoom: f32, at: Option<glam::Vec3>,
                  pose: Option<Vec<[f32; 16]>>, outdoor: bool,
                  view: Option<(glam::Mat4, glam::Vec3)>) {
    let mut still = Still::new(car, width, height, yaw, pitch, zoom, at, pose, outdoor, view).await;
    let st = FxState::from_env();
    let pixels = still.render(&st);
    // EVOVIEW_BENCH: time to first frame since process start. (The old N-frame
    // uncapped loop is retired - frame timing is `--bench` / tools/bench.py,
    // which honours EVOVIEW_MAX_FPS and runs the real culled renderer.)
    if std::env::var("EVOVIEW_BENCH").is_ok() {
        if let Some(t0) = crate::T0.get() {
            println!("BENCH first_frame_s {:.3}", t0.elapsed().as_secs_f64());
        }
    }
    // EVOVIEW_DASH_DUMP=<dir>: the dash screens as they were drawn
    if let Ok(dir) = std::env::var("EVOVIEW_DASH_DUMP") {
        for (k, (pi, t)) in still.materials.dash_targets.iter().enumerate() {
            let (w, h, px) = crate::fxgpu::read_rgba(&still.device, &still.queue, t);
            write_png(&Path::new(&dir).join(format!("dash_{k}_page{pi}.png")), w, h, &px);
        }
    }
    if still.fxg.particle_count() > 0 {
        println!("fx: {} particle(s) in the shot", still.fxg.particle_count());
    }
    write_png(out, width, height, &pixels);
}
