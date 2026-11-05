//! Screen-space reflections on tracks (ssr.wgsl), after the game's own: it
//! ships AMD FidelityFX SSSR (the cbSSSR / cbClassifierReflection /
//! cbDenoiserReflections compute blobs, intersect lifted from 054a9160).
//!
//! evoview is forward-shaded, so the main pass writes two more targets for
//! it: `env` (the environment specular the material already added, in output
//! units, + the weight that reflection had) and `geo` (octahedral world
//! normal, perceptual roughness, a kind flag: 1/3 static, 2/3 the driven
//! car). Then, before TAA:
//!   hiz       min-depth pyramid from the depth, level 0 at half size
//!   trace     one GGX-VNDF ray per 2x2 quad, FFX's hierarchical march and
//!             hit validation; the hit's colour x confidence
//!   temporal  the reprojected history (virtual-point reprojection for the
//!             mirror image, the car box for the car), clipped to this
//!             frame's 3x3 neighbourhood
//!   composite adds conf x (weight x hit - env) onto the scene, so where the
//!             ray found nothing the material's sky / probe stays
//! Only rough < cutoff surfaces (FFX's roughness threshold, on GGX alpha)
//! trace; the composite fades out toward the cutoff.
//!
//! EVOVIEW_SSR=0 off. EVOVIEW_SSR_CUTOFF (alpha, 0.2), EVOVIEW_SSR_STEPS
//! (march iterations, 64), EVOVIEW_SSR_THICK (depth thickness per metre of
//! distance, 0.02), EVOVIEW_SSR_STRENGTH (1), EVOVIEW_SSR_HIST (new-frame
//! weight, 0.15), EVOVIEW_SSR_BLUR (prefilter radius cap, half-size texels, 4).
//! Needs the HDR scene, single-sampled, with TAA (driving).

use glam::{Mat4, Vec3};
use std::sync::atomic::{AtomicBool, Ordering};

pub const ENV_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const GEO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgb10a2Unorm;
const TRACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Hi-Z levels: level 0 is padded to a multiple of 2^(LEVELS-1)
const LEVELS: u32 = 8;

static ON: AtomicBool = AtomicBool::new(false);

/// EVOVIEW_SSR=0 turns it off.
pub fn wanted() -> bool {
    crate::envc::var("EVOVIEW_SSR").map_or(true, |v| v != "0")
}

/// Whether the main pass carries the two SSR targets (set once at startup,
/// before any main-pass pipeline is built).
pub fn set_active(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

pub fn active() -> bool {
    ON.load(Ordering::Relaxed)
}

/// The SSR targets every main-pass pipeline declares after its colour
/// target: written by the opaque materials (`write`), left alone by the rest
/// (sky, glass reflections, HUD). Empty while SSR is off.
pub fn extra_targets(write: bool) -> Vec<Option<wgpu::ColorTargetState>> {
    targets_with(if write { Some(None) } else { None })
}

/// What a see-through surface does to the reflection change behind it:
/// `Tint` the glass's dst x src (its fs_glass_tint_ssr), `Cover` an
/// alpha-blended surface's dst x (1 - alpha) (fs_main_cover). Without it
/// the road seen through the windscreen lost the whole sky reflection
/// while its colour had only kept the glass's share - black blotches.
#[derive(Clone, Copy, PartialEq)]
pub enum Behind {
    Tint,
    Cover,
}

pub fn behind_targets(b: Behind) -> Vec<Option<wgpu::ColorTargetState>> {
    let f = match b {
        Behind::Tint => wgpu::BlendFactor::Src,
        Behind::Cover => wgpu::BlendFactor::OneMinusSrcAlpha,
    };
    let c = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: f, operation: wgpu::BlendOperation::Add };
    targets_with(Some(Some(wgpu::BlendState { color: c, alpha: c })))
}

/// env target: None untouched, Some(blend) written with it; geo only when
/// the env is written unblended
fn targets_with(env: Option<Option<wgpu::BlendState>>) -> Vec<Option<wgpu::ColorTargetState>> {
    if !active() {
        return Vec::new();
    }
    let none = wgpu::ColorWrites::empty();
    let (env_blend, env_mask) = match env {
        Some(b) => (b, wgpu::ColorWrites::ALL),
        None => (None, none),
    };
    let geo_mask = if env == Some(None) { wgpu::ColorWrites::ALL } else { none };
    vec![
        Some(wgpu::ColorTargetState { format: ENV_FORMAT, blend: env_blend, write_mask: env_mask }),
        Some(wgpu::ColorTargetState { format: GEO_FORMAT, blend: None, write_mask: geo_mask }),
    ]
}

fn env_f32(k: &str, d: f32) -> f32 {
    crate::envc::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct U {
    vp: [[f32; 4]; 4],
    inv_vp: [[f32; 4]; 4],
    prev_vp: [[f32; 4]; 4],
    obj_inv: [[f32; 4]; 4],
    obj_prev: [[f32; 4]; 4],
    eye: [f32; 4],
    size: [f32; 4],
    p0: [f32; 4],
    p1: [f32; 4],
    p2: [f32; 4],
}

struct Hiz {
    all: wgpu::TextureView,
    mips: Vec<wgpu::TextureView>,
}

pub struct Ssr {
    pub env: wgpu::TextureView,
    pub geo: wgpu::TextureView,
    hiz: [Hiz; 2],
    trace: wgpu::TextureView,
    /// the trace after the spatial prefilter
    filt: wgpu::TextureView,
    dist: wgpu::TextureView,
    hist: [wgpu::TextureView; 2],
    hiz0: wgpu::ComputePipeline,
    hizn: wgpu::ComputePipeline,
    trace_pipe: wgpu::ComputePipeline,
    prefilter: wgpu::ComputePipeline,
    temporal: wgpu::ComputePipeline,
    composite: wgpu::RenderPipeline,
    samp: wgpu::Sampler,
    ubuf: wgpu::Buffer,
    idx: usize,
    frame: u32,
    prev_vp: Option<Mat4>,
    obj: Option<Mat4>,
    obj_prev: Option<Mat4>,
    rsize: (u32, u32),
    cutoff: f32,
    thick: f32,
    steps: f32,
    strength: f32,
    hist_w: f32,
    blur: f32,
}

impl Ssr {
    pub fn new(device: &wgpu::Device, scene_format: wgpu::TextureFormat, rw: u32, rh: u32) -> Ssr {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssr"),
            // EVOVIEW_SSR_SHADER=<path>: the WGSL from disk, for tuning without a rebuild
            source: wgpu::ShaderSource::Wgsl(crate::envc::var("EVOVIEW_SSR_SHADER").ok()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .unwrap_or_else(|| include_str!("ssr.wgsl").to_string()).into()),
        });
        let cp = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: entry,
            compilation_options: Default::default(),
            cache: None,
        });
        let composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssr composite"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs_full",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_composite",
                compilation_options: Default::default(),
                // added onto the scene (signed: a dark hit takes sky out)
                targets: &[Some(wgpu::ColorTargetState {
                    format: scene_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                        alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssr history"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssr uniforms"),
            size: std::mem::size_of::<U>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (env, geo, hiz, trace, filt, dist, hist) = Self::textures(device, rw, rh);
        let s = Ssr {
            env, geo, hiz, trace, filt, dist, hist,
            prefilter: cp("prefilter"),
            hiz0: cp("hiz0"),
            hizn: cp("hizn"),
            trace_pipe: cp("trace"),
            temporal: cp("temporal"),
            composite,
            samp,
            ubuf,
            idx: 0,
            frame: 0,
            prev_vp: None,
            obj: None,
            obj_prev: None,
            rsize: (rw, rh),
            cutoff: env_f32("EVOVIEW_SSR_CUTOFF", 0.2),
            thick: env_f32("EVOVIEW_SSR_THICK", 0.02),
            steps: env_f32("EVOVIEW_SSR_STEPS", 64.0),
            strength: env_f32("EVOVIEW_SSR_STRENGTH", 1.0),
            hist_w: env_f32("EVOVIEW_SSR_HIST", 0.15),
            blur: env_f32("EVOVIEW_SSR_BLUR", 4.0),
        };
        println!("ssr: on (cutoff alpha {}, {} steps; EVOVIEW_SSR=0 off)", s.cutoff, s.steps);
        s
    }

    #[allow(clippy::type_complexity)]
    fn textures(device: &wgpu::Device, rw: u32, rh: u32)
            -> (wgpu::TextureView, wgpu::TextureView, [Hiz; 2], wgpu::TextureView, wgpu::TextureView, wgpu::TextureView,
                [wgpu::TextureView; 2]) {
        let tex = |label: &str, fmt, w: u32, h: u32, mips: u32, usage| device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        });
        let rt = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let st = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let env = tex("ssr env", ENV_FORMAT, rw, rh, 1, rt).create_view(&Default::default());
        let geo = tex("ssr geo", GEO_FORMAT, rw, rh, 1, rt).create_view(&Default::default());
        let (hw, hh) = (rw.div_ceil(2), rh.div_ceil(2));
        let pad = 1u32 << (LEVELS - 1);
        let (pw, ph) = (hw.div_ceil(pad) * pad, hh.div_ceil(pad) * pad);
        let hiz = || {
            let t = tex("ssr hiz", wgpu::TextureFormat::R32Float, pw, ph, LEVELS, st);
            Hiz {
                all: t.create_view(&Default::default()),
                mips: (0..LEVELS).map(|m| t.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: m,
                    mip_level_count: Some(1),
                    ..Default::default()
                })).collect(),
            }
        };
        let trace = tex("ssr trace", TRACE_FORMAT, hw, hh, 1, st).create_view(&Default::default());
        let filt = tex("ssr prefiltered", TRACE_FORMAT, hw, hh, 1, st).create_view(&Default::default());
        let dist = tex("ssr hit distance", wgpu::TextureFormat::R32Float, hw, hh, 1, st).create_view(&Default::default());
        let hist = [tex("ssr history", TRACE_FORMAT, hw, hh, 1, st).create_view(&Default::default()),
                    tex("ssr history", TRACE_FORMAT, hw, hh, 1, st).create_view(&Default::default())];
        (env, geo, [hiz(), hiz()], trace, filt, dist, hist)
    }

    pub fn resize(&mut self, device: &wgpu::Device, rw: u32, rh: u32) {
        if (rw, rh) == self.rsize {
            return;
        }
        let (env, geo, hiz, trace, filt, dist, hist) = Self::textures(device, rw, rh);
        (self.env, self.geo, self.hiz, self.trace, self.filt, self.dist, self.hist) = (env, geo, hiz, trace, filt, dist, hist);
        self.rsize = (rw, rh);
        self.prev_vp = None;
    }

    /// The main pass's extra attachments: cleared (nothing reflects) at its
    /// start, loaded when the pass is resumed (the rain's glass pass).
    pub fn attachments(&self, clear: bool) -> [Option<wgpu::RenderPassColorAttachment<'_>>; 2] {
        let load = if clear { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) } else { wgpu::LoadOp::Load };
        let att = |v| Some(wgpu::RenderPassColorAttachment {
            view: v,
            resolve_target: None,
            ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
        });
        [att(&self.env), att(&self.geo)]
    }

    /// The driven car's box -> world this frame (taa.rs set_object), so its
    /// own reflections reproject with it.
    pub fn set_object(&mut self, m: Option<Mat4>) {
        if m.is_none() {
            self.obj_prev = None;
        }
        self.obj = m;
    }

    pub fn reset(&mut self) {
        self.prev_vp = None;
    }

    /// Trace, accumulate and add the reflections onto `scene` (the main
    /// pass's resolved colour, which is also what the rays read). `vp` is
    /// the projection the scene was drawn with (jittered).
    #[allow(clippy::too_many_arguments)]
    pub fn run(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder,
               depth: &wgpu::TextureView, scene: &wgpu::TextureView, vp: Mat4, eye: Vec3) {
        let (rw, rh) = self.rsize;
        let (hw, hh) = (rw.div_ceil(2), rh.div_ceil(2));
        // metres per render pixel at 1 m: 2 / (proj y scale * height)
        let inv = vp.inverse();
        let fy = {
            // the projection's y scale, from the clip-space row lengths
            let r1 = vp.row(1).truncate().length();
            let r3 = vp.row(3).truncate().length();
            r1 / r3.max(1e-6)
        };
        let u = U {
            vp: vp.to_cols_array_2d(),
            inv_vp: inv.to_cols_array_2d(),
            prev_vp: self.prev_vp.unwrap_or(vp).to_cols_array_2d(),
            obj_inv: self.obj.map_or(Mat4::IDENTITY, |m| m.inverse()).to_cols_array_2d(),
            obj_prev: self.obj.map_or(Mat4::IDENTITY, |m| self.obj_prev.unwrap_or(m)).to_cols_array_2d(),
            eye: [eye.x, eye.y, eye.z, self.frame as f32],
            size: [rw as f32, rh as f32, 1.0 / rw.max(1) as f32, 1.0 / rh.max(1) as f32],
            p0: [self.cutoff, self.thick, self.steps, self.strength],
            p1: [if self.obj.is_some() { 1.0 } else { 0.0 }, (LEVELS - 1) as f32,
                 if self.prev_vp.is_none() { 1.0 } else { 0.0 }, self.hist_w],
            p2: [2.0 / (fy * rh.max(1) as f32), self.blur, 0.0, 0.0],
        };
        queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&u));
        let cur = self.idx;
        let prev = 1 - self.idx;
        let bg = |pipe_layout: &wgpu::BindGroupLayout, entries: &[(u32, wgpu::BindingResource)]| {
            let e: Vec<wgpu::BindGroupEntry> = entries.iter()
                .map(|(b, r)| wgpu::BindGroupEntry { binding: *b, resource: r.clone() })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("ssr bg"), layout: pipe_layout, entries: &e })
        };
        use wgpu::BindingResource as R;
        let groups = |w: u32, h: u32| (w.div_ceil(8), h.div_ceil(8));
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ssr"), timestamp_writes: None });
            // Hi-Z
            let l0 = self.hiz0.get_bind_group_layout(0);
            let g0 = bg(&l0, &[(0, self.ubuf.as_entire_binding()), (1, R::TextureView(depth)),
                               (2, R::TextureView(&self.hiz[cur].mips[0]))]);
            cp.set_pipeline(&self.hiz0);
            cp.set_bind_group(0, &g0, &[]);
            let pad = 1u32 << (LEVELS - 1);
            let (pw, ph) = (hw.div_ceil(pad) * pad, hh.div_ceil(pad) * pad);
            let (gx, gy) = groups(pw, ph);
            cp.dispatch_workgroups(gx, gy, 1);
            let ln = self.hizn.get_bind_group_layout(0);
            cp.set_pipeline(&self.hizn);
            for m in 1..LEVELS {
                let g = bg(&ln, &[(2, R::TextureView(&self.hiz[cur].mips[m as usize])),
                                  (3, R::TextureView(&self.hiz[cur].mips[m as usize - 1]))]);
                cp.set_bind_group(0, &g, &[]);
                let (gx, gy) = groups(pw >> m, ph >> m);
                cp.dispatch_workgroups(gx, gy, 1);
            }
            // trace
            let lt = self.trace_pipe.get_bind_group_layout(0);
            let gt = bg(&lt, &[(0, self.ubuf.as_entire_binding()), (1, R::TextureView(depth)),
                               (4, R::TextureView(&self.hiz[cur].all)), (5, R::TextureView(&self.geo)),
                               (6, R::TextureView(scene)), (7, R::TextureView(&self.trace)),
                               (8, R::TextureView(&self.dist)), (9, R::TextureView(&self.env))]);
            cp.set_pipeline(&self.trace_pipe);
            cp.set_bind_group(0, &gt, &[]);
            let (gx, gy) = groups(hw, hh);
            cp.dispatch_workgroups(gx, gy, 1);
            // prefilter
            let lp = self.prefilter.get_bind_group_layout(0);
            let gp = bg(&lp, &[(0, self.ubuf.as_entire_binding()), (4, R::TextureView(&self.hiz[cur].all)),
                               (5, R::TextureView(&self.geo)), (10, R::TextureView(&self.trace)),
                               (11, R::TextureView(&self.dist)), (16, R::TextureView(&self.filt))]);
            cp.set_pipeline(&self.prefilter);
            cp.set_bind_group(0, &gp, &[]);
            cp.dispatch_workgroups(gx, gy, 1);
            // temporal
            let lr = self.temporal.get_bind_group_layout(0);
            let gr = bg(&lr, &[(0, self.ubuf.as_entire_binding()), (4, R::TextureView(&self.hiz[cur].all)),
                               (5, R::TextureView(&self.geo)), (10, R::TextureView(&self.filt)),
                               (11, R::TextureView(&self.dist)), (12, R::TextureView(&self.hist[prev])),
                               (13, R::Sampler(&self.samp)), (14, R::TextureView(&self.hiz[prev].all)),
                               (15, R::TextureView(&self.hist[cur]))]);
            cp.set_pipeline(&self.temporal);
            cp.set_bind_group(0, &gr, &[]);
            cp.dispatch_workgroups(gx, gy, 1);
        }
        crate::bench::mark(enc, "ssr trace");
        {
            let lc = self.composite.get_bind_group_layout(0);
            let gc = bg(&lc, &[(0, self.ubuf.as_entire_binding()), (1, R::TextureView(depth)),
                               (4, R::TextureView(&self.hiz[cur].all)), (5, R::TextureView(&self.geo)),
                               (9, R::TextureView(&self.env)), (12, R::TextureView(&self.hist[cur]))]);
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssr composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.composite);
            pass.set_bind_group(0, &gc, &[]);
            pass.draw(0..3, 0..1);
        }
        self.idx = prev;
        self.prev_vp = Some(vp);
        self.obj_prev = self.obj;
        self.frame = self.frame.wrapping_add(1);
    }
}
