//! Post-processing after TAA on tracks (post.wgsl): the game's veil (bloom),
//! eye-focus vignette, sharpening, motion blur and auto exposure.
//!
//! What the game has and where its numbers come from:
//!   veil        EnvironmentShadingGlobalSetting.natural_veil_params_sun_
//!               elevation (common_assets/curves/veil_curves/natural.curve4):
//!               x intensity 0.5 by day / 0.42 at night, y threshold 2 / 0.02
//!               (the night threshold is what makes lamps glow)
//!   vignette    PostProcessingSettings.eyeParameters (natural / tv presets:
//!               eyeFocusVignette, pow 0.65, intensity 1)
//!   sharpen     VideoSettings.sharpening (AMD CAS in the exe)
//!   motion blur VideoSettings.motionBlur strength
//!   exposure    PostProcessingSettings.exposure: auto exposure on in every
//!               natural / tv preset, speedUp / speedDown per second
//!   dof         the TV cameras' .graphicsoverride depthOfField (cams.rs Dof),
//!               the game's Low/Medium path (exe shaders 047806d0 CoC and
//!               0477f350 composite): a blurred mip chain, two mips mixed in
//!               by a foreground and a background circle of confusion
//! ⚠ The input is the TAA output, already tonemapped and graded, so the veil
//! threshold is mapped into display range and auto exposure is a feedback
//! loop on the displayed luminance (an exposure multiplier the scene shaders
//! pick up next frame through grade_uniform), not the game's histogram of
//! linear light. Strengths not stored in content are guesses (strict PP1).

use glam::{Mat4, Vec3};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

const MIP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const LUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const LUM_W: u32 = 64;
const LUM_H: u32 = 36;
const LEVELS: usize = 6;
/// uniform slots kept for the depth of field's mip chain
const DOF_SLOTS: usize = 16;
/// The blur-chain level clarity takes its local mean from (1/32 size) and
/// the one the sun shafts gather (1/8 size).
const CLARITY_MIP: usize = 4;
const SHAFT_MIP: usize = 2;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct U {
    inv_vp: [[f32; 4]; 4],
    prev_vp: [[f32; 4]; 4],
    obj_inv: [[f32; 4]; 4],
    obj_prev: [[f32; 4]; 4],
    obj_half: [f32; 4],
    veil: [f32; 4],
    fx: [f32; 4],
    dst: [f32; 4],
    src: [f32; 4],
    eye: [f32; 4],
    // x 1 = HDR input (tonemap here through the LUT), y LUT z (1 outdoor)
    hdr: [f32; 4],
    // last frame's view-projection seen from this frame's orientation: the
    // camera's travel without its rotation
    prev_vp_t: [[f32; 4]; 4],
    // x share of the camera rotation's blur
    blur: [f32; 4],
    // depth of field: x focus distance (<= 0 auto focus), y CoC range
    // (20 - aperture * 0.9375), z foreground distance, w background distance
    dof: [f32; 4],
    // x auto-focus samples, y their radius (pixels), zw focus point (uv)
    dof2: [f32; 4],
    // xyz camera forward, w 1 = depth of field on
    fwd: [f32; 4],
    // sun shafts: xy the sun on screen (uv), z intensity, w 1 = on
    shaft: [f32; 4],
    // x clarity (local contrast) strength, y its midtone width; zw unused
    look: [f32; 4],
}

/// The auto-exposure multiplier (log2, as f32 bits) the scene shaders use.
static AE_LOG2: AtomicU32 = AtomicU32::new(0);

/// Exposure multiplier from auto exposure (1 when off).
pub fn ae_mult() -> f32 {
    f32::from_bits(AE_LOG2.load(Ordering::Relaxed)).exp2()
}

fn env_f(name: &str, d: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

struct Target {
    view: wgpu::TextureView,
    w: u32,
    h: u32,
}

pub struct Post {
    pipe_mip: wgpu::RenderPipeline,
    pipe_up: wgpu::RenderPipeline,
    pipe_out: wgpu::RenderPipeline,
    pipe_lum: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    samp: wgpu::Sampler,
    ubufs: Vec<wgpu::Buffer>,
    /// TAA writes here; the composite reads it
    pub input: wgpu::TextureView,
    mips: Vec<Target>,
    /// the depth of field's blurred chain: [0] half size .. 1x1
    dof_mips: Vec<Target>,
    lum: wgpu::Texture,
    lum_view: wgpu::TextureView,
    readback: wgpu::Buffer,
    pending: Arc<AtomicBool>,
    mapped: Arc<AtomicBool>,
    format: wgpu::TextureFormat,
    size: (u32, u32),
    prev_vp: Option<Mat4>,
    obj: Option<(Mat4, Vec3)>,
    obj_prev: Option<Mat4>,
    eye: Vec3,
    prev_eye: Option<Vec3>,
    dt: f32,
    pub blur_rot: f32,
    black: wgpu::TextureView,
    /// strengths (EVOVIEW_VEIL, _VIGNETTE, _SHARPEN, _MOTION_BLUR, _AE)
    pub veil: f32,
    pub vignette: f32,
    pub sharpen: f32,
    pub motion_blur: f32,
    /// moving cars blur by their motion vectors (EVOVIEW_MOTION_BLUR_MV=0:
    /// the depth reprojection only, every car but the driven one static)
    mv_blur: bool,
    pub ae: bool,
    /// the current camera's depth of field (TV cameras), EVOVIEW_DOF=0 off
    pub dof: Option<crate::cams::Dof>,
    pub dof_on: bool,
    /// local contrast (EVOVIEW_CLARITY, 0 off) and sun shafts
    /// (EVOVIEW_SHAFTS, 0 off)
    pub clarity: f32,
    pub shafts: f32,
    ae_log2: f32,
    /// the game's [min, max] brightness clamp on the adapted average
    ae_clamp: bool,
    /// time since the last luminance readback arrived (auto exposure steps
    /// once per readback, every 2-3 frames)
    ae_dt: f32,
    last: std::time::Instant,
}

impl Post {
    /// `format` is what TAA writes (HDR_FORMAT in the HDR pipeline), `out`
    /// the screen's; `lut` the game's grading LUT (grade.rs) the HDR
    /// composite tonemaps through.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, out: wgpu::TextureFormat, w: u32, h: u32,
               samples: u32, _lut: &wgpu::TextureView) -> Post {
        // the output dither is for an 8-bit swapchain only
        let dither = matches!(out, wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Rgba8UnormSrgb)
            && std::env::var("EVOVIEW_DITHER").map_or(true, |v| v != "0");
        let src = include_str!("post.wgsl")
            .replace("DEPTH_TEXTURE", if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" })
            .replace("DITHER_ON", if dither { "true" } else { "false" });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let tex = |b: u32, st: wgpu::TextureSampleType, ms: bool| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type: st, view_dimension: wgpu::TextureViewDimension::D2, multisampled: ms },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex(1, wgpu::TextureSampleType::Float { filterable: true }, false),
                tex(2, wgpu::TextureSampleType::Float { filterable: true }, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex(4, wgpu::TextureSampleType::Depth, samples > 1),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                                     view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                    count: None,
                },
                // depth of field: the foreground and background blur mips
                tex(6, wgpu::TextureSampleType::Float { filterable: true }, false),
                tex(7, wgpu::TextureSampleType::Float { filterable: true }, false),
                // motion vectors (motion.rs), for the motion blur
                tex(8, wgpu::TextureSampleType::Float { filterable: true }, false),
                // the input's blur chain for clarity (local mean) and the sun
                // shafts' source
                tex(9, wgpu::TextureSampleType::Float { filterable: true }, false),
                tex(10, wgpu::TextureSampleType::Float { filterable: true }, false),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let mk = |label: &str, fmt: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: fmt, blend, write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };
        let add = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let pipe_mip = mk("post mip", MIP_FORMAT, None);
        let pipe_up = mk("post up", MIP_FORMAT, Some(add));
        let pipe_out = mk("post out", out, None);
        let pipe_lum = mk("post lum", LUM_FORMAT, None);
        let samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let ubufs = (0..(LEVELS * 2 + 3 + DOF_SLOTS)).map(|_| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post u"),
            size: std::mem::size_of::<U>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })).collect();
        let (input, mips, dof_mips) = Self::targets(device, format, w, h);
        let lum = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post lum"),
            size: wgpu::Extent3d { width: LUM_W, height: LUM_H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LUM_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let lum_view = lum.create_view(&Default::default());
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post lum readback"),
            size: 256 * LUM_H as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let black = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post black"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: MIP_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&Default::default());
        let ae = std::env::var("EVOVIEW_AE").map_or(true, |v| v != "0");
        if !ae {
            AE_LOG2.store(0f32.to_bits(), Ordering::Relaxed);
        }
        Post {
            pipe_mip, pipe_up, pipe_out, pipe_lum, bgl, samp, ubufs, input, mips, dof_mips, lum, lum_view, readback,
            pending: Arc::new(AtomicBool::new(false)),
            mapped: Arc::new(AtomicBool::new(false)),
            format, size: (w, h), prev_vp: None, obj: None, obj_prev: None, eye: Vec3::ZERO, prev_eye: None, dt: 1.0 / 60.0, black,

            veil: env_f("EVOVIEW_VEIL", 1.0),
            // off: Natural 7 (Josh's preset) has no eyeParameters, so the
            // game draws no eye-focus vignette. EVOVIEW_VIGNETTE=1 brings it back.
            vignette: env_f("EVOVIEW_VIGNETTE", 0.0),
            // 0.5. ⚠ Was 1.0, tuned on edge sharpness alone: against a 4x
            // supersampled no-TAA reference of the same moving frame it put
            // 1.25x the reference's fine-detail energy on screen and the
            // surplus was TAA's jitter residue made into grain - textures
            // read low-quality / compressed. 0.5 lands on the reference's
            // detail (1.01x) with the best match to it (correlation 0.794
            // vs 0.762; road 0.889 vs 0.856). gfx/score.py measures it.
            sharpen: env_f("EVOVIEW_SHARPEN", 0.5),
            // the shutter as a share of a 60 Hz frame (4 ms), whatever the
            // frame rate: it was a share of the frame itself, so the smear
            // grew as fps fell. Camera rotation blurs at 35 %: turning in a
            // corner smeared the whole screen.
            motion_blur: env_f("EVOVIEW_MOTION_BLUR", 0.25),
            blur_rot: env_f("EVOVIEW_BLUR_ROT", 0.35),
            mv_blur: std::env::var("EVOVIEW_MOTION_BLUR_MV").map_or(true, |v| v != "0"),
            ae,
            dof: None,
            dof_on: std::env::var("EVOVIEW_DOF").map_or(true, |v| v != "0"),
            // off by default: next to the game they crushed the shade and
            // read as a filter (user, 2026-11-03: 'a lot different from the
            // real game in a worse way, kinda depressing')
            clarity: env_f("EVOVIEW_CLARITY", 0.0),
            shafts: env_f("EVOVIEW_SHAFTS", 0.0),
            ae_log2: 0.0,
            ae_clamp: std::env::var("EVOVIEW_AE_CLAMP").map_or(true, |v| v != "0"),
            ae_dt: 0.0,
            last: std::time::Instant::now(),
        }
    }

    fn targets(device: &wgpu::Device, format: wgpu::TextureFormat, w: u32, h: u32) -> (wgpu::TextureView, Vec<Target>, Vec<Target>) {
        let mk = |label: &str, fmt, w: u32, h: u32| device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&Default::default());
        let input = mk("post input", format, w, h);
        let mut mips = Vec::new();
        let (mut mw, mut mh) = (w, h);
        for _ in 0..LEVELS {
            mw = (mw / 2).max(1);
            mh = (mh / 2).max(1);
            mips.push(Target { view: mk("post mip", MIP_FORMAT, mw, mh), w: mw, h: mh });
        }
        // after the veil's levels: the sun shafts (the 1/4-size one's size)
        let (sw, sh) = (mips[1].w, mips[1].h);
        mips.push(Target { view: mk("post shafts", MIP_FORMAT, sw, sh), w: sw, h: sh });
        let mut dof_mips = Vec::new();
        let (mut mw, mut mh) = (w, h);
        while (mw > 1 || mh > 1) && dof_mips.len() < DOF_SLOTS {
            mw = (mw / 2).max(1);
            mh = (mh / 2).max(1);
            dof_mips.push(Target { view: mk("post dof mip", MIP_FORMAT, mw, mh), w: mw, h: mh });
        }
        (input, mips, dof_mips)
    }

    pub fn resize(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if (w, h) == self.size {
            return;
        }
        let (input, mips, dof_mips) = Self::targets(device, self.format, w, h);
        self.input = input;
        self.mips = mips;
        self.dof_mips = dof_mips;
        self.size = (w, h);
        self.prev_vp = None;
    }

    pub fn set_eye(&mut self, e: Vec3) {
        self.eye = e;
    }

    /// The driven car, as for TAA (taa.rs set_object).
    pub fn set_object(&mut self, m: Option<Mat4>, half: Vec3) {
        if m.is_none() {
            self.obj_prev = None;
        }
        self.obj = m.map(|m| (m, half));
    }

    /// A camera cut: no motion blur across it.
    pub fn cut(&mut self) {
        self.prev_vp = None;
        self.obj_prev = None;
    }

    /// The veil's (intensity, threshold) for the current sun, keyed by sun
    /// zenith (0 overhead .. 180): the curve of the game's Veil video setting
    /// (common_assets/curves/veil_curves/<name>.curve4, x intensity, y
    /// threshold). EVOVIEW_VEIL_CURVE=soft|natural|cinematic, default natural.
    /// ⚠ Josh's video.videosettings leaves `veil` unset, which is enum 0 =
    /// Veil_Soft, but the HDR veil scale (0.14 in run) was tuned on natural.
    fn veil_params() -> (f32, f32) {
        let zen = if crate::sky::active() { crate::sky::now().zenith } else { 40.0 };
        static CURVE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        let lerp = |keys: &[(f32, f32, f32)]| {
            let i = keys.iter().position(|k| zen <= k.0).unwrap_or(keys.len() - 1);
            if i == 0 {
                return (keys[0].1, keys[0].2);
            }
            let (a, b) = (keys[i - 1], keys[i]);
            let t = ((zen - a.0) / (b.0 - a.0).max(1e-3)).clamp(0.0, 1.0);
            (a.1 + (b.1 - a.1) * t, a.2 + (b.2 - a.2) * t)
        };
        match CURVE.get_or_init(|| std::env::var("EVOVIEW_VEIL_CURVE").unwrap_or_default().to_lowercase()).as_str() {
            "soft" => lerp(&[(0.0, 0.35, 0.3), (80.0, 0.35, 0.3), (90.0, 0.33, 0.005), (180.0, 0.33, 0.005)]),
            "cinematic" => lerp(&[(0.0, 1.3, 1.5), (50.0, 1.3, 1.5), (80.0, 1.0, 1.1), (90.0, 0.9, 0.05),
                                  (100.0, 0.9, 0.01), (180.0, 0.9, 0.01)]),
            _ => lerp(&[(0.0, 0.5, 2.0), (80.0, 0.5, 2.0), (90.0, 0.42, 0.02), (180.0, 0.42, 0.02)]),
        }
    }

    /// Read back last frames' luminance and move the exposure toward the
    /// target, at the preset's speeds (log2 units a second).
    fn adapt(&mut self) {
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last).as_secs_f32().min(0.25);
        self.last = now;
        // EVOVIEW_FIXED_DT (captures, A/B shots): the blur follows the
        // simulated frame, not how fast this machine drew it
        static FIXED: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
        self.dt = FIXED.get_or_init(|| std::env::var("EVOVIEW_FIXED_DT").ok().and_then(|v| v.parse::<f32>().ok()))
            .unwrap_or(dt).max(1e-4);
        // ⚠ the readback lands every 2-3 frames: stepped by one frame's dt
        // each time, the exposure adapted at 1/2-1/3 of its speed
        self.ae_dt = (self.ae_dt + dt).min(0.25);
        if !self.ae || !self.mapped.load(Ordering::Acquire) {
            return;
        }
        let dt = std::mem::take(&mut self.ae_dt);
        let mut sum = 0.0f32;
        let mut n = 0u32;
        {
            let data = self.readback.slice(..).get_mapped_range();
            for y in 0..LUM_H as usize {
                for x in 0..LUM_W as usize {
                    let o = y * 256 + x * 2;
                    let v = f16_to_f32(u16::from_le_bytes([data[o], data[o + 1]]));
                    // the centre counts double (what the eye looks at)
                    let dx = x as f32 / LUM_W as f32 - 0.5;
                    let dy = y as f32 / LUM_H as f32 - 0.5;
                    let w = if dx * dx + dy * dy < 0.08 { 2 } else { 1 };
                    if v.is_finite() {
                        sum += v * w as f32;
                        n += w;
                    }
                }
            }
        }
        self.readback.unmap();
        self.mapped.store(false, Ordering::Release);
        REQUESTED.store(false, Ordering::Release);
        self.pending.store(false, Ordering::Release);
        if n == 0 {
            return;
        }
        let avg = sum / n as f32;
        // the displayed average log2 luminance of a well exposed daylight
        // frame (measured on the calibrated sunny Monza view: -3.1 at x1)
        // HDR measures linear light: the same calibrated frame is -2.84
        let hdr = crate::render::HDR.load(Ordering::Relaxed);
        let target = env_f("EVOVIEW_AE_TARGET", if hdr { -2.85 } else { -3.0 });
        let mut err = (target - avg).clamp(-2.0, 2.0);
        // ⚠ The game divides by the adapted average CLAMPED to the preset's
        // [minBrightness, maxBrightness] (natural7 by day: 0.05..10). A day
        // scene in the game's units sits far above 10, so its exposure is
        // pinned: a cockpit or a shaded pit lane does not open up. Without
        // the clamp this loop raised the cockpit x2.7, pushed the outside
        // through the path-to-white shoulder and washed it out (windscreen
        // HSV saturation 0.137 vs 0.164 pinned). Model it: this frame's
        // average in game units = the sky model's average x how far the
        // frame sits from a normal one; the exposure the clamp allows,
        // relative to the sky model's, is the target. EVOVIEW_AE_CLAMP=0 off.
        if self.ae_clamp {
            let (l_model, lo, hi) = crate::sky::ae_clamp();
            let sky_exp = crate::sky::exposure().max(1e-6).log2();
            // how far the scene itself (exposure removed) sits from a normal frame
            let dev = avg - self.ae_log2 - sky_exp - target;
            let l_now = l_model * dev.exp2();
            let want = (l_model.clamp(lo, hi) / l_now.clamp(lo, hi)).log2();
            // Only where the model has the game pinned at the max clamp (day):
            // the night average leaves out the lamps (it would pin a lit
            // night at minBrightness, 1.5 stops darker than before), so
            // dusk and night keep the free loop. Blended over +-1 stop.
            let day = ((l_model / hi).log2() * 0.5 + 0.5).clamp(0.0, 1.0);
            err = err + ((want - self.ae_log2).clamp(-2.0, 2.0) - err) * day;
        }
        let (up, down) = (2.0f32, 1.0f32);
        let rate = if err > 0.0 { up } else { down };
        self.ae_log2 = (self.ae_log2 + err * rate * 0.35 * dt).clamp(-1.0, 1.6);
        AE_LOG2.store(self.ae_log2.to_bits(), Ordering::Relaxed);
        if std::env::var_os("EVOVIEW_AE_LOG").is_some() {
            eprintln!("auto exposure: avg log2 lum {avg:.2}, target {target:.2}, exposure x{:.2} (step over {:.1} ms)",
                      self.ae_log2.exp2(), dt * 1e3);
        }
    }

    fn pass(&self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, i: usize, u: U,
            pipe: &wgpu::RenderPipeline, a: &wgpu::TextureView, b: &wgpu::TextureView, depth: &wgpu::TextureView,
            out: &wgpu::TextureView, load: bool, w: u32, h: u32, lut: &wgpu::TextureView) {
        self.pass_dof(device, queue, enc, i, u, pipe, a, b, depth, out, load, w, h, lut, (b, b), &self.black, (b, b));
    }

    fn pass_dof(&self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, i: usize, mut u: U,
            pipe: &wgpu::RenderPipeline, a: &wgpu::TextureView, b: &wgpu::TextureView, depth: &wgpu::TextureView,
            out: &wgpu::TextureView, load: bool, w: u32, h: u32, lut: &wgpu::TextureView,
            dof: (&wgpu::TextureView, &wgpu::TextureView), mv: &wgpu::TextureView,
            look: (&wgpu::TextureView, &wgpu::TextureView)) {
        u.dst = [w as f32, h as f32, 1.0 / w.max(1) as f32, 1.0 / h.max(1) as f32];
        queue.write_buffer(&self.ubufs[i], 0, bytemuck::bytes_of(&u));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.ubufs[i].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(a) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(b) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.samp) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(lut) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(dof.0) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(dof.1) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(mv) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(look.0) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(look.1) },
            ],
        });
        let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("post"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: out,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if load { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(wgpu::Color::BLACK) },
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        p.set_pipeline(pipe);
        p.set_bind_group(0, &bg, &[]);
        p.draw(0..3, 0..1);
    }

    /// Everything after TAA: `self.input` -> `out`. `vp` is the unjittered
    /// view-projection of this frame; `mv` this frame's motion vectors
    /// (motion.rs), when there are any.
    #[allow(clippy::too_many_arguments)]
    pub fn run(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder,
               depth: &wgpu::TextureView, out: &wgpu::TextureView, vp: Mat4, lut: &wgpu::TextureView,
               mv: Option<&wgpu::TextureView>) {
        self.adapt();
        let (w, h) = self.size;
        let (vi, vt) = Self::veil_params();
        let hdr = crate::render::HDR.load(Ordering::Relaxed);
        // HDR: the game's threshold is in exposed linear light, as is ours
        let thr = if hdr { vt } else { vt / (1.0 + vt) };
        let base = U {
            inv_vp: vp.inverse().to_cols_array_2d(),
            prev_vp: self.prev_vp.unwrap_or(vp).to_cols_array_2d(),
            obj_inv: self.obj.map_or(Mat4::IDENTITY, |o| o.0.inverse()).to_cols_array_2d(),
            obj_prev: self.obj.map_or(Mat4::IDENTITY, |o| self.obj_prev.unwrap_or(o.0)).to_cols_array_2d(),
            obj_half: self.obj.map_or([0.0; 4], |o| [o.1.x, o.1.y, o.1.z, 1.0]),
            // display-range bloom is weaker than the game's HDR veil: scale
            veil: [thr, vi * if hdr { 0.14 } else { 0.35 } * self.veil, 0.5, 0.0],
            fx: [self.vignette, 0.65, self.sharpen,
                 if self.prev_vp.is_some() { self.motion_blur * (1.0 / 60.0) / self.dt.max(1.0 / 1000.0) } else { 0.0 }],
            dst: [0.0; 4],
            src: [1.0 / w.max(1) as f32, 1.0 / h.max(1) as f32, 0.0, 0.0],
            eye: [self.eye.x, self.eye.y, self.eye.z, 1.0],
            hdr: [if crate::render::HDR.load(Ordering::Relaxed) { 1.0 } else { 0.0 }, 1.0, 0.0, 0.0],
            prev_vp_t: (vp * Mat4::from_translation(self.eye - self.prev_eye.unwrap_or(self.eye))).to_cols_array_2d(),
            // y 1: the cars' pixels move by the motion vectors
            blur: [self.blur_rot, if mv.is_some() && self.mv_blur { 1.0 } else { 0.0 }, 0.0, 0.0],
            dof: [0.0; 4],
            dof2: [0.0; 4],
            fwd: [0.0; 4],
            shaft: [0.0; 4],
            look: [0.0; 4],
        };
        let mut slot = 0;
        if self.veil > 0.0 {
            // down: input -> mip0 (threshold), mip i-1 -> mip i
            for i in 0..LEVELS {
                let mut u = base;
                let (src_view, sw, sh) = if i == 0 { (&self.input, w, h) } else { (&self.mips[i - 1].view, self.mips[i - 1].w, self.mips[i - 1].h) };
                u.src = [1.0 / sw.max(1) as f32, 1.0 / sh.max(1) as f32, 0.0, 0.0];
                u.veil[3] = if i == 0 { 0.0 } else { 1.0 };
                let t = &self.mips[i];
                self.pass(device, queue, enc, slot, u, &self.pipe_mip, src_view, &self.black, depth, &t.view, false, t.w, t.h, lut);
                slot += 1;
            }
            // up: mip i+1 added into mip i
            for i in (0..LEVELS - 1).rev() {
                let mut u = base;
                let s = &self.mips[i + 1];
                u.src = [1.0 / s.w.max(1) as f32, 1.0 / s.h.max(1) as f32, 0.0, 0.0];
                u.veil[3] = 2.0;
                let t = &self.mips[i];
                self.pass(device, queue, enc, slot, u, &self.pipe_up, &s.view, &self.black, depth, &t.view, true, t.w, t.h, lut);
                slot += 1;
            }
        }
        crate::bench::mark(enc, "bloom");
        // depth of field (TV cameras): the input's blurred mip chain, the
        // game's gaussianBlurredColorTexture
        let mut u = base;
        let mut dof_views = (&self.black, &self.black);
        let dof = self.dof.filter(|_| self.dof_on && !self.dof_mips.is_empty());
        // the input's blur chain: all of it for depth of field, to the
        // clarity mip otherwise
        let look_on = self.clarity > 0.0 || self.shafts > 0.0;
        let chain = if dof.is_some() { self.dof_mips.len().min(DOF_SLOTS) }
                    else if look_on { (CLARITY_MIP + 1).min(self.dof_mips.len()) } else { 0 };
        for i in 0..chain {
            let mut du = base;
            let (src_view, sw, sh) = if i == 0 { (&self.input, w, h) } else { (&self.dof_mips[i - 1].view, self.dof_mips[i - 1].w, self.dof_mips[i - 1].h) };
            du.src = [1.0 / sw.max(1) as f32, 1.0 / sh.max(1) as f32, 0.0, 0.0];
            du.veil[3] = 1.0;
            let t = &self.dof_mips[i];
            self.pass(device, queue, enc, LEVELS * 2 + 2 + i, du, &self.pipe_mip, src_view, &self.black, depth, &t.view, false, t.w, t.h, lut);
        }
        let look_views = if look_on && chain > CLARITY_MIP {
            (&self.dof_mips[CLARITY_MIP].view, &self.mips[LEVELS].view)
        } else {
            (&self.black, &self.black)
        };
        if let Some(d) = dof {
            let n = self.dof_mips.len().min(DOF_SLOTS);
            // GUESS: the blur settings (0.15 .. 1, some 150 / 700) pick an
            // integer mip. The exe's mapping is not decoded. As a share of
            // the whole chain, 1.0 picked the 1x1 mip: out-of-focus cars
            // became flat smears of colour ("censored"). A lens blurs a few
            // pixels: 1.0 = mip 3 (1/8 size), EVOVIEW_DOF_MAX_MIP to tune.
            let levels = n + 1;
            let top = std::env::var("EVOVIEW_DOF_MAX_MIP").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(3).min(levels - 1);
            let mip = |b: f32| ((b.clamp(0.0, 1.0) * top as f32).round() as usize).min(top);
            let view = |m: usize| if m == 0 { &self.input } else { &self.dof_mips[m - 1].view };
            dof_views = (view(mip(d.fg_blur)), view(mip(d.bg_blur)));
            let range = 20.0 - d.aperture * 0.9375;
            let range = if range.abs() < 1e-3 { 1e-3 } else { range };
            let inv = vp.inverse();
            let a = inv * glam::Vec4::new(0.0, 0.0, 0.5, 1.0);
            let fwd = (a.truncate() / a.w - self.eye).normalize_or_zero();
            let c = d.center.unwrap_or([0.5, 0.5]);
            u.dof = [d.focus.unwrap_or(0.0), range, d.fg_dist, d.bg_dist];
            u.dof2 = [d.coc_samples.clamp(1, 32) as f32, d.coc_depth_radius, c[0], c[1]];
            // EVOVIEW_DOF=debug: red the foreground CoC, green the background
            let dbg = std::env::var("EVOVIEW_DOF").map_or(false, |v| v == "debug");
            u.fwd = [fwd.x, fwd.y, fwd.z, if dbg { 2.0 } else { 1.0 }];
            if std::env::var_os("EVOVIEW_DOF_LOG").is_some() {
                eprintln!("dof {d:?} mips fg {} bg {} of {levels}, range {range:.2}", mip(d.fg_blur), mip(d.bg_blur));
            }
        }
        // composite; hdr.w 1: the game's B-spline veil read
        // (EVOVIEW_VEIL_BSPLINE=0 the old bilinear one)
        if std::env::var("EVOVIEW_VEIL_BSPLINE").map_or(true, |v| v != "0") {
            u.hdr[3] = 1.0;
        }
        u.veil[3] = 3.0;
        if self.veil <= 0.0 {
            u.veil[1] = 0.0;
        }
        if look_on && chain > CLARITY_MIP {
            u.look = [self.clarity, env_f("EVOVIEW_CLARITY_MID", 0.5), 0.0, 0.0];
            // the sun as a point at infinity: on screen when in front
            let sd = crate::sky::key_dir();
            if std::env::var_os("EVOVIEW_SHAFT_LOG").is_some() {
                eprintln!("sun dir {sd:?}");
            }
            let c = vp * glam::Vec4::new(sd.x, sd.y, sd.z, 0.0);
            if self.shafts > 0.0 && c.w > 1e-4 && sd.y > -0.05 {
                let suv = [c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5];
                // fade the sun out from 0.5 screens past the edge, and as it sets
                let off = (suv[0] - 0.5).abs().max((suv[1] - 0.5).abs());
                let k = (1.0 - (off - 1.0) / 0.5).clamp(0.0, 1.0) * (sd.y / 0.1).clamp(0.0, 1.0);
                if k > 0.0 {
                    u.shaft = [suv[0], suv[1], self.shafts * k, 1.0];
                    // the rays, gathered at 1/4 size from the 1/8-size blur
                    let mut su = base;
                    su.shaft = u.shaft;
                    let src = &self.dof_mips[SHAFT_MIP];
                    su.src = [1.0 / src.w as f32, 1.0 / src.h as f32, 0.0, 0.0];
                    su.veil[3] = 5.0;
                    let t = &self.mips[LEVELS];
                    self.pass(device, queue, enc, LEVELS * 2 + 2 + DOF_SLOTS, su, &self.pipe_mip, &src.view, &self.black, depth, &t.view, false, t.w, t.h, lut);
                }
            }
        }
        let bloom = if self.veil > 0.0 { &self.mips[0].view } else { &self.black };
        self.pass_dof(device, queue, enc, slot, u, &self.pipe_out, &self.input, bloom, depth, out, false, w, h, lut, dof_views,
                      mv.unwrap_or(&self.black), look_views);
        slot += 1;
        // auto exposure: a 64 x 36 log-luminance image, read back later
        if self.ae && !self.pending.load(Ordering::Acquire) {
            let mut u = base;
            u.veil[3] = 4.0;
            self.pass(device, queue, enc, slot, u, &self.pipe_lum, &self.input, &self.black, depth, &self.lum_view, false, LUM_W, LUM_H, lut);
            enc.copy_texture_to_buffer(
                self.lum.as_image_copy(),
                wgpu::ImageCopyBuffer {
                    buffer: &self.readback,
                    layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(LUM_H) },
                },
                wgpu::Extent3d { width: LUM_W, height: LUM_H, depth_or_array_layers: 1 },
            );
            self.pending.store(true, Ordering::Release);
        }
        self.prev_vp = Some(vp);
        self.prev_eye = Some(self.eye);
        self.obj_prev = self.obj.map(|o| o.0);
    }

    /// After the frame's submit: ask for the luminance readback.
    pub fn after_submit(&self) {
        if self.ae && self.pending.load(Ordering::Acquire) && !self.mapped.load(Ordering::Acquire) {
            let mapped = self.mapped.clone();
            // map_async may be requested only once per pending copy
            if !REQUESTED.load(Ordering::Acquire) {
                self.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    if r.is_ok() {
                        mapped.store(true, Ordering::Release);
                    }
                });
                REQUESTED.store(true, Ordering::Release);
            }
        }
    }
}

static REQUESTED: AtomicBool = AtomicBool::new(false);

fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    match e {
        0 => s * m * (2f32).powi(-24),
        31 => if m == 0.0 { s * f32::INFINITY } else { f32::NAN },
        _ => s * (1.0 + m / 1024.0) * (2f32).powi(e - 15),
    }
}
