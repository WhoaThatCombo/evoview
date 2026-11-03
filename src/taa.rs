//! Temporal anti-aliasing for the interactive viewer (taa.wgsl).
//!
//! 4x MSAA alone leaves thin distant geometry - fences, wires, kerb lines,
//! tree-card edges - shimmering from frame to frame, because a few samples
//! a pixel cannot resolve something a tenth of a pixel wide. TAA jitters the
//! projection by a sub-pixel Halton offset every frame and blends each new
//! frame into a reprojected history, so a still or slowly moving view
//! integrates many sample positions.
//!
//! Reprojection: per-pixel motion vectors (motion.rs) - the camera through
//! the depth buffer, and every car (driven, replay, its wheels) from its own
//! last-frame bones and instances. EVOVIEW_MOTION=0 falls back to depth + the
//! previous camera with the driven car as one box (replay cars smeared).
//! `run(.., ext)` takes an external upscaler's output (DLSS) instead of the
//! history and only adds the occlusion and fog (fs_ext).
//! The main pass resolves into `cur` instead of the swapchain; `run` writes
//! the swapchain and the next history in one pass. EVOVIEW_TAA=0 turns it
//! off; the T key toggles it in the viewer.
//!
//! Upsampling (EVOVIEW_RENDER_SCALE below 1): the scene is drawn smaller
//! than the screen and the jitter walks render pixels; the resolve takes each
//! screen pixel's nearest jittered sample, weighted by how near it fell, and
//! the history builds the full-size image over a few frames. Occlusion and
//! fog are computed once per render pixel (fs_aofog) in either case.

use glam::{Mat4, Vec4};

pub const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// occlusion + its view depth (fs_aofog's disocclusion test)
pub const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct U {
    inv_vp: [[f32; 4]; 4],
    prev_vp: [[f32; 4]; 4],
    ctl: [f32; 4],
    size: [f32; 4],
    // the driven car (set_object): world -> its box space this frame, and
    // its box space -> world last frame; obj_half.w 1 = in use
    obj_inv: [[f32; 4]; 4],
    obj_prev: [[f32; 4]; 4],
    obj_half: [f32; 4],
    // this frame's unjittered world -> clip (the depth kept in history.a)
    vp: [[f32; 4]; 4],
    eye: [f32; 4],
    // ambient occlusion: radius m, strength, frame, on
    ao: [f32; 4],
    // volumetric fog (set_volumetrics): sun shadow cascades
    lvp_near: [[f32; 4]; 4],
    lvp_far: [[f32; 4]; 4],
    // xyz towards the sun, w strength
    sun: [f32; 4],
    // rgb sun colour, w 1 = on
    sun_col: [f32; 4],
    // x density /m, y forward g, z sun boost, w max distance m
    fog: [f32; 4],
    // rgb ambient (sky) colour, w far cascade present
    amb: [f32; 4],
    // the driven car's wheels: xyz centre, w radius (0 = none)
    wheels: [[f32; 4]; 4],
    rsize: [f32; 4],
    jit: [f32; 4],
    obj_rect: [f32; 4],
    // x the current frame's weight on sky pixels
    sky: [f32; 4],
    // the middle cascade (layer 2), and per cascade (x near, y far, z mid)
    // the colour pass's depth bias and normal offset (m); sbias.w 1 = mid on
    lvp_mid: [[f32; 4]; 4],
    sbias: [f32; 4],
    snofs: [f32; 4],
}

/// Per-frame inputs of the volumetric fog (set_volumetrics).
#[derive(Clone, Copy)]
pub struct Volumetrics {
    pub lvp_near: glam::Mat4,
    pub lvp_far: glam::Mat4,
    pub far: bool,
    /// the middle cascade, None = off
    pub lvp_mid: Option<glam::Mat4>,
    /// the cascades' depth bias and normal offset (m), near / far / mid, as
    /// the colour pass has them (render::Uniforms shadow / _far / _mid y, w)
    pub bias: [f32; 3],
    pub nofs: [f32; 3],
    pub sun_dir: glam::Vec3,
    pub sun_strength: f32,
    pub sun_col: [f32; 3],
    pub amb: [f32; 3],
}

pub struct Taa {
    pipe: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    /// fs_aofog: this frame's occlusion + fog at the render size
    aofog_pipe: wgpu::RenderPipeline,
    aofog_bgl: wgpu::BindGroupLayout,
    samp: wgpu::Sampler,
    ubuf: wgpu::Buffer,
    pub cur: wgpu::TextureView,
    /// `cur`'s texture (DLSS reads it by its Vulkan image)
    pub cur_tex: wgpu::Texture,
    /// fs_ext: an external upscaler's output (DLSS) + this frame's
    /// occlusion and fog, no history
    ext_pipe: wgpu::RenderPipeline,
    hist: [wgpu::TextureView; 2],
    /// ambient-occlusion history (occlusion, view depth) at fs_aofog's
    /// size, ping-ponged with `hist`
    ao_hist: [wgpu::TextureView; 2],
    /// the camera position this frame (set_eye), for the occlusion pass
    eye: glam::Vec3,
    /// EVOVIEW_AO=0 off; strength otherwise (default 0.85)
    pub ao_strength: f32,
    /// volumetric fog history (Rgba16F: in-scatter rgb, transmittance a),
    /// at fs_aofog's size
    fog_hist: [wgpu::TextureView; 2],
    vol: Option<Volumetrics>,
    /// EVOVIEW_FOG density per metre (0 off)
    pub fog_density: f32,
    idx: usize,
    frame: u32,
    prev_vp: Option<Mat4>,
    size: (u32, u32),
    /// what the scene is drawn at (`cur` and the depth): `size` x the
    /// render scale
    rsize: (u32, u32),
    format: wgpu::TextureFormat,
    pub on: bool,
    obj: Option<(Mat4, glam::Vec3)>,
    obj_prev: Option<Mat4>,
    wheels: [[f32; 4]; 4],
    /// stands in for the motion vectors when there are none
    no_mv: wgpu::TextureView,
}

/// Whether TAA starts on (EVOVIEW_TAA=0 off).
pub fn wanted() -> bool {
    std::env::var("EVOVIEW_TAA").map_or(true, |v| v != "0")
}

/// Texture LOD bias for the scene's materials while TAA resolves them
/// (EVOVIEW_MIP_BIAS overrides). The jitter samples every texel over a few
/// frames, so mips can be picked half a level sharper than one frame would
/// alias at - the usual TAA bias - and, upsampling, by the screen's size
/// rather than the smaller render's. Without it trilinear blends in the
/// half-size mip wherever a texture is a little minified, and liveries and
/// signs read soft next to the game.
pub fn mip_bias() -> f32 {
    if let Some(b) = std::env::var("EVOVIEW_MIP_BIAS").ok().and_then(|v| v.parse::<f32>().ok()) {
        return b;
    }
    if !wanted() {
        return 0.0;
    }
    // a full level (was half): on the wheel test distant road, grass and
    // signs read sharper at -1 with no shimmer under TAA
    render_scale().log2() - 1.0
}

/// `src` with every plain textureSample(...) turned into a textureSampleBias
/// by `bias` (wgpu 22 samplers have no LOD bias of their own). Only colour
/// textures go through textureSample in these shaders; depth maps use the
/// Compare / Load forms, which are left alone.
pub fn with_mip_bias(src: &str, bias: f32) -> String {
    if bias == 0.0 {
        return src.to_string();
    }
    with_mip_bias_expr(src, &format!("{bias:.3}"))
}

/// `with_mip_bias` by a WGSL expression (a uniform: one shader for views
/// with and without TAA).
pub fn with_mip_bias_expr(src: &str, bias: &str) -> String {
    const KEY: &str = "textureSample(";
    let mut out = String::with_capacity(src.len() + 4096);
    let mut rest = src;
    while let Some(i) = rest.find(KEY) {
        out.push_str(&rest[..i]);
        out.push_str("textureSampleBias(");
        rest = &rest[i + KEY.len()..];
        let mut depth = 1;
        let mut end = rest.len();
        for (k, c) in rest.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = k;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push_str(&rest[..end]);
        out.push_str(", ");
        out.push_str(bias);
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}

/// EVOVIEW_RENDER_SCALE: the scene's size over the screen's, per axis
/// (0.5..1, default 1).
pub fn render_scale() -> f32 {
    // down to a third for DLSS / FSR's ultra performance modes
    let lo = if crate::dlss::wanted() || crate::fsr::wanted() { 0.33 } else { 0.5 };
    std::env::var("EVOVIEW_RENDER_SCALE").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0).clamp(lo, 1.0)
}

/// Render pixels per occlusion / fog texel, per axis (EVOVIEW_AOFOG_FULL=1: 1).
fn aofog_step() -> u32 {
    if std::env::var_os("EVOVIEW_AOFOG_FULL").is_some() { 1 } else { 2 }
}

/// The render size for a screen of `w` x `h` at `scale`.
pub fn scaled(w: u32, h: u32, scale: f32) -> (u32, u32) {
    if scale >= 0.999 {
        return (w.max(1), h.max(1));
    }
    (((w as f32 * scale).round() as u32).max(1), ((h as f32 * scale).round() as u32).max(1))
}

fn halton(i: u32, b: u32) -> f32 {
    let (mut f, mut r, mut i) = (1.0f32, 0.0f32, i);
    while i > 0 {
        f /= b as f32;
        r += f * (i % b) as f32;
        i /= b;
    }
    r
}

fn env_f(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

impl Taa {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, w: u32, h: u32, rw: u32, rh: u32, samples: u32) -> Taa {
        let src = include_str!("taa.wgsl")
            .replace("DEPTH_TEXTURE", if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" })
            // EVOVIEW_TAA_MOVE_W: this frame's share on any moving pixel
            .replace("TAA_MOVE_W", &format!("{:.3}", std::env::var("EVOVIEW_TAA_MOVE_W").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.2).clamp(0.1, 0.35)))
            .replace("TAA_GAMMA", &format!("{:.3}", std::env::var("EVOVIEW_TAA_GAMMA").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.25)))
            .replace("FOG_STEPS", &std::env::var("EVOVIEW_FOG_STEPS").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(24).clamp(4, 64).to_string())
            .replace("AO_SAMPLES", &std::env::var("EVOVIEW_AO_SAMPLES").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(12).clamp(4, 32).to_string())
            .replace("AOF_STEP", if aofog_step() == 2 { "2" } else { "1" })
            // depth-aware upsampling + bilinear, depth-tested history of the
            // half-res occlusion / fog (EVOVIEW_AOFOG_UP=0: plain bilinear)
            .replace("AOFUP_ON", if std::env::var("EVOVIEW_AOFOG_UP").map_or(true, |v| v != "0") { "true" } else { "false" })
            // ambient occlusion (the game's 04732820) and contact shadows:
            // EVOVIEW_AO_POW, _AO_BIAS, _AO_SUNLIT; EVOVIEW_CONTACT=0 off,
            // EVOVIEW_CONTACT_STEPS / _PX / _LEN / _THICK (march steps, render px,
            // max m, surface thickness m)
            .replace("AO_POW", &format!("{:.3}", env_f("EVOVIEW_AO_POW", 2.5)))
            .replace("AO_BIAS", &format!("{:.3}", env_f("EVOVIEW_AO_BIAS", 0.1)))
            .replace("AO_SUNLIT", &format!("{:.3}", env_f("EVOVIEW_AO_SUNLIT", 0.6).clamp(0.0, 1.0)))
            .replace("CS_STEPS", &(if std::env::var("EVOVIEW_CONTACT").map_or(false, |v| v == "0") { 0 }
                                   else { env_f("EVOVIEW_CONTACT_STEPS", 12.0).clamp(1.0, 64.0) as u32 }).to_string())
            .replace("CS_PX", &format!("{:.3}", env_f("EVOVIEW_CONTACT_PX", 40.0)))
            .replace("CS_MAXLEN", &format!("{:.3}", env_f("EVOVIEW_CONTACT_LEN", 1.0)))
            .replace("CS_THICKK", &format!("{:.4}", env_f("EVOVIEW_CONTACT_THICK_K", 0.001)))
            .replace("CS_THICK", &format!("{:.3}", env_f("EVOVIEW_CONTACT_THICK", 0.05)))
            .replace("SV_OLD", if std::env::var_os("EVOVIEW_SUNVIS_OLD").is_some() { "true" } else { "false" })
            .replace("CS_OLD", if std::env::var_os("EVOVIEW_CONTACT_OLD").is_some() { "true" } else { "false" })
            .replace("MV_DEBUG", if std::env::var_os("EVOVIEW_MOTION_DEBUG").is_some() { "true" } else { "false" })
            .replace("AO_CAR_W", &format!("{:.3}", std::env::var("EVOVIEW_AO_CAR_W").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.2)))
            .replace("SPIN_FAST_W", &format!("{:.3}", std::env::var("EVOVIEW_WHEEL_TAA_W").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0)))
            .replace("MV_WHEELS_OFF", if std::env::var("EVOVIEW_MOTION_WHEELS").map_or(false, |v| v == "0") { "true" } else { "false" })
            .replace("DEPTH_LOAD", if samples > 1 { "textureLoad(t_depth, p, 0)" } else { "textureLoad(t_depth, p, 0)" });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("taa"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let tex = |b: u32, st: wgpu::TextureSampleType, ms: bool| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type: st, view_dimension: wgpu::TextureViewDimension::D2, multisampled: ms },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taa bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex(1, wgpu::TextureSampleType::Float { filterable: true }, false),
                tex(2, wgpu::TextureSampleType::Float { filterable: true }, false),
                tex(3, wgpu::TextureSampleType::Depth, samples > 1),
                tex(5, wgpu::TextureSampleType::Float { filterable: true }, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                tex(8, wgpu::TextureSampleType::Float { filterable: true }, false),
                tex(9, wgpu::TextureSampleType::Float { filterable: true }, false),
                tex(10, wgpu::TextureSampleType::Float { filterable: true }, false),
                tex(11, wgpu::TextureSampleType::Float { filterable: true }, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let aofog_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taa aofog bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex(3, wgpu::TextureSampleType::Depth, samples > 1),
                tex(5, wgpu::TextureSampleType::Float { filterable: true }, false),
                tex(8, wgpu::TextureSampleType::Float { filterable: true }, false),
                tex(11, wgpu::TextureSampleType::Float { filterable: true }, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let aofog_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taa aofog layout"),
            bind_group_layouts: &[&aofog_bgl],
            push_constant_ranges: &[],
        });
        let aofog_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("taa aofog"),
            layout: Some(&aofog_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_aofog",
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState { format: AO_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: HISTORY_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                ],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taa layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("taa"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: HISTORY_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                ],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let ext_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("taa ext"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_ext",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("taa"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taa"),
            size: std::mem::size_of::<U>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hist = Self::textures(device, w, h);
        let (cur_tex, cur, ao_hist, fog_hist) = Self::render_textures(device, format, rw, rh);
        let fog_density = std::env::var("EVOVIEW_FOG").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0012f32);
        let ao_strength = std::env::var("EVOVIEW_AO").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0f32);
        Taa { pipe, bgl, aofog_pipe, aofog_bgl, samp, ubuf, cur, cur_tex, ext_pipe, hist, ao_hist, eye: glam::Vec3::ZERO, ao_strength, fog_hist, vol: None, fog_density,
              idx: 0, frame: 0, prev_vp: None, size: (w, h), rsize: (rw, rh), format, on: wanted(), obj: None, obj_prev: None, wheels: [[0.0; 4]; 4],
              no_mv: Self::mk(device, "taa no motion", crate::motion::FORMAT, 1, 1) }
    }

    fn mk(device: &wgpu::Device, label: &str, fmt: wgpu::TextureFormat, w: u32, h: u32) -> wgpu::TextureView {
        Self::mk_tex(device, label, fmt, w, h).create_view(&Default::default())
    }

    fn mk_tex(device: &wgpu::Device, label: &str, fmt: wgpu::TextureFormat, w: u32, h: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    /// The render-size targets: the scene, and its occlusion and fog
    /// histories (at fs_aofog's size).
    fn render_textures(device: &wgpu::Device, format: wgpu::TextureFormat, rw: u32, rh: u32)
            -> (wgpu::Texture, wgpu::TextureView, [wgpu::TextureView; 2], [wgpu::TextureView; 2]) {
        let s = aofog_step();
        let (aw, ah) = (rw.div_ceil(s), rh.div_ceil(s));
        let cur = Self::mk_tex(device, "taa current", format, rw, rh);
        let cv = cur.create_view(&Default::default());
        (cur, cv,
         [Self::mk(device, "taa ao 0", AO_FORMAT, aw, ah), Self::mk(device, "taa ao 1", AO_FORMAT, aw, ah)],
         [Self::mk(device, "taa fog 0", HISTORY_FORMAT, aw, ah), Self::mk(device, "taa fog 1", HISTORY_FORMAT, aw, ah)])
    }

    /// What the scene is drawn at (the depth target must match).
    pub fn render_size(&self) -> (u32, u32) {
        self.rsize
    }

    fn upsampling(&self) -> bool {
        self.rsize != self.size
    }

    /// This frame's jitter in render pixels (Halton 2,3; 8 positions, 16
    /// when upsampling: each screen pixel needs more of them to fill).
    pub fn jitter(&self) -> (f32, f32) {
        let n = if self.upsampling() { 16 } else { 8 };
        let i = self.frame % n + 1;
        (halton(i, 2) - 0.5, halton(i, 3) - 0.5)
    }

    /// The screen-size colour history.
    fn textures(device: &wgpu::Device, w: u32, h: u32) -> [wgpu::TextureView; 2] {
        [Self::mk(device, "taa history 0", HISTORY_FORMAT, w, h), Self::mk(device, "taa history 1", HISTORY_FORMAT, w, h)]
    }

    pub fn resize(&mut self, device: &wgpu::Device, w: u32, h: u32, rw: u32, rh: u32) {
        if (w, h) == self.size && (rw, rh) == self.rsize {
            return;
        }
        let hist = Self::textures(device, w, h);
        let (cur_tex, cur, ao_hist, fog_hist) = Self::render_textures(device, self.format, rw, rh);
        self.cur = cur;
        self.cur_tex = cur_tex;
        self.hist = hist;
        self.ao_hist = ao_hist;
        self.fog_hist = fog_hist;
        self.size = (w, h);
        self.rsize = (rw, rh);
        self.prev_vp = None;
    }

    /// Forget the history (camera cut, teleport).
    pub fn reset(&mut self) {
        self.prev_vp = None;
    }

    /// `vp` with this frame's sub-pixel jitter (Halton 2,3, 8 positions).
    pub fn jittered(&self, vp: Mat4) -> Mat4 {
        if !self.on {
            return vp;
        }
        let (jx, jy) = self.jitter();
        let (w, h) = (self.rsize.0.max(1) as f32, self.rsize.1.max(1) as f32);
        // shift clip x/y by (2j/size) * w: a translation in NDC
        let t = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(2.0 * jx / w, 2.0 * jy / h, 0.0, 1.0),
        );
        t * vp
    }

    /// One moving rigid object (the driven car) as a box: `m` maps the
    /// unit-free box space (centre origin, axes = the car's) to world, `half`
    /// its half extents. Pixels inside it reproject with the car's own
    /// motion instead of as static scenery, which is what smeared it.
    /// The driven car's wheel centres and radius: a spinning rim has no
    /// motion vector, its history is a rim at another angle, and the clip
    /// box let it fade into the tyre at speed - these pixels take no history.
    pub fn set_wheels(&mut self, w: Option<([glam::Vec3; 4], f32)>) {
        self.wheels = w.map_or([[0.0; 4]; 4], |(c, r)| c.map(|c| [c.x, c.y, c.z, r]));
    }

    pub fn set_object(&mut self, m: Option<Mat4>, half: glam::Vec3) {
        if m.is_none() {
            self.obj_prev = None;
        }
        self.obj = m.map(|m| (m, half));
    }

    /// Where on screen (uv) the driven car's tests can hit: its box grown
    /// as taa.wgsl grows it for the road band, plus its shadow (the box swept 8 m
    /// away from the sun), and a few pixels for the halo check. The whole
    /// screen when part of it is behind the eye.
    fn obj_rect(&self, vp: Mat4) -> [f32; 4] {
        let Some((m, half)) = self.obj else { return [0.0, 0.0, 1.0, 1.0] };
        let e = half + glam::Vec3::new(0.8, 0.5, 1.2);
        let sun = self.vol.map_or(glam::Vec3::ZERO, |v| v.sun_dir.normalize_or_zero() * 8.0);
        let (mut lo, mut hi) = (glam::Vec2::splat(f32::MAX), glam::Vec2::splat(f32::MIN));
        for k in 0..16 {
            let l = glam::Vec3::new(if k & 1 == 0 { -e.x } else { e.x }, if k & 2 == 0 { -e.y } else { e.y },
                                    if k & 4 == 0 { -e.z } else { e.z });
            let w = m.transform_point3(l) - if k & 8 != 0 { sun } else { glam::Vec3::ZERO };
            let c = vp * w.extend(1.0);
            if c.w <= 0.05 {
                return [0.0, 0.0, 1.0, 1.0];
            }
            let uv = glam::Vec2::new(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
            lo = lo.min(uv);
            hi = hi.max(uv);
        }
        let px = glam::Vec2::new(4.0 / self.size.0.max(1) as f32, 4.0 / self.size.1.max(1) as f32);
        [lo.x - px.x, lo.y - px.y, hi.x + px.x, hi.y + px.y]
    }

    /// This frame's sun and shadow cascades for the volumetric fog (None: off).
    pub fn set_volumetrics(&mut self, v: Option<Volumetrics>) {
        self.vol = v;
    }

    /// The camera position for this frame's occlusion pass.
    pub fn set_eye(&mut self, eye: glam::Vec3) {
        self.eye = eye;
    }

    /// Resolve: `vp` is this frame's UNjittered view-projection.
    pub fn run(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder,
               depth: &wgpu::TextureView, out: &wgpu::TextureView, vp: Mat4,
               shadow: (&wgpu::TextureView, &wgpu::Sampler), mv: Option<&wgpu::TextureView>,
               ext: Option<&wgpu::TextureView>) {
        let (w, h) = self.size;
        let (rw, rh) = self.rsize;
        let (jx, jy) = self.jitter();
        let u = U {
            rsize: [rw as f32, rh as f32, 1.0 / rw.max(1) as f32, 1.0 / rh.max(1) as f32],
            // the scene moved by (+jx, -jy) render pixels (jittered)
            // w: 1 = reproject through the motion vectors
            jit: [jx, -jy, if self.upsampling() { 1.0 } else { 0.0 }, if mv.is_some() { 1.0 } else { 0.0 }],
            // x: the current frame's weight on sky pixels (EVOVIEW_TAA_SKY)
            sky: [std::env::var("EVOVIEW_TAA_SKY").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05f32), 0.0, 0.0, 0.0],
            obj_rect: self.obj_rect(vp),
            inv_vp: vp.inverse().to_cols_array_2d(),
            prev_vp: self.prev_vp.unwrap_or(vp).to_cols_array_2d(),
            // z: depth-based history rejection (EVOVIEW_TAA_REJECT=0 off)
            ctl: [if self.prev_vp.is_none() { 1.0 } else { 0.0 }, 0.1,
                  if std::env::var("EVOVIEW_TAA_REJECT").map_or(true, |v| v != "0") { 1.0 } else { 0.0 },
                  // w: the relative depth jump that counts as disocclusion
                  // (taa.wgsl: 10 % with motion vectors, 3 % on the old box)
                  std::env::var("EVOVIEW_TAA_TOL").ok().and_then(|v| v.parse().ok())
                      .unwrap_or(if mv.is_some() { 0.10f32 } else { 0.03 })],
            size: [w as f32, h as f32, 1.0 / w.max(1) as f32, 1.0 / h.max(1) as f32],
            obj_inv: self.obj.map_or(Mat4::IDENTITY, |o| o.0.inverse()).to_cols_array_2d(),
            obj_prev: self.obj.map_or(Mat4::IDENTITY, |o| self.obj_prev.unwrap_or(o.0)).to_cols_array_2d(),
            obj_half: self.obj.map_or([0.0; 4], |o| [o.1.x, o.1.y, o.1.z, 1.0]),
            vp: vp.to_cols_array_2d(),
            eye: [self.eye.x, self.eye.y, self.eye.z, 1.0],
            lvp_near: self.vol.map_or(Mat4::IDENTITY, |v| v.lvp_near).to_cols_array_2d(),
            lvp_far: self.vol.map_or(Mat4::IDENTITY, |v| v.lvp_far).to_cols_array_2d(),
            lvp_mid: self.vol.and_then(|v| v.lvp_mid).unwrap_or(Mat4::IDENTITY).to_cols_array_2d(),
            sbias: self.vol.map_or([0.0; 4], |v| [v.bias[0], v.bias[1], v.bias[2],
                                                  if v.lvp_mid.is_some() { 1.0 } else { 0.0 }]),
            snofs: self.vol.map_or([0.0; 4], |v| [v.nofs[0], v.nofs[1], v.nofs[2], 0.0]),
            sun: self.vol.map_or([0.0, 1.0, 0.0, 0.0], |v| [v.sun_dir.x, v.sun_dir.y, v.sun_dir.z, v.sun_strength]),
            sun_col: self.vol.map_or([0.0; 4], |v| [v.sun_col[0], v.sun_col[1], v.sun_col[2],
                                                    if self.fog_density > 0.0 { 1.0 } else { 0.0 }]),
            // the game's summer_sunny volumetricfogsettings: forward lobe
            // g 0.8 (phase_directional_g), a sun boost with exponent 25
            fog: [self.fog_density, 0.8, 1.0, 450.0],
            amb: self.vol.map_or([0.0; 4], |v| [v.amb[0], v.amb[1], v.amb[2], if v.far { 1.0 } else { 0.0 }]),
            wheels: self.wheels,
            ao: [std::env::var("EVOVIEW_AO_RADIUS").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5f32), self.ao_strength, (self.frame % 64) as f32, if self.ao_strength > 0.0 { 1.0 } else { 0.0 }],
        };
        queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&u));
        let read = self.idx;
        let write = 1 - self.idx;
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taa bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.cur) },
                // an external upscaler's output in the history's place
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(ext.unwrap_or(&self.hist[read])) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.samp) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.ao_hist[read]) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(shadow.0) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(shadow.1) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.fog_hist[read]) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.ao_hist[write]) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&self.fog_hist[write]) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(mv.unwrap_or(&self.no_mv)) },
            ],
        });
        {
            let abg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("taa aofog bg"),
                layout: &self.aofog_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.ao_hist[read]) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.fog_hist[read]) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(shadow.0) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(shadow.1) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(mv.unwrap_or(&self.no_mv)) },
                ],
            });
            let att = |v| Some(wgpu::RenderPassColorAttachment {
                view: v,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
            });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("taa aofog"),
                color_attachments: &[att(&self.ao_hist[write]), att(&self.fog_hist[write])],
                ..Default::default()
            });
            pass.set_pipeline(&self.aofog_pipe);
            pass.set_bind_group(0, &abg, &[]);
            pass.draw(0..3, 0..1);
        }
        crate::bench::mark(enc, "aofog");
        {
            let att = |v| Some(wgpu::RenderPassColorAttachment {
                view: v,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            });
            let atts = [att(out), att(&self.hist[write])];
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("taa"),
                color_attachments: if ext.is_some() { &atts[..1] } else { &atts },
                ..Default::default()
            });
            pass.set_pipeline(if ext.is_some() { &self.ext_pipe } else { &self.pipe });
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        self.idx = write;
        self.prev_vp = Some(vp);
        self.obj_prev = self.obj.map(|o| o.0);
        self.frame = self.frame.wrapping_add(1);
    }
}
