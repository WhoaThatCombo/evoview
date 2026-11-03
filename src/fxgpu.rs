//! GPU side of the effects: particles, the emissive glow (bloom) and the
//! dash screens. Shared by the window (render.rs) and headless shots
//! (shot.rs / check.rs) so both draw the same thing.
//!
//! Bloom: the game's veil (SPEC 1.4) blurs the whole HDR frame above a
//! threshold. evoview tonemaps in the material shader (no HDR frame), so the
//! glow is rebuilt from what actually glows: every emissive surface is drawn
//! a second time with `fs_glow` into an HDR buffer (main depth, LessEqual),
//! plus the flame particles, then downsampled, blurred at 1/4 and 1/8 and
//! added back. Off unless asked for (EVOVIEW_BLOOM / the N key), so stills
//! and lookcmp are unchanged.

use crate::asset::Car;
use crate::dash::{Draw, Image, Page};
use crate::fxstate::FxState;
use crate::material::Materials;
use crate::particles::{Instance, Sim};
use crate::render::{GpuMesh, Pipelines};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

pub const GLOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DASH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Flames: Emissive_Intensity (0.05..0.08) * basecolour * texture in the
/// game's units; scaled to ours by eye.
const FLAME_GAIN: f32 = 40.0;
/// Smoke / dust are lit by the studio's ambient; their basecolour * this.
const SMOKE_LIGHT: f32 = 0.55;
// 5 cars smoking in a replay or donut overran 4096: the cars uploaded last had no smoke
const MAX_PARTICLES: usize = 32768;

/// A dash screen's render target (bound as the display material's t_emis),
/// with a full mip chain: a single level read at ~1/4 size in the cockpit
/// sparkled on every glyph edge as the head moved. The page draws level 0;
/// the rest are box-filtered from it after each redraw (`mip_page`).
pub fn dash_target(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Texture {
    let (w, h) = (w.clamp(16, 2048), h.clamp(16, 2048));
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dash"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: if std::env::var("EVOVIEW_DASH_MIPS").map_or(true, |v| v != "0") { 32 - w.max(h).leading_zeros() } else { 1 },
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DASH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Read an RGBA8 texture back (dash dumps, --check).
pub fn read_rgba(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture) -> (u32, u32, Vec<u8>) {
    let sz = tex.size();
    let (w, h) = (sz.width, sz.height);
    let padded = (w * 4 + 255) / 256 * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::ImageCopyBuffer { buffer: &buf, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) } },
        sz,
    );
    queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::Maintain::Wait);
    let _ = rx.recv();
    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h {
        let s = (row * padded) as usize;
        out.extend_from_slice(&data[s..s + (w * 4) as usize]);
    }
    drop(data);
    buf.unmap();
    (w, h, out)
}

fn module(device: &wgpu::Device, label: &str, part: &str, samples: u32) -> wgpu::ShaderModule {
    let src = include_str!("fx.wgsl").replace("SCENE_DEPTH_TEX",
        if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" });
    let src = src.as_str();
    let b = src.find("// ---- bloom").unwrap();
    let d = src.find("// ---- dash").unwrap();
    let text = match part {
        "particle" => src[..b].to_string(),
        "bloom" => src[b..d].to_string(),
        _ => src[d..].to_string(),
    };
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(text.into()),
    })
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn samp_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn buf_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn solid_tex(device: &wgpu::Device, queue: &wgpu::Queue, rgba: [u8; 4]) -> wgpu::TextureView {
    rgba_tex(device, queue, 1, 1, &rgba, true)
}

fn rgba_tex(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, px: &[u8], srgb: bool) -> wgpu::TextureView {
    let t = device.create_texture_with_data(queue, &wgpu::TextureDescriptor {
        label: Some("fx rgba"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm },
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }, wgpu::util::TextureDataOrder::LayerMajor, px);
    t.create_view(&Default::default())
}

// ------------------------------------------------------------------- glow --

struct GlowTargets {
    w: u32,
    h: u32,
    msaa: Option<wgpu::TextureView>,
    full: wgpu::TextureView,
    /// 1/2, 1/4 (a, b), 1/8 (a, b)
    half: wgpu::TextureView,
    q_a: wgpu::TextureView,
    q_b: wgpu::TextureView,
    e_a: wgpu::TextureView,
    e_b: wgpu::TextureView,
}

fn hdr(device: &wgpu::Device, w: u32, h: u32, samples: u32) -> wgpu::TextureView {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("glow"),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format: GLOW_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }).create_view(&Default::default())
}

/// The half-resolution particle buffer: colour + transmittance.
const HALF_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

struct BloomStep {
    bg: wgpu::BindGroup,
    ubuf: wgpu::Buffer,
}

// -------------------------------------------------------------- particles --

struct PartDef {
    bg: wgpu::BindGroup,
    additive: bool,
}

// ------------------------------------------------------------------- dash --

#[derive(Clone, Copy)]
struct GlyphSlot {
    uv: [f32; 4],
    /// px offset of the bitmap from the pen position (x right, y DOWN from
    /// the baseline) and its size
    off: [f32; 2],
    size: [f32; 2],
    advance: f32,
}

struct Atlas {
    view: wgpu::TextureView,
    tex: wgpu::Texture,
    dim: u32,
    x: u32,
    y: u32,
    row_h: u32,
    slots: HashMap<(usize, usize, u32, char), GlyphSlot>,
}

struct DashPage {
    /// index into CarFx::dash
    page: usize,
    target: wgpu::TextureView,
    w: u32,
    h: u32,
    images: Vec<Option<wgpu::BindGroup>>,
    atlas: Atlas,
    atlas_bg: wgpu::BindGroup,
    last: String,
    /// per mip level past 0: (its render view, a bind group reading the level above)
    mips: Vec<(wgpu::TextureView, wgpu::BindGroup)>,
}

pub struct FxGpu {
    samples: u32,
    // glow
    glow_pipe_part: wgpu::RenderPipeline,
    glow: Option<GlowTargets>,
    steps: Vec<BloomStep>,
    bloom_bgl: wgpu::BindGroupLayout,
    bloom_down: wgpu::RenderPipeline,
    bloom_blur: wgpu::RenderPipeline,
    bloom_comp: wgpu::RenderPipeline,
    bloom_samp: wgpu::Sampler,
    out_format: wgpu::TextureFormat,
    // particles
    part_add: wgpu::RenderPipeline,
    part_alpha: wgpu::RenderPipeline,
    soft_add: wgpu::RenderPipeline,
    soft_alpha: wgpu::RenderPipeline,
    /// half resolution (draw_particles_half): into HALF_FORMAT with the
    /// transmittance in alpha, then one composite
    half_add: wgpu::RenderPipeline,
    half_alpha: wgpu::RenderPipeline,
    part_comp: wgpu::RenderPipeline,
    half: Option<(u32, u32, wgpu::TextureView, BloomStep)>,
    pdepth_bgl: wgpu::BindGroupLayout,
    /// 1 x 1 stand-in for the refraction scene (depth_group)
    no_scene: wgpu::TextureView,
    part_cam: wgpu::BindGroup,
    /// the same camera without TAA's jitter, for particles drawn after the
    /// resolve (set_late_camera): with the jittered one they wobbled +-0.5 px
    late_buf: wgpu::Buffer,
    late_cam: wgpu::BindGroup,
    part_defs: Vec<PartDef>,
    quad: wgpu::Buffer,
    inst: wgpu::Buffer,
    /// (def, first instance, count) for this frame
    batches: Vec<(usize, u32, u32)>,
    pub sim: Sim,
    // dash
    dash_pipe: wgpu::RenderPipeline,
    dash_bgl: wgpu::BindGroupLayout,
    dash_samp: wgpu::Sampler,
    white_bg: wgpu::BindGroup,
    dash: Vec<DashPage>,
    dash_t: f32,
    /// a white full-target quad (vs_dash's vertices) and its unblended
    /// pipeline, for the mip levels
    dash_quad: wgpu::Buffer,
    dash_mip_pipe: wgpu::RenderPipeline,
}

impl FxGpu {
    #[allow(clippy::too_many_arguments)]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, car: &Car, mats: &Materials,
               out_format: wgpu::TextureFormat, samples: u32, uniform_buf: &wgpu::Buffer) -> Self {
        let pm = module(device, "fx particles", "particle", samples);
        let bm = module(device, "fx bloom", "bloom", samples);
        let dm = module(device, "fx dash", "dash", samples);
        let lin = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fx linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        // ---- particles ----
        let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx cam bgl"),
            entries: &[
                buf_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                samp_entry(2),
            ],
        });
        let part_cam = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx cam"),
            layout: &cam_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&mats.lut_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&lin) },
            ],
        });
        let late_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx cam (unjittered)"),
            size: uniform_buf.size(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let late_cam = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx cam (unjittered)"),
            layout: &cam_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: late_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&mats.lut_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&lin) },
            ],
        });
        let pmat_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx pmat bgl"),
            entries: &[buf_entry(0, wgpu::ShaderStages::FRAGMENT), tex_entry(1), tex_entry(2), samp_entry(3), tex_entry(4)],
        });
        let playout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx particle layout"),
            bind_group_layouts: &[&cam_bgl, &pmat_bgl],
            push_constant_ranges: &[],
        });
        let vbufs = [
            wgpu::VertexBufferLayout {
                array_stride: 8,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Instance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4],
            },
        ];
        let add = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
        };
        let mk_part = |label: &str, entry: &str, fmt: wgpu::TextureFormat, blend: wgpu::BlendState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&playout),
                vertex: wgpu::VertexState { module: &pm, entry_point: "vs_particle", compilation_options: Default::default(), buffers: &vbufs },
                fragment: Some(wgpu::FragmentState {
                    module: &pm, entry_point: entry, compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: fmt, blend: Some(blend), write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
                multiview: None,
                cache: None,
            })
        };
        let part_add = mk_part("particles add", "fs_particle", out_format, add);
        let part_alpha = mk_part("particles alpha", "fs_particle", out_format, wgpu::BlendState::ALPHA_BLENDING);
        // Soft particles (live view): their own pass after the main one,
        // reading the scene depth - a sprite fades out within 0.8 m of what
        // is behind it instead of slicing into the road in hard straight
        // edges, and hides behind the cars by the same test.
        let pdepth_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx particle depth bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: samples > 1,
                },
                count: None,
            },
            // the scene behind the sprites, for water to refract (the
            // half-res pass only; a 1 x 1 stand-in elsewhere)
            tex_entry(1)],
        });
        let soft_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx soft particle layout"),
            bind_group_layouts: &[&cam_bgl, &pmat_bgl, &pdepth_bgl],
            push_constant_ranges: &[],
        });
        let mk_soft = |label: &str, blend: wgpu::BlendState, entry: &str, fmt: wgpu::TextureFormat, count: u32| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&soft_layout),
                vertex: wgpu::VertexState { module: &pm, entry_point: "vs_particle", compilation_options: Default::default(), buffers: &vbufs },
                fragment: Some(wgpu::FragmentState {
                    module: &pm, entry_point: entry, compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: fmt, blend: Some(blend), write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
                depth_stencil: None,
                multisample: wgpu::MultisampleState { count, ..Default::default() },
                multiview: None,
                cache: None,
            })
        };
        let soft_add = mk_soft("particles add (soft)", add, "fs_particle_soft", out_format, samples);
        let soft_alpha = mk_soft("particles alpha (soft)", wgpu::BlendState::ALPHA_BLENDING, "fs_particle_soft", out_format, samples);
        // half resolution: colour as above, alpha = what is still seen
        // through (cleared to 1, x (1 - a) per sprite; additive keeps it)
        let through = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
        };
        let half_add = mk_soft("particles add (half)", add, "fs_particle_soft_half", HALF_FORMAT, 1);
        let half_alpha = mk_soft("particles alpha (half)", through, "fs_particle_soft_half", HALF_FORMAT, 1);
        let glow_pipe_part = mk_part("particles glow", "fs_particle_glow", GLOW_FORMAT, add);
        let quad = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fx quad"),
            contents: bytemuck::cast_slice(&[-1.0f32, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let inst = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx particles"),
            size: (MAX_PARTICLES * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let white = solid_tex(device, queue, [255, 255, 255, 255]);
        let flat = solid_tex(device, queue, [128, 128, 255, 255]);
        let no_scene = solid_tex(device, queue, [0, 0, 0, 255]);
        // clamped, as the game's samLinearClamp: a flipbook cell must not
        // wrap into the sheet's far side
        let wrap = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fx clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mut part_defs = Vec::new();
        if let Some(fx) = car.fx.as_deref() {
            for d in &fx.emitter_defs {
                let m = &d.material;
                if std::env::var_os("EVOVIEW_FX_DEFS").is_some() {
                    eprintln!("FXDEF {} fade {} mat {} flip {} {}x{} col {:?} op {:?} op_alpha {} additive {}", d.path, m.depth_fade, m.file, m.flipbook, m.cols, m.rows,
                        m.color_tex.map(|i| (car.textures[i].name.clone(), car.textures[i].width, car.textures[i].height, car.textures[i].format.clone())),
                        m.opacity_tex.map(|i| (car.textures[i].name.clone(), car.textures[i].width, car.textures[i].height, car.textures[i].format.clone())),
                        m.opacity_alpha, m.additive);
                }
                let col = m.color_tex.and_then(|i| mats.textures.get(device, queue, car, i, true));
                let op = m.opacity_tex.and_then(|i| mats.textures.get(device, queue, car, i, false));
                let additive = m.additive;
                // Water (spray, splash): its Refraction_Transmission share is
                // the scene behind seen through it (fx.wgsl), the rest lit in
                // the material's own colour - the spray's is a dark blue, so
                // a grey-white stand-in here read as smoke
                let w = m.water;
                let bc = m.basecolor;
                // EVOVIEW_SPRAY_GAME=0: the scene seen through untinted plus a
                // brighter mist (before the game's particle shader was read)
                // EVOVIEW_SPRAY=k: how thick the tyres' spray is drawn (alpha
                // x k); the body's long haze (4-10 m sprites) keeps its own
                let spray_k = if w > 0.0 && d.path.to_lowercase().contains("tyre") {
                    std::env::var("EVOVIEW_SPRAY").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(2.5).max(0.0)
                } else { 1.0 };
                let nrm = m.normal_tex.and_then(|i| mats.textures.get(device, queue, car, i, false));
                let p: [f32; 20] = [
                    bc[0], bc[1], bc[2], m.emissive,
                    m.cols, m.rows, if m.flipbook { 1.0 } else { 0.0 }, if additive { 1.0 } else { 0.0 },
                    if col.is_some() { 1.0 } else { 0.0 }, if op.is_some() { 1.0 } else { 0.0 },
                    if additive { FLAME_GAIN } else if m.refracting && w <= 0.0 { SMOKE_LIGHT * 0.25 } else { SMOKE_LIGHT },
                    if m.opacity_alpha { 1.0 } else { 0.0 },
                    if m.depth_fade > 0.0 { m.depth_fade } else { 0.8 }, w, m.distortion,
                    if nrm.is_some() { 1.0 } else { 0.0 },
                    m.refr_rough, if w > 0.0 { m.amb_int.max(1.0) } else { 1.0 }, spray_k, if std::env::var("EVOVIEW_SPRAY_GAME").map_or(true, |v| v != "0") { 1.0 } else { 0.0 },
                ];
                let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pmat"), contents: bytemuck::cast_slice(&p), usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pmat"),
                    layout: &pmat_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(col.as_deref().unwrap_or(&white)) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(op.as_deref().unwrap_or(&white)) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&wrap) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(nrm.as_deref().unwrap_or(&flat)) },
                    ],
                });
                part_defs.push(PartDef { bg, additive });
            }
        }
        // ---- bloom ----
        let bloom_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx bloom bgl"),
            entries: &[tex_entry(0), samp_entry(1), buf_entry(2, wgpu::ShaderStages::FRAGMENT)],
        });
        let blayout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx bloom layout"),
            bind_group_layouts: &[&bloom_bgl],
            push_constant_ranges: &[],
        });
        let mk_full = |label: &str, entry: &str, fmt: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&blayout),
                vertex: wgpu::VertexState { module: &bm, entry_point: "vs_full", compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState {
                    module: &bm, entry_point: entry, compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: fmt, blend, write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };
        let bloom_down = mk_full("bloom down", "fs_down", GLOW_FORMAT, None);
        let bloom_blur = mk_full("bloom blur", "fs_blur", GLOW_FORMAT, None);
        let bloom_comp = mk_full("bloom composite", "fs_composite", out_format, Some(add));
        // half-res particles over the scene: rgb + scene x transmittance
        let part_comp = mk_full("particles composite", "fs_part_comp", out_format, Some(wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::SrcAlpha, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
        }));
        // ---- dash ----
        let dash_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx dash bgl"),
            entries: &[tex_entry(0), samp_entry(1)],
        });
        let dlayout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx dash layout"),
            bind_group_layouts: &[&dash_bgl],
            push_constant_ranges: &[],
        });
        let mk_dash = |blend: Option<wgpu::BlendState>| device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dash"),
            layout: Some(&dlayout),
            vertex: wgpu::VertexState {
                module: &dm, entry_point: "vs_dash", compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 32,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &dm, entry_point: "fs_dash", compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: DASH_FORMAT, blend, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let dash_pipe = mk_dash(Some(wgpu::BlendState::ALPHA_BLENDING));
        // the mip levels: a straight copy of the filtered level above
        let dash_mip_pipe = mk_dash(None);
        let dash_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("dash samp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let mk_bg = |v: &wgpu::TextureView| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dash img"),
            layout: &dash_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&dash_samp) },
            ],
        });
        let white_bg = mk_bg(&white);
        let mut dash = Vec::new();
        if let Some(fx) = car.fx.as_deref() {
            for (pi, t) in &mats.dash_targets {
                let Some(pg) = fx.dash.get(*pi) else { continue };
                let images = pg.images.iter().map(|im| match im {
                    Image::Tex(i, _, _) => mats.textures.get(device, queue, car, *i, true).map(|v| mk_bg(&v)),
                    Image::Rgba(w, h, px) => Some(mk_bg(&rgba_tex(device, queue, *w, *h, px, true))),
                }).collect();
                let dim = 1024;
                let at = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("glyphs"),
                    size: wgpu::Extent3d { width: dim, height: dim, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let av = at.create_view(&Default::default());
                let atlas_bg = mk_bg(&av);
                let size = t.size();
                let level = |l: u32| t.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: l,
                    mip_level_count: Some(1),
                    ..Default::default()
                });
                let mips = (1..t.mip_level_count()).map(|l| (level(l), mk_bg(&level(l - 1)))).collect();
                dash.push(DashPage {
                    page: *pi,
                    mips,
                    target: level(0),
                    w: size.width,
                    h: size.height,
                    images,
                    atlas: Atlas { view: av, tex: at, dim, x: 1, y: 1, row_h: 0, slots: HashMap::new() },
                    atlas_bg,
                    last: String::new(),
                });
            }
        }
        FxGpu {
            samples, glow_pipe_part, glow: None, steps: Vec::new(), bloom_bgl, bloom_down, bloom_blur, bloom_comp,
            bloom_samp: lin, out_format, part_add, part_alpha, soft_add, soft_alpha, half_add, half_alpha, part_comp, half: None, pdepth_bgl, no_scene, part_cam, late_buf, late_cam, part_defs, quad, inst,
            batches: Vec::new(), sim: Sim::default(), dash_pipe, dash_bgl, dash_samp, white_bg,
            dash, dash_t: 1e9, dash_mip_pipe,
            dash_quad: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dash mip quad"),
                contents: bytemuck::cast_slice(&[
                    -1.0f32, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,      -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,      -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                ]),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        }
    }

    pub fn dash_pages(&self) -> usize {
        self.dash.len()
    }

    // ------------------------------------------------------------ update --

    /// Advance particles, refresh the dash (every frame its content changed:
    /// a 10 Hz refresh made rev bars and speed step like a slideshow).
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder,
                  fx: Option<&crate::fxdata::CarFx>, st: &FxState, dt: f32, eye: [f32; 3]) {
        let Some(fx) = fx else { return };
        self.sim.step(fx, st, dt);
        self.upload_particles(queue, fx, eye);
        self.dash_t += dt;
        self.render_dash(device, queue, enc, fx, st, false);
    }

    /// Headless sequence frame: step the particles by `dt`, redraw the dash.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder,
                         fx: Option<&crate::fxdata::CarFx>, st: &FxState, dt: f32, eye: [f32; 3]) {
        let Some(fx) = fx else { return };
        self.sim.step(fx, st, dt);
        self.upload_particles(queue, fx, eye);
        self.render_dash(device, queue, enc, fx, st, true);
    }

    /// Headless: simulate particles up to the shot and draw the dash once.
    pub fn prepare_still(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder,
                         car: &Car, st: &FxState, eye: [f32; 3]) {
        let Some(fx) = car.fx.as_deref() else { return };
        self.sim = Sim::default();
        let warm = if st.backfire_age >= 0.0 { st.backfire_age + 0.02 } else if st.smoke > 0.0 || st.dust > 0.0 { 2.5 } else { 0.0 };
        if warm > 0.0 {
            self.sim.warm(fx, st, warm);
        }
        self.upload_particles(queue, fx, eye);
        self.render_dash(device, queue, enc, fx, st, true);
    }

    fn upload_particles(&mut self, queue: &wgpu::Queue, fx: &crate::fxdata::CarFx, eye: [f32; 3]) {
        self.batches.clear();
        let mut all: Vec<Instance> = Vec::new();
        for di in 0..fx.emitter_defs.len() {
            let v = self.sim.instances(fx, di, eye);
            if v.is_empty() {
                continue;
            }
            let room = MAX_PARTICLES - all.len();
            let n = v.len().min(room);
            if n == 0 {
                break;
            }
            self.batches.push((di, all.len() as u32, n as u32));
            all.extend_from_slice(&v[..n]);
        }
        if !all.is_empty() {
            queue.write_buffer(&self.inst, 0, bytemuck::cast_slice(&all));
        }
    }

    pub fn particle_count(&self) -> usize {
        self.batches.iter().map(|b| b.2 as usize).sum()
    }

    pub fn has_particles(&self) -> bool {
        !self.batches.is_empty()
    }

    /// The scene depth for `draw_particles_soft` (after the main pass), and
    /// the scene colour water refracts - only where the particles are not
    /// drawn into that same texture (the half-res pass).
    pub fn depth_group(&self, device: &wgpu::Device, depth: &wgpu::TextureView, scene: Option<&wgpu::TextureView>) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx particle depth"),
            layout: &self.pdepth_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(scene.unwrap_or(&self.no_scene)) },
            ],
        })
    }

    /// The camera uniform (render.rs Uniforms) with the unjittered
    /// view-projection, for the passes after TAA (`late`).
    pub fn set_late_camera(&self, queue: &wgpu::Queue, cam: &[u8]) {
        queue.write_buffer(&self.late_buf, 0, &cam[..cam.len().min(self.late_buf.size() as usize)]);
    }

    /// In a pass of its own after the main one (no depth attachment): soft
    /// particles test and fade against `depth` themselves. `late`: after
    /// TAA's resolve, through the unjittered camera.
    pub fn draw_particles_soft<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, depth: &'a wgpu::BindGroup, late: bool) {
        if self.batches.is_empty() {
            return;
        }
        pass.set_bind_group(0, if late { &self.late_cam } else { &self.part_cam }, &[]);
        pass.set_bind_group(2, depth, &[]);
        pass.set_vertex_buffer(0, self.quad.slice(..));
        pass.set_vertex_buffer(1, self.inst.slice(..));
        for &(di, first, n) in &self.batches {
            let Some(d) = self.part_defs.get(di) else { continue };
            pass.set_pipeline(if d.additive { &self.soft_add } else { &self.soft_alpha });
            pass.set_bind_group(1, &d.bg, &[]);
            pass.draw(0..4, first..first + n);
        }
    }

    /// `draw_particles_soft` at half the resolution, composited onto `out`
    /// (samples 1 only). Smoke is soft, and a cloud of it filling the screen
    /// at 3440x1440 cost up to 9 ms of overdraw at full size.
    pub fn draw_particles_half(&mut self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder,
                               depth: &wgpu::BindGroup, out: &wgpu::TextureView, w: u32, h: u32) {
        if self.batches.is_empty() {
            return;
        }
        let (hw, hh) = (w.div_ceil(2).max(1), h.div_ceil(2).max(1));
        if self.half.as_ref().map_or(true, |t| t.0 != hw || t.1 != hh) {
            let view = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("particles half"),
                size: wgpu::Extent3d { width: hw, height: hh, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HALF_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }).create_view(&Default::default());
            let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("particles composite"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particles composite"),
                layout: &self.bloom_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.bloom_samp) },
                    wgpu::BindGroupEntry { binding: 2, resource: ubuf.as_entire_binding() },
                ],
            });
            self.half = Some((hw, hh, view, BloomStep { bg, ubuf }));
        }
        let (_, _, view, step) = self.half.as_ref().unwrap();
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("particles half"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_bind_group(0, &self.late_cam, &[]);
            pass.set_bind_group(2, depth, &[]);
            pass.set_vertex_buffer(0, self.quad.slice(..));
            pass.set_vertex_buffer(1, self.inst.slice(..));
            for &(di, first, n) in &self.batches {
                let Some(d) = self.part_defs.get(di) else { continue };
                pass.set_pipeline(if d.additive { &self.half_add } else { &self.half_alpha });
                pass.set_bind_group(1, &d.bg, &[]);
                pass.draw(0..4, first..first + n);
            }
        }
        let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("particles composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: out,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        p.set_pipeline(&self.part_comp);
        p.set_bind_group(0, &step.bg, &[]);
        p.draw(0..3, 0..1);
    }

    /// Inside the main pass, after the transparent phase.
    pub fn draw_particles<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.batches.is_empty() {
            return;
        }
        pass.set_bind_group(0, &self.part_cam, &[]);
        pass.set_vertex_buffer(0, self.quad.slice(..));
        pass.set_vertex_buffer(1, self.inst.slice(..));
        for &(di, first, n) in &self.batches {
            let Some(d) = self.part_defs.get(di) else { continue };
            pass.set_pipeline(if d.additive { &self.part_add } else { &self.part_alpha });
            pass.set_bind_group(1, &d.bg, &[]);
            pass.draw(0..4, first..first + n);
        }
    }

    // -------------------------------------------------------------- glow --

    fn ensure_glow(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if self.glow.as_ref().map_or(false, |g| g.w == w && g.h == h) {
            return;
        }
        self.steps.clear();
        let (w2, h2) = ((w / 2).max(1), (h / 2).max(1));
        let (w4, h4) = ((w / 4).max(1), (h / 4).max(1));
        let (w8, h8) = ((w / 8).max(1), (h / 8).max(1));
        self.glow = Some(GlowTargets {
            w, h,
            msaa: if self.samples > 1 { Some(hdr(device, w, h, self.samples)) } else { None },
            full: hdr(device, w, h, 1),
            half: hdr(device, w2, h2, 1),
            q_a: hdr(device, w4, h4, 1),
            q_b: hdr(device, w4, h4, 1),
            e_a: hdr(device, w8, h8, 1),
            e_b: hdr(device, w8, h8, 1),
        });
    }

    /// The bloom chain's bind groups, built once per target size.
    fn make_steps(&self, device: &wgpu::Device, w: u32, h: u32) -> Vec<BloomStep> {
        let g = self.glow.as_ref().unwrap();
        let px = |w: u32, h: u32, dx: f32, dy: f32| [1.0 / w.max(1) as f32, 1.0 / h.max(1) as f32, dx, dy];
        let (w2, h2) = ((w / 2).max(1), (h / 2).max(1));
        let (w4, h4) = ((w / 4).max(1), (h / 4).max(1));
        let (w8, h8) = ((w / 8).max(1), (h / 8).max(1));
        let k0 = [0.0; 4];
        vec![
            self.step_bg(device, &g.full, px(w, h, 0.0, 0.0), k0),
            self.step_bg(device, &g.half, px(w2, h2, 0.0, 0.0), k0),
            self.step_bg(device, &g.q_a, px(w4, h4, 1.0, 0.0), k0),
            self.step_bg(device, &g.q_b, px(w4, h4, 0.0, 1.0), k0),
            self.step_bg(device, &g.q_a, px(w4, h4, 0.0, 0.0), k0),
            self.step_bg(device, &g.e_a, px(w8, h8, 1.0, 0.0), k0),
            self.step_bg(device, &g.e_b, px(w8, h8, 0.0, 1.0), k0),
            self.step_bg(device, &g.q_a, [0.0; 4], k0),
            self.step_bg(device, &g.e_a, [0.0; 4], k0),
        ]
    }

    fn step_bg(&self, device: &wgpu::Device, src: &wgpu::TextureView, px: [f32; 4], k: [f32; 4]) -> BloomStep {
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bloom p"),
            contents: bytemuck::cast_slice(&[px, k]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom"),
            layout: &self.bloom_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.bloom_samp) },
                wgpu::BindGroupEntry { binding: 2, resource: ubuf.as_entire_binding() },
            ],
        });
        BloomStep { bg, ubuf }
    }

    fn full_pass(&self, enc: &mut wgpu::CommandEncoder, pipe: &wgpu::RenderPipeline, dst: &wgpu::TextureView,
                 step: &BloomStep, clear: bool) {
        let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bloom"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if clear { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load },
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        p.set_pipeline(pipe);
        p.set_bind_group(0, &step.bg, &[]);
        p.draw(0..3, 0..1);
    }

    /// Glow pass + bloom, composited onto `out` (the resolved, tonemapped
    /// frame). `depth` is the main pass's depth (same sample count).
    #[allow(clippy::too_many_arguments)]
    pub fn bloom(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, strength: f32,
                 meshes: &[GpuMesh], mats: &Materials, pipes: &Pipelines,
                 depth: &wgpu::TextureView, out: &wgpu::TextureView, w: u32, h: u32) {
        if strength <= 0.0 {
            return;
        }
        self.ensure_glow(device, w, h);
        let g = self.glow.as_ref().unwrap();
        {
            let (view, resolve) = match &g.msaa {
                Some(m) => (m, Some(&g.full)),
                None => (&g.full, None),
            };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("glow"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: resolve,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&pipes.glow);
            pass.set_bind_group(0, &pipes.bind_group, &[]);
            pass.set_bind_group(2, &pipes.bone_bg, &[]);
            for m in meshes {
                let mut bound = false;
                for &(start, count, mi) in &m.draws {
                    let emissive = mats.light_srcs.get(mi).and_then(|s| s.as_ref())
                        .map_or(false, |s| s.class != crate::lights::Class::None);
                    if !emissive {
                        continue;
                    }
                    if !bound {
                        bound = true;
                        pass.set_vertex_buffer(0, m.vertices.slice(..));
                        pass.set_vertex_buffer(1, m.instances.slice(..));
                        pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                    }
                    pass.set_bind_group(1, &mats.groups[mi], &[]);
                    pass.draw_indexed(start..start + count, 0, 0..m.instance_count);
                }
            }
            // flames
            if !self.batches.is_empty() {
                pass.set_pipeline(&self.glow_pipe_part);
                pass.set_bind_group(0, &self.part_cam, &[]);
                pass.set_vertex_buffer(0, self.quad.slice(..));
                pass.set_vertex_buffer(1, self.inst.slice(..));
                for &(di, first, n) in &self.batches {
                    let Some(d) = self.part_defs.get(di) else { continue };
                    if !d.additive {
                        continue;
                    }
                    pass.set_bind_group(1, &d.bg, &[]);
                    pass.draw(0..4, first..first + n);
                }
            }
        }
        if self.steps.is_empty() {
            self.steps = self.make_steps(device, w, h);
        }
        let g = self.glow.as_ref().unwrap();
        let st = &self.steps;
        queue.write_buffer(&st[7].ubuf, 0, bytemuck::cast_slice(&[[0.0f32; 4], [strength * 0.6, 0.0, 0.0, 0.0]]));
        queue.write_buffer(&st[8].ubuf, 0, bytemuck::cast_slice(&[[0.0f32; 4], [strength * 0.4, 0.0, 0.0, 0.0]]));
        self.full_pass(enc, &self.bloom_down, &g.half, &st[0], true);
        self.full_pass(enc, &self.bloom_down, &g.q_a, &st[1], true);
        self.full_pass(enc, &self.bloom_blur, &g.q_b, &st[2], true);
        self.full_pass(enc, &self.bloom_blur, &g.q_a, &st[3], true);
        self.full_pass(enc, &self.bloom_down, &g.e_a, &st[4], true);
        self.full_pass(enc, &self.bloom_blur, &g.e_b, &st[5], true);
        self.full_pass(enc, &self.bloom_blur, &g.e_a, &st[6], true);
        // composite both radii (veil intensity 0.6 in the showroom presets)
        self.full_pass(enc, &self.bloom_comp, out, &st[7], false);
        self.full_pass(enc, &self.bloom_comp, out, &st[8], false);
        let _ = self.out_format;
    }

    // -------------------------------------------------------------- dash --

    fn glyph(atlas: &mut Atlas, queue: &wgpu::Queue, font: &ab_glyph::FontRef, fi: usize, page: usize,
             px: f32, c: char) -> Option<GlyphSlot> {
        use ab_glyph::{Font, ScaleFont};
        let q = (px * 2.0).round() as u32;
        let key = (page, fi, q, c);
        if let Some(s) = atlas.slots.get(&key) {
            return Some(*s);
        }
        let scaled = font.as_scaled(ab_glyph::PxScale::from(q as f32 * 0.5));
        let id = font.glyph_id(c);
        let advance = scaled.h_advance(id);
        let glyph = id.with_scale_and_position(q as f32 * 0.5, ab_glyph::point(0.0, 0.0));
        let Some(outline) = font.outline_glyph(glyph) else {
            let s = GlyphSlot { uv: [0.0; 4], off: [0.0; 2], size: [0.0; 2], advance };
            atlas.slots.insert(key, s);
            return Some(s);
        };
        let b = outline.px_bounds();
        let (gw, gh) = (b.width().ceil() as u32 + 2, b.height().ceil() as u32 + 2);
        if gw > atlas.dim || gh > atlas.dim {
            return None;
        }
        if atlas.x + gw >= atlas.dim {
            atlas.x = 1;
            atlas.y += atlas.row_h + 1;
            atlas.row_h = 0;
        }
        if atlas.y + gh >= atlas.dim {
            return None; // full: text past here is skipped this frame
        }
        let mut px_data = vec![0u8; (gw * gh * 4) as usize];
        outline.draw(|x, y, cov| {
            let i = (((y + 1) * gw + x + 1) * 4) as usize;
            if i + 3 < px_data.len() {
                px_data[i] = 255;
                px_data[i + 1] = 255;
                px_data[i + 2] = 255;
                px_data[i + 3] = (cov.clamp(0.0, 1.0) * 255.0) as u8;
            }
        });
        queue.write_texture(
            wgpu::ImageCopyTexture { texture: &atlas.tex, mip_level: 0, origin: wgpu::Origin3d { x: atlas.x, y: atlas.y, z: 0 }, aspect: wgpu::TextureAspect::All },
            &px_data,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(gw * 4), rows_per_image: Some(gh) },
            wgpu::Extent3d { width: gw, height: gh, depth_or_array_layers: 1 },
        );
        let d = atlas.dim as f32;
        let s = GlyphSlot {
            uv: [atlas.x as f32 / d, atlas.y as f32 / d, (atlas.x + gw) as f32 / d, (atlas.y + gh) as f32 / d],
            off: [b.min.x - 1.0, b.min.y - 1.0],
            size: [gw as f32, gh as f32],
            advance,
        };
        atlas.x += gw + 1;
        atlas.row_h = atlas.row_h.max(gh);
        atlas.slots.insert(key, s);
        Some(s)
    }

    fn render_dash(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder,
                   fx: &crate::fxdata::CarFx, st: &FxState, force: bool) {
        let blink = st.blink_phase.unwrap_or(st.time.rem_euclid(1.0) < 0.5);
        let shift = fx.led_sys.as_ref().and_then(|l| l.group(crate::fxdata::LED_RPM, 0).iter()
            .map(|i| i.threshold).filter(|t| *t > 0.0).reduce(f32::max)).unwrap_or(0.0);
        let limiter = fx.led_sys.as_ref().map_or(0.0, |l| l.limiter_rpm);
        let mut targets: Vec<(usize, &Page)> = Vec::new();
        for (k, d) in self.dash.iter().enumerate() {
            if let Some(p) = fx.dash.get(d.page) {
                targets.push((k, p));
            }
        }
        for (k, page) in targets {
            let body = crate::dash::body_state(st, &page.default_page, blink, shift, limiter);
            let list = page.draw_list(st, &body);
            // the whole list: a bar that only changes width (rpm, throttle)
            // is a change too, and keying on the text alone froze it
            let key = format!("{:?}{:?}", body.classes, list);
            let dp = &mut self.dash[k];
            if !force && key == dp.last {
                continue;
            }
            dp.last = key;
            let bg = page.background(&body);
            Self::draw_page(device, queue, enc, dp, page, &list, bg, &self.dash_pipe, &self.white_bg, k);
            Self::mip_page(enc, dp, &self.dash_mip_pipe, &self.dash_quad);
        }
        let _ = (&self.dash_bgl, &self.dash_samp);
    }

    /// Each mip level from the one above: the bilinear tap at the centre of
    /// every 2x2 block is their average (in linear light, sRGB on both ends).
    fn mip_page(enc: &mut wgpu::CommandEncoder, dp: &DashPage, pipe: &wgpu::RenderPipeline, quad: &wgpu::Buffer) {
        for (view, src) in &dp.mips {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dash mip"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(pipe);
            pass.set_vertex_buffer(0, quad.slice(..));
            pass.set_bind_group(0, src, &[]);
            pass.draw(0..6, 0..1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_page(device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, dp: &mut DashPage,
                 page: &Page, list: &[Draw], bg: [f32; 4], pipe: &wgpu::RenderPipeline,
                 white_bg: &wgpu::BindGroup, pi: usize) {
        let (sx, sy) = (dp.w as f32 / page.width as f32, dp.h as f32 / page.height as f32);
        let (w, h) = (dp.w as f32, dp.h as f32);
        // batches: (bind group source, vertices)
        enum Src { White, Img(usize), Atlas }
        let mut batches: Vec<(Src, Vec<f32>)> = Vec::new();
        let mut push = |src: Src, x: f32, y: f32, qw: f32, qh: f32, uv: [f32; 4], c: [f32; 4], batches: &mut Vec<(Src, Vec<f32>)>| {
            let same = matches!((batches.last().map(|b| &b.0), &src),
                (Some(Src::White), Src::White) | (Some(Src::Atlas), Src::Atlas))
                || matches!((batches.last().map(|b| &b.0), &src), (Some(Src::Img(a)), Src::Img(b)) if a == b);
            if !same {
                batches.push((src, Vec::new()));
            }
            let v = &mut batches.last_mut().unwrap().1;
            let (x0, y0, x1, y1) = (x * sx, y * sy, (x + qw) * sx, (y + qh) * sy);
            let cx = |x: f32| x / w * 2.0 - 1.0;
            let cy = |y: f32| 1.0 - y / h * 2.0;
            for (px, py, u, vv) in [(x0, y0, uv[0], uv[1]), (x1, y0, uv[2], uv[1]), (x1, y1, uv[2], uv[3]),
                                    (x0, y0, uv[0], uv[1]), (x1, y1, uv[2], uv[3]), (x0, y1, uv[0], uv[3])] {
                v.extend_from_slice(&[cx(px), cy(py), u, vv, c[0], c[1], c[2], c[3]]);
            }
        };
        let fonts: Vec<ab_glyph::FontRef> = page.fonts.iter().filter_map(|b| ab_glyph::FontRef::try_from_slice(b).ok()).collect();
        for d in list {
            match d {
                Draw::Rect { x, y, w, h, color } => push(Src::White, *x, *y, *w, *h, [0.0, 0.0, 1.0, 1.0], *color, &mut batches),
                Draw::Image { x, y, w, h, img, uv, alpha } => {
                    if dp.images.get(*img).map_or(false, |b| b.is_some()) {
                        push(Src::Img(*img), *x, *y, *w, *h, *uv, [1.0, 1.0, 1.0, *alpha], &mut batches);
                    }
                }
                Draw::Text { x, y, w: bw, align, size, font, text, color, spacing } => {
                    let Some(f) = fonts.get(*font).or(fonts.first()) else { continue };
                    let fi = if *font < fonts.len() { *font } else { 0 };
                    // measure with the real font
                    let px = size * sx.max(sy);
                    let mut total = 0.0;
                    let mut gl = Vec::new();
                    for ch in text.chars() {
                        if let Some(g) = Self::glyph(&mut dp.atlas, queue, f, fi, pi, px, ch) {
                            total += g.advance / sx + spacing;
                            gl.push(g);
                        }
                    }
                    let mut pen = match align { 1 => x + (bw - total) * 0.5, 2 => x + bw - total, _ => *x };
                    for g in gl {
                        if g.size[0] > 0.0 {
                            push(Src::Atlas, pen + g.off[0] / sx, y + g.off[1] / sy, g.size[0] / sx, g.size[1] / sy, g.uv, *color, &mut batches);
                        }
                        pen += g.advance / sx + spacing;
                    }
                }
            }
        }
        let verts: Vec<f32> = batches.iter().flat_map(|b| b.1.iter().copied()).collect();
        let vb = if verts.is_empty() { None } else {
            Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dash verts"), contents: bytemuck::cast_slice(&verts), usage: wgpu::BufferUsages::VERTEX,
            }))
        };
        let lin = |c: f32| if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dash"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dp.target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: lin(bg[0]) as f64, g: lin(bg[1]) as f64, b: lin(bg[2]) as f64, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        let Some(vb) = vb.as_ref() else { return };
        pass.set_pipeline(pipe);
        pass.set_vertex_buffer(0, vb.slice(..));
        let mut first = 0u32;
        for (src, v) in &batches {
            let n = (v.len() / 8) as u32;
            let bgp = match src {
                Src::White => white_bg,
                Src::Img(i) => dp.images[*i].as_ref().unwrap(),
                Src::Atlas => &dp.atlas_bg,
            };
            pass.set_bind_group(0, bgp, &[]);
            pass.draw(first..first + n, 0..1);
            first += n;
        }
        let _ = &dp.atlas.view;
    }
}
