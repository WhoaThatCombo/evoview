//! Complete a texture's mip chain on the GPU.
//!
//! ⚠ ~1,200 of the package's textures stop their chain early (`--tex-audit`
//! "short-chain"): the shared car parts' logos, stitching, belts, decals and
//! number fonts ship ONLY their top level, and some track maps stop at 32 px.
//! The sampler has nothing smaller to fall back to, so wherever such a
//! texture is minified it point-samples a big level: sparkle, moire and
//! crawling in motion. The game streams and builds its own; we build them
//! here.
//!
//! The levels that exist are kept as they are (copied texel for texel), the
//! rest are 2x2 box averages of the level above, rendered into an
//! uncompressed texture with the full chain: RGBA8 (sRGB when the source is
//! colour) for the 8-bit and BC formats, RGBA16F for the HDR ones. Averaging
//! happens in linear light (an sRGB view decodes on load and encodes on
//! store). BC4 / BC5 keep their channel layout ((r, 0, 0, 1) / (r, g, 0, 1)
//! read the same either way). EVOVIEW_MIPFILL=0 off.

use std::collections::HashMap;
use std::sync::Mutex;

const WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
// x mode (0 copy, 1 2x2 average), yz the source level's size
@group(0) @binding(1) var<uniform> p: vec4<u32>;
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let d = vec2<i32>(pos.xy);
    let m = vec2<i32>(p.yz) - vec2<i32>(1);
    if (p.x == 0u) {
        return textureLoad(src, min(d, m), 0);
    }
    let s = d * 2;
    return (textureLoad(src, min(s, m), 0) + textureLoad(src, min(s + vec2<i32>(1, 0), m), 0)
          + textureLoad(src, min(s + vec2<i32>(0, 1), m), 0) + textureLoad(src, min(s + vec2<i32>(1, 1), m), 0)) * 0.25;
}
"#;

struct Gen {
    bgl: wgpu::BindGroupLayout,
    pipes: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    module: wgpu::ShaderModule,
}

static GEN: Mutex<Option<Gen>> = Mutex::new(None);

pub fn wanted() -> bool {
    std::env::var("EVOVIEW_MIPFILL").map_or(true, |v| v != "0")
}

/// The uncompressed format a short chain is completed in.
fn target(src: wgpu::TextureFormat) -> wgpu::TextureFormat {
    use wgpu::TextureFormat as F;
    match src {
        F::Bc6hRgbUfloat | F::Bc6hRgbFloat | F::Rgba16Float => F::Rgba16Float,
        f if f.is_srgb() => F::Rgba8UnormSrgb,
        _ => F::Rgba8Unorm,
    }
}

/// `src` (a 2D texture with `have` levels) with its chain completed down
/// to 1x1, or None when it is already complete (or nothing can be done).
pub fn complete(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Texture, have: u32) -> Option<wgpu::Texture> {
    let (w, h) = (src.width(), src.height());
    let full = 32 - w.max(h).leading_zeros();
    if have >= full || have == 0 {
        return None;
    }
    let fmt = target(src.format());
    let mut g = GEN.lock().ok()?;
    let g = g.get_or_insert_with(|| {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mipfill"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mipfill bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        Gen { bgl, pipes: HashMap::new(), module }
    });
    if !g.pipes.contains_key(&fmt) {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mipfill layout"),
            bind_group_layouts: &[&g.bgl],
            push_constant_ranges: &[],
        });
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mipfill"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &g.module, entry_point: "vs", compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &g.module,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: fmt, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        g.pipes.insert(fmt, pipe);
    }
    let pipe = &g.pipes[&fmt];
    let out = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mipfill"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: full,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let one = |t: &wgpu::Texture, lvl: u32| t.create_view(&wgpu::TextureViewDescriptor {
        base_mip_level: lvl,
        mip_level_count: Some(1),
        ..Default::default()
    });
    let mut enc = device.create_command_encoder(&Default::default());
    let mut keep = Vec::new();
    for lvl in 0..full {
        // the stored levels copied, the rest averaged from the level above
        let (from, from_lvl, mode) = if lvl < have { (src, lvl, 0u32) } else { (&out, lvl - 1, 1u32) };
        let sw = (w >> from_lvl).max(1);
        let sh = (h >> from_lvl).max(1);
        use wgpu::util::DeviceExt;
        let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mipfill p"),
            contents: bytemuck::cast_slice(&[mode, sw, sh, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let sv = one(from, from_lvl);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mipfill bg"),
            layout: &g.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&sv) },
                wgpu::BindGroupEntry { binding: 1, resource: ub.as_entire_binding() },
            ],
        });
        let dv = one(&out, lvl);
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mipfill"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dv,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            rp.set_pipeline(pipe);
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..3, 0..1);
        }
        keep.push((ub, sv, bg, dv));
    }
    queue.submit(Some(enc.finish()));
    drop(keep);
    Some(out)
}
