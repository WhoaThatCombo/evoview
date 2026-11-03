//! The driven car's reflection probe: a cube of the track around the car.
//!
//! On a track the car's paint used to reflect `env_sky`, a blue-to-grey
//! gradient - no circuit, no clouds, no grandstands. A clear coat with
//! nothing in it to reflect is what made the paint read flat and grey next
//! to the game, whose cars sit in their surroundings.
//!
//! One face a frame is drawn through the track pipelines (like the mirrors,
//! the track only - the car does not reflect itself), from the middle of
//! the car, and that face's mips are then blurred down from the level above
//! so a rough surface sees a wider cone (cube_lod in shader.wgsl). All six
//! faces are fresh every 6 frames.
//!
//! Faces are stored the way the game samples its cubes, with z negated
//! (shader.wgsl cube_dir): seen that way every face is an ordinary
//! right-handed camera, no mirrored image and no flipped winding.

use glam::{Mat4, Vec3};

/// face size in pixels (EVOVIEW_PROBE_RES)
fn size() -> u32 {
    std::env::var("EVOVIEW_PROBE_RES").ok().and_then(|v| v.parse().ok()).unwrap_or(256u32).clamp(16, 1024)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    /// x face, y blur radius (tan of the half-angle), z destination size
    p: [f32; 4],
}

pub struct Probe {
    pub cube: wgpu::TextureView,
    /// render target per (mip, face): mip * 6 + face
    faces: Vec<wgpu::TextureView>,
    msaa: Option<wgpu::TextureView>,
    depth: wgpu::TextureView,
    pipe: wgpu::RenderPipeline,
    /// blur of (mip >= 1, face) from the level above: (mip - 1) * 6 + face
    blur: Vec<wgpu::BindGroup>,
    pub mips: u32,
    pub size: u32,
    next: usize,
}

const BLUR: &str = r#"
struct P { p: vec4<f32> };
@group(0) @binding(0) var src: texture_cube<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> u: P;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(i & 1u) * 4 - 1);
    let y = f32(i32(i >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

/// texture-space direction of face texel uv (the cube map convention)
fn face_dir(face: i32, uv: vec2<f32>) -> vec3<f32> {
    let sc = uv.x * 2.0 - 1.0;
    let tc = uv.y * 2.0 - 1.0;
    switch face {
        case 0: { return vec3<f32>(1.0, -tc, -sc); }
        case 1: { return vec3<f32>(-1.0, -tc, sc); }
        case 2: { return vec3<f32>(sc, 1.0, tc); }
        case 3: { return vec3<f32>(sc, -1.0, -tc); }
        case 4: { return vec3<f32>(sc, -tc, 1.0); }
        default: { return vec3<f32>(-sc, -tc, -1.0); }
    }
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let d = normalize(face_dir(i32(u.p.x), pos.xy / u.p.z));
    let t = normalize(cross(select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(d.y) > 0.9), d));
    let b = cross(d, t);
    // a 5x5 gaussian over the cone, from the sharper level above
    var c = vec3<f32>(0.0);
    var w = 0.0;
    for (var i = -2; i <= 2; i = i + 1) {
        for (var j = -2; j <= 2; j = j + 1) {
            let o = vec2<f32>(f32(i), f32(j)) * 0.5;
            let k = exp(-dot(o, o));
            c += textureSampleLevel(src, samp, normalize(d + (t * o.x + b * o.y) * u.p.y), 0.0).rgb * k;
            w += k;
        }
    }
    return vec4<f32>(c / w, 1.0);
}
"#;

impl Probe {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat, samples: u32) -> Probe {
        let size = size();
        let mips = size.ilog2() - 1; // down to 4 px faces
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("reflection probe"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 6 },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let cube = tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let mut faces = Vec::new();
        for mip in 0..mips {
            for face in 0..6 {
                faces.push(tex.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: face,
                    array_layer_count: Some(1),
                    ..Default::default()
                }));
            }
        }
        let att = |fmt, s| device.create_texture(&wgpu::TextureDescriptor {
            label: Some("reflection probe target"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: s,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        }).create_view(&Default::default());
        let msaa = (samples > 1).then(|| att(format, samples));
        let depth = att(wgpu::TextureFormat::Depth32Float, samples);

        let samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("probe blur"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe blur"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("probe blur"),
            source: wgpu::ShaderSource::Wgsl(BLUR.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe blur"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("probe blur"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &module, entry_point: "vs", buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let mut blur = Vec::new();
        for mip in 1..mips {
            let src = tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                base_mip_level: mip - 1,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let dst = (size >> mip) as f32;
            for face in 0..6 {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("probe blur"),
                    size: 16,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                // the cone grows with the destination texel (2 / size across
                // a face), so each level is about twice as blurred as the last
                queue.write_buffer(&buf, 0, bytemuck::bytes_of(&Params { p: [face as f32, 2.0 / dst, dst, 0.0] }));
                blur.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("probe blur"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&samp) },
                        wgpu::BindGroupEntry { binding: 2, resource: buf.as_entire_binding() },
                    ],
                }));
            }
        }
        Probe { cube, faces, msaa, depth, pipe, blur, mips, size, next: 0 }
    }

    /// The face to draw this frame.
    pub fn due(&mut self) -> usize {
        let f = self.next % 6;
        self.next += 1;
        f
    }

    /// Every face has been drawn at least once.
    pub fn ready(&self) -> bool {
        self.next >= 6
    }

    /// (view_proj, eye) of `face` from `pos` (world).
    pub fn view(face: usize, pos: Vec3) -> (Mat4, Vec3) {
        // world forward and up of each face, cube_dir's z negation included
        let (f, up) = match face {
            0 => (Vec3::X, Vec3::Y),
            1 => (Vec3::NEG_X, Vec3::Y),
            2 => (Vec3::Y, Vec3::Z),
            3 => (Vec3::NEG_Y, Vec3::NEG_Z),
            4 => (Vec3::NEG_Z, Vec3::Y),
            _ => (Vec3::Z, Vec3::Y),
        };
        let proj = Mat4::perspective_infinite_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.15);
        (proj * Mat4::look_at_rh(pos, pos + f, up), pos)
    }

    /// The pass's colour attachment for `face` (mip 0).
    pub fn attachment(&self, face: usize, clear: wgpu::Color) -> wgpu::RenderPassColorAttachment<'_> {
        wgpu::RenderPassColorAttachment {
            view: self.msaa.as_ref().unwrap_or(&self.faces[face]),
            resolve_target: self.msaa.is_some().then(|| &self.faces[face]),
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
        }
    }

    pub fn depth(&self) -> &wgpu::TextureView {
        &self.depth
    }

    /// Blur `face`'s mip chain down from the freshly drawn level 0.
    pub fn filter(&self, enc: &mut wgpu::CommandEncoder, face: usize) {
        for mip in 1..self.mips as usize {
            let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("probe blur"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.faces[mip * 6 + face],
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            p.set_pipeline(&self.pipe);
            p.set_bind_group(0, &self.blur[(mip - 1) * 6 + face], &[]);
            p.draw(0..3, 0..1);
        }
    }
}
