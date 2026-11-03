//! Falling rain: a field of streaks around the camera, drawn after TAA.
//!
//! The game emits its falling_rain mesh (a cluster of streak quads) as
//! particles around the camera (content/particles/falling_rain). Here each
//! streak is an instance placed on a world-fixed lattice that wraps around
//! the eye, so walking or driving through the rain does not drag it along.
//! A streak spans the distance its drop travels RELATIVE TO THE CAMERA in a
//! short exposure, so rain slants back at speed. It is drawn after TAA -
//! thin, fast and without motion vectors, the history would wipe it out -
//! and tests itself against the scene depth like the soft particles.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct U {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    /// x time, y intensity, z streak count, w field radius (m)
    p: [f32; 4],
    /// drop velocity relative to the camera (m/s), w exposure (s)
    vel: [f32; 4],
    /// sun direction + strength
    sun: [f32; 4],
    /// sun colour, w exposure multiplier (HDR)
    sun_c: [f32; 4],
    /// sky colour (the ambient the streaks scatter)
    sky: [f32; 4],
    /// target size (xy), depth size (zw)
    size: [f32; 4],
}

const SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    p: vec4<f32>,
    vel: vec4<f32>,
    sun: vec4<f32>,
    sun_c: vec4<f32>,
    sky: vec4<f32>,
    size: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_depth: texture_depth_2d;

struct O {
    @builtin(position) clip: vec4<f32>,
    @location(0) t: vec2<f32>,
    @location(1) a: f32,
    @location(2) vz: f32,
};

fn hash(n: u32) -> f32 {
    var x = n * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    return f32(x) / 4294967295.0;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> O {
    var o: O;
    let r = u.p.w;
    let h = r * 0.8;
    // a world-fixed lattice, wrapped into the box around the eye
    let seed = vec3<f32>(hash(ii * 3u), hash(ii * 3u + 1u), hash(ii * 3u + 2u));
    var pos = seed * vec3<f32>(2.0 * r, 2.0 * h, 2.0 * r);
    pos.y = pos.y - u.p.x * 9.0;
    let base = u.eye.xyz - vec3<f32>(r, h, r);
    pos = base + (pos - base - floor((pos - base) / vec3<f32>(2.0 * r, 2.0 * h, 2.0 * r)) * vec3<f32>(2.0 * r, 2.0 * h, 2.0 * r));
    // the streak: where the drop is, and where it was an exposure ago
    let a = pos;
    let b = pos - u.vel.xyz * u.vel.w;
    let ca = u.view_proj * vec4<f32>(a, 1.0);
    let cb = u.view_proj * vec4<f32>(b, 1.0);
    let end = (vi & 1u) == 1u;
    let side = select(-1.0, 1.0, (vi & 2u) == 2u);
    let c = select(ca, cb, end);
    // screen-space width: ~1.3 px, wider for drops very close
    let sa = ca.xy / max(ca.w, 0.05);
    let sb = cb.xy / max(cb.w, 0.05);
    var dir = (sb - sa) * u.size.xy;
    if (dot(dir, dir) < 1e-6) {
        dir = vec2<f32>(0.0, 1.0);
    }
    let nrm = normalize(vec2<f32>(-dir.y, dir.x)) / u.size.xy;
    let px = 1.3 + 3.0 / max(c.w, 0.3);
    o.clip = c + vec4<f32>(nrm * side * px * c.w, 0.0, 0.0);
    o.t = vec2<f32>(side, select(0.0, 1.0, end));
    // fade out near the eye (no drops in your face) and at the field's edge
    let d = distance(a, u.eye.xyz);
    // (none within 2 m: glass writes no depth, and from the cockpit drops
    // between the eye and the windshield would fall inside the car)
    o.a = smoothstep(2.0, 3.5, d) * (1.0 - smoothstep(r * 0.7, r, d)) * select(0.0, 1.0, ca.w > 0.1);
    o.vz = c.w;
    // fewer streaks for lighter rain
    if (f32(ii) > u.p.y * u.p.z) {
        o.clip = vec4<f32>(0.0, 0.0, -1.0, 1.0);
    }
    return o;
}

@fragment
fn fs(in: O) -> @location(0) vec4<f32> {
    // behind the scene: the depth at this pixel (render size may differ)
    let px = vec2<i32>(in.clip.xy * u.size.zw / u.size.xy);
    let dz = textureLoad(t_depth, px, 0);
    if (dz < 0.9999999) {
        // infinite-far projection: ndc = 1 - near / w (fx.wgsl soft_fade)
        let ws = in.vz * (1.0 - in.clip.z) / max(1.0 - dz, 1e-7);
        if (ws < in.vz) {
            discard;
        }
    }
    // a soft streak: thin across, fading at both ends
    let across = 1.0 - in.t.x * in.t.x;
    let along = sin(in.t.y * 3.14159);
    let cov = across * along * in.a * 0.35;
    // water scatters the sky and catches the sun
    let col = u.sky.rgb * 0.9 + u.sun_c.rgb * u.sun.w * 0.25;
    return vec4<f32>(col * u.sun_c.w * cov, cov);
}
"#;

pub struct Rainfall {
    pipe: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    buf: wgpu::Buffer,
    time: f32,
}

/// Streaks at full rain.
const COUNT: u32 = 24000;

impl Rainfall {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Rainfall {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rainfall"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rainfall"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rainfall"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rainfall"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &module, entry_point: "vs", buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // premultiplied: colour already scaled by coverage
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                        alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rainfall"),
            size: std::mem::size_of::<U>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Rainfall { pipe, bgl, buf, time: 0.0 }
    }

    /// Draw onto `target` (the HDR scene after TAA). `depth`: the scene's
    /// (single-sample) depth; `car_vel`: the camera's world velocity.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder,
                target: &wgpu::TextureView, target_size: (u32, u32), depth: &wgpu::TextureView, depth_size: (u32, u32),
                view_proj: glam::Mat4, eye: glam::Vec3, car_vel: glam::Vec3, intensity: f32, exposure: f32, dt: f32) {
        if intensity <= 0.0 {
            return;
        }
        self.time += dt;
        let b = crate::sky::block();
        // drops fall at ~9 m/s; relative to a moving camera they come at it
        let vel = glam::Vec3::new(0.0, -9.0, 0.0) - car_vel;
        let u = U {
            view_proj: view_proj.to_cols_array_2d(),
            eye: [eye.x, eye.y, eye.z, 1.0],
            p: [self.time, intensity.clamp(0.0, 1.0), COUNT as f32, 18.0],
            vel: [vel.x, vel.y, vel.z, 0.022],
            sun: [b[0][0], b[0][1], b[0][2], b[0][3].min(1.0)],
            sun_c: [b[1][0], b[1][1], b[1][2], exposure],
            sky: [b[2][0] * b[1][3], b[2][1] * b[1][3], b[2][2] * b[1][3], 0.0],
            size: [target_size.0 as f32, target_size.1 as f32, depth_size.0 as f32, depth_size.1 as f32],
        };
        queue.write_buffer(&self.buf, 0, bytemuck::bytes_of(&u));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rainfall"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
            ],
        });
        let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("rainfall"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        p.set_pipeline(&self.pipe);
        p.set_bind_group(0, &bg, &[]);
        p.draw(0..4, 0..COUNT);
    }
}
