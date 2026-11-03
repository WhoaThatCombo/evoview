//! The driven car's mirrors (actor f23): centre, left and right.
//!
//! Each mirror is a camera at its glass, rendering the track into a small
//! texture which the glass batch (MIRROR_n) then shows. The aim is the
//! game's: straight back, turned by the actor's yaw / pitch (0 / 0 on the
//! door mirrors, pitch -1 on the centre), with its vertical field of view.
//! Reflecting the driver's eye off the glass instead put the centre mirror
//! on the sky - its modelled angle does not suit the seat.
//!
//! The glass's own uv layout decides the image orientation: the u and v
//! directions are recovered from its triangles, and flipped where a mirror
//! has to (a camera looking back has the car's right on its left).
//!
//! Image quality: the mirrors were a fixed 200-240 px tall, single-sampled,
//! read without mips and drawn with TAA's -1 texture bias (no TAA here), so
//! they shimmered. Now each is sized to its glass's height on screen times
//! EVOVIEW_MIRROR_SS (1.5; 2 is cleaner for +0.6 ms at 3440x1440), drawn at mip bias 0 (track.rs
//! update_mirror), and mipped after every redraw so the glass reads it
//! trilinearly: the extra samples average down instead of aliasing. A
//! mirror whose glass is off screen (chase cameras) is not drawn at all.
//! EVOVIEW_MIRROR_RES=<px> pins the old fixed height (no fitting).

use glam::{Mat4, Vec2, Vec3};

use crate::fxdata::MirrorDef;
use crate::render::GpuMesh;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    /// uv min (xy), 1 / uv range (zw)
    uv: [f32; 4],
    /// flip u, flip v, brightness, -
    flags: [f32; 4],
}

pub struct Mirror {
    pub def: MirrorDef,
    /// (gpu mesh, batch start) of the glass
    glass: Vec<(usize, u32, u32)>,
    /// mesh space: glass centre, normal (towards the driver), u / v directions
    centre: Vec3,
    normal: Vec3,
    du: Vec3,
    dv: Vec3,
    aspect: f32,
    /// the bone carrying most of the glass (the door mirrors ride the doors,
    /// everything rides the rig's hull lean)
    bone: Option<usize>,
    pub msaa: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    /// mip 0, rendered into
    pub resolve: wgpu::TextureView,
    /// per mip level past 0: its view and a bind group reading the level above
    mips: Vec<(wgpu::TextureView, wgpu::BindGroup)>,
    params: wgpu::Buffer,
    bg: wgpu::BindGroup,
    uv: [f32; 4],
    /// the render target's height, pixels
    pub height: u32,
    /// mesh space: half the glass along its v (image height) direction
    half_v: Vec3,
}

pub struct Mirrors {
    pub list: Vec<Mirror>,
    pipe: wgpu::RenderPipeline,
    tex_bgl: wgpu::BindGroupLayout,
    samp: wgpu::Sampler,
    mip_pipe: wgpu::RenderPipeline,
    mip_bgl: wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    /// EVOVIEW_MIRROR_RES: a fixed height, no fitting to the screen
    fixed: Option<u32>,
    /// render pixels per glass pixel on screen, per axis (EVOVIEW_MIRROR_SS)
    ss: f32,
    cam: wgpu::Buffer,
    cam_bg: wgpu::BindGroup,
    bone_bg: wgpu::BindGroup,
    /// this frame's skinning matrices (what the car shader gets)
    pose: Vec<Mat4>,
    next: usize,
    samples: u32,
}

impl Mirrors {
    /// `car_meshes`: the driven car's gpu meshes; `eye`: the driver's eye in
    /// mesh space (the cockpit camera).
    pub fn new(
        device: &wgpu::Device,
        car: &crate::asset::Car,
        meshes: &[GpuMesh],
        car_meshes: &[usize],
        defs: &[MirrorDef],
        eye: Vec3,
        bone_buf: &wgpu::Buffer,
        format: wgpu::TextureFormat,
        samples: u32,
    ) -> Option<Mirrors> {
        let samp = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mirror tex"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
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
        // the mip chain: each level the bilinear average of the one above
        let mip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mirror mip"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
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
            ],
        });
        let mip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mirror mip"),
            source: wgpu::ShaderSource::Wgsl(MIP_WGSL.into()),
        });
        let mip_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mirror mip"),
            bind_group_layouts: &[&mip_bgl],
            push_constant_ranges: &[],
        });
        let mip_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mirror mip"),
            layout: Some(&mip_layout),
            vertex: wgpu::VertexState { module: &mip_shader, entry_point: "vs", compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &mip_shader,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let mut list = Vec::new();
        for d in defs {
            if d.batch.is_empty() {
                continue;
            }
            let mut glass = Vec::new();
            let (mut c, mut nsum, mut tsum, mut bsum, mut wsum) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 0.0f32);
            let (mut uvlo, mut uvhi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
            let mut nv = 0u32;
            let mut bone_w: std::collections::HashMap<usize, f32> = Default::default();
            for &gi in car_meshes {
                let m = &meshes[gi];
                let Some(src) = car.meshes.get(m.src) else { continue };
                for b in src.batches.iter().filter(|b| b.name.eq_ignore_ascii_case(&d.batch) && b.count > 0) {
                    let pos: Vec<f32> = car.floats(src.positions);
                    let nrm: Vec<f32> = car.floats(src.normals);
                    let uvs: Vec<f32> = car.floats(src.uvs);
                    let idx: Vec<u32> = car.u32s(src.indices);
                    let bidx: Vec<f32> = car.floats(src.bone_idx);
                    let bwgt: Vec<f32> = car.floats(src.bone_wgt);
                    let p = |i: u32| Vec3::new(pos[i as usize * 3], pos[i as usize * 3 + 1], pos[i as usize * 3 + 2]);
                    let t = |i: u32| if uvs.len() > i as usize * 2 + 1 { Vec2::new(uvs[i as usize * 2], uvs[i as usize * 2 + 1]) } else { Vec2::ZERO };
                    let end = ((b.start + b.count) as usize).min(idx.len());
                    for tri in idx[b.start as usize..end].chunks_exact(3) {
                        let (p0, p1, p2) = (p(tri[0]), p(tri[1]), p(tri[2]));
                        let (t0, t1, t2) = (t(tri[0]), t(tri[1]), t(tri[2]));
                        let (e1, e2) = (p1 - p0, p2 - p0);
                        let (d1, d2) = (t1 - t0, t2 - t0);
                        let area = e1.cross(e2).length();
                        let det = d1.x * d2.y - d2.x * d1.y;
                        if area < 1e-9 || det.abs() < 1e-12 {
                            continue;
                        }
                        // dP/du, dP/dv
                        let r = 1.0 / det;
                        tsum += (e1 * d2.y - e2 * d1.y) * r * area;
                        bsum += (e2 * d1.x - e1 * d2.x) * r * area;
                        c += (p0 + p1 + p2) / 3.0 * area;
                        wsum += area;
                        for &i in tri {
                            if nrm.len() > i as usize * 3 + 2 {
                                nsum += Vec3::new(nrm[i as usize * 3], nrm[i as usize * 3 + 1], nrm[i as usize * 3 + 2]);
                            }
                            for k in 0..4 {
                                let j = i as usize * 4 + k;
                                if j < bidx.len() && j < bwgt.len() && bwgt[j] > 0.0 {
                                    *bone_w.entry(bidx[j].max(0.0) as usize).or_default() += bwgt[j];
                                }
                            }
                            uvlo = uvlo.min(t(i));
                            uvhi = uvhi.max(t(i));
                            nv += 1;
                        }
                    }
                    glass.push((gi, b.start, b.count));
                }
            }
            if glass.is_empty() || wsum <= 0.0 || nv == 0 {
                eprintln!("mirror {}: batch {} not found on the car", d.kind, d.batch);
                continue;
            }
            let centre = c / wsum;
            let bone = bone_w.into_iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|b| b.0)
                .filter(|&b| b < 128);
            let mut normal = nsum.normalize_or_zero();
            // the glass faces the driver
            if normal.dot(eye - centre) < 0.0 {
                normal = -normal;
            }
            let range = (uvhi - uvlo).max(Vec2::splat(1e-4));
            let (du, dv) = (tsum / wsum, bsum / wsum);
            let aspect = ((du.length() * range.x) / (dv.length() * range.y).max(1e-4)).clamp(0.5, 6.0);
            let fixed: Option<u32> = std::env::var("EVOVIEW_MIRROR_RES").ok().and_then(|v| v.parse().ok());
            // until fitted to the screen (`fit`): the old fixed size
            let h: u32 = fixed.unwrap_or(if d.kind == 0 { 200 } else { 240 });
            let w = ((h as f32 * aspect).round() as u32).clamp(64, 1024);
            let uv = [uvlo.x, uvlo.y, 1.0 / range.x, 1.0 / range.y];
            let params = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mirror params"),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let half_v = dv * range.y * 0.5;
            let t = Targets::new(device, &tex_bgl, &samp, &mip_bgl, &params, format, samples, w, h);
            println!("mirror {} ({}): {} batch(es), {}x{}, fov {:.0}", d.kind, d.batch, glass.len(), w, h, d.fov);
            list.push(Mirror { def: d.clone(), glass, centre, normal, du, dv, aspect, bone, msaa: t.msaa, depth: t.depth,
                               resolve: t.resolve, mips: t.mips, params, bg: t.bg, uv, height: h, half_v });
        }
        if list.is_empty() {
            return None;
        }
        let cam = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mirror glass camera"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bone_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mirror bones"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bone_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mirror bones"),
            layout: &bone_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: bone_buf.as_entire_binding() }],
        });
        let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mirror glass camera"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mirror glass camera"),
            layout: &cam_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam.as_entire_binding() }],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mirror"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mirror.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mirror"),
            bind_group_layouts: &[&cam_bgl, &tex_bgl, &bone_bgl],
            push_constant_ranges: &[],
        });
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mirror glass"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: (crate::render::FLOATS_PER_VERTEX * 4) as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4,
                            4 => Float32x4, 5 => Float32x4],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 80,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Float32x4,
                            10 => Float32x4],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                // drawn in the main pass: its SSR targets too, untouched
                targets: &{
                    let mut t = vec![Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })];
                    t.extend(crate::ssr::extra_targets(false));
                    t
                },
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
            multiview: None,
            cache: None,
        });
        Some(Mirrors {
            list, pipe, tex_bgl, samp, mip_pipe, mip_bgl, format,
            fixed: std::env::var("EVOVIEW_MIRROR_RES").ok().and_then(|v| v.parse().ok()),
            ss: std::env::var("EVOVIEW_MIRROR_SS").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5f32).clamp(0.5, 4.0),
            cam, cam_bg, bone_bg, pose: Vec::new(), next: 0, samples,
        })
    }

    /// The skinning matrices uploaded this frame (row-major engine floats
    /// read column-major, as the shader does).
    pub fn set_pose(&mut self, pose: &[[f32; 16]]) {
        self.pose.clear();
        self.pose.extend(pose.iter().map(Mat4::from_cols_array));
    }

    /// Whether this car batch is mirror glass (drawn by `draw_glass` instead).
    pub fn is_glass(&self, mesh: usize, start: u32) -> bool {
        self.list.iter().any(|m| m.glass.iter().any(|g| g.0 == mesh && g.1 == start))
    }

    /// The mirror to redraw this frame, one in turn: they share one camera
    /// uniform and one cull slot, both written before the frame is submitted.
    /// The glass mirror to redraw this frame (one a frame, in turn), or None
    /// when the car only has a rear-view camera.
    pub fn due(&mut self) -> Option<usize> {
        let glass: Vec<usize> = (0..self.list.len()).filter(|&i| !self.list[i].def.camera).collect();
        if glass.is_empty() {
            return None;
        }
        let i = glass[self.next % glass.len()];
        self.next = self.next.wrapping_add(1);
        Some(i)
    }

    /// The rear-view camera screen, redrawn EVERY frame: a digital mirror is
    /// a live video feed, and at one mirror a frame in turn it ran at a
    /// quarter of the frame rate.
    pub fn camera(&self) -> Option<usize> {
        (0..self.list.len()).find(|&i| self.list[i].def.camera)
    }

    /// (view_proj, eye) of mirror `i` with the car body at `body`; writes
    /// its glass parameters too.
    pub fn view(&self, queue: &wgpu::Queue, i: usize, body: Mat4) -> (Mat4, Vec3) {
        let m = &self.list[i];
        // the glass where its bone has it this frame
        let body = match m.bone.and_then(|b| self.pose.get(b)) {
            Some(bm) => body * *bm,
            None => body,
        };
        // a rear-view camera sits where the actor puts it (on the roof or
        // the rear deck); a mirror at its glass
        let pos = if m.def.camera {
            body.transform_point3(Vec3::from(m.def.offset))
        } else {
            body.transform_point3(m.centre + m.normal * 0.01 + Vec3::from(m.def.offset))
        };
        // straight back (the car faces +Z), turned by the actor's yaw / pitch
        let local = glam::Quat::from_rotation_y(m.def.yaw.to_radians())
            * glam::Quat::from_rotation_x(-m.def.pitch.to_radians()) * -Vec3::Z;
        let dir = body.transform_vector3(local).normalize_or_zero();
        let up = body.transform_vector3(Vec3::Y).normalize_or_zero();
        let right = dir.cross(up).normalize_or_zero();
        let fov = m.def.fov.clamp(5.0, 90.0).to_radians();
        let proj = Mat4::perspective_infinite_rh(fov, m.aspect, 0.15);
        let vp = proj * Mat4::look_at_rh(pos, pos + dir, up);
        // orientation: the camera's right / up against the glass's u / v
        let du = body.transform_vector3(m.du);
        let dv = body.transform_vector3(m.dv);
        let flip_u = if du.dot(right) < 0.0 { 1.0 } else { 0.0 };
        let flip_v = if dv.dot(up) > 0.0 { 1.0 } else { 0.0 };
        let p = Params { uv: m.uv, flags: [flip_u, flip_v, 0.85, 0.0] };
        queue.write_buffer(&m.params, 0, bytemuck::bytes_of(&p));
        (vp, pos)
    }

    /// Whether mirror `i`'s glass can show on the main camera's screen: its
    /// centre in front of the camera and inside the frame (with a margin
    /// for the glass's own size), its face turned towards the eye.
    pub fn seen(&self, i: usize, body: Mat4, vp: Mat4, eye: Vec3) -> bool {
        let m = &self.list[i];
        let body = match m.bone.and_then(|b| self.pose.get(b)) {
            Some(bm) => body * *bm,
            None => body,
        };
        let c = body.transform_point3(m.centre);
        let n = body.transform_vector3(m.normal);
        if n.dot(eye - c) <= 0.0 {
            return false;
        }
        let p = vp * c.extend(1.0);
        p.w > 0.0 && p.x.abs() < p.w * 1.3 && p.y.abs() < p.w * 1.3
    }

    /// Size mirror `i`'s image to its glass on screen (`vp` the main
    /// camera, `screen_h` its height in pixels) times the supersampling.
    /// Recreated only when the size is off by more than a fifth, in 16 px
    /// steps. False: the glass is off screen (nothing to draw).
    pub fn fit(&mut self, device: &wgpu::Device, i: usize, body: Mat4, vp: Mat4, screen_h: u32) -> bool {
        let m = &self.list[i];
        let body = match m.bone.and_then(|b| self.pose.get(b)) {
            Some(bm) => body * *bm,
            None => body,
        };
        let c = body.transform_point3(m.centre);
        let hv = body.transform_vector3(m.half_v);
        let ndc = |p: Vec3| {
            let q = vp * p.extend(1.0);
            (q.w > 0.05).then(|| Vec3::new(q.x / q.w, q.y / q.w, q.w))
        };
        let (Some(a), Some(b), Some(o)) = (ndc(c - hv), ndc(c + hv), ndc(c)) else { return false };
        // its centre well off screen: not drawn (a door mirror out of view)
        if o.x.abs() > 1.6 || o.y.abs() > 1.6 {
            return false;
        }
        if self.fixed.is_some() {
            return true;
        }
        let px = (b.truncate() - a.truncate()).length() * 0.5 * screen_h as f32;
        let want = ((px * self.ss / 16.0).ceil() as u32 * 16).clamp(64, 768);
        let h = self.list[i].height;
        if (want as f32 - h as f32).abs() > 0.2 * h as f32 {
            let m = &self.list[i];
            let w = ((want as f32 * m.aspect).round() as u32).clamp(64, 2048);
            let t = Targets::new(device, &self.tex_bgl, &self.samp, &self.mip_bgl, &m.params, self.format, self.samples, w, want);
            let m = &mut self.list[i];
            (m.msaa, m.depth, m.resolve, m.mips, m.bg, m.height) = (t.msaa, t.depth, t.resolve, t.mips, t.bg, want);
            if std::env::var_os("EVOVIEW_MIRROR_LOG").is_some() {
                println!("mirror {}: {w}x{want} ({px:.0} px tall on screen)", m.def.kind);
            }
        }
        true
    }

    /// Mirror `i`'s mip chain, after its pass.
    pub fn mips(&self, enc: &mut wgpu::CommandEncoder, i: usize) {
        for (view, src) in &self.list[i].mips {
            let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mirror mip"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            p.set_pipeline(&self.mip_pipe);
            p.set_bind_group(0, src, &[]);
            p.draw(0..3, 0..1);
        }
    }

    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// Every mirror's glass with what it sees; the main camera's view_proj.
    pub fn draw_glass<'a>(&'a self, queue: &wgpu::Queue, pass: &mut wgpu::RenderPass<'a>, meshes: &'a [GpuMesh], vp: Mat4) {
        queue.write_buffer(&self.cam, 0, bytemuck::bytes_of(&vp.to_cols_array_2d()));
        pass.set_pipeline(&self.pipe);
        pass.set_bind_group(0, &self.cam_bg, &[]);
        pass.set_bind_group(2, &self.bone_bg, &[]);
        for m in &self.list {
            pass.set_bind_group(1, &m.bg, &[]);
            for &(gi, start, count) in &m.glass {
                let g = &meshes[gi];
                pass.set_vertex_buffer(0, g.vertices.slice(..));
                pass.set_vertex_buffer(1, g.instances.slice(..));
                pass.set_index_buffer(g.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(start..start + count, 0, 0..g.instance_count.max(1));
            }
        }
    }
}

const MIP_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
struct V { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) i: u32) -> V {
    var o: V;
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    o.p = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}
@fragment
fn fs(v: V) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, s, v.uv, 0.0);
}
"#;

/// A mirror's render targets at one size.
struct Targets {
    msaa: wgpu::TextureView,
    depth: wgpu::TextureView,
    resolve: wgpu::TextureView,
    mips: Vec<(wgpu::TextureView, wgpu::BindGroup)>,
    bg: wgpu::BindGroup,
}

impl Targets {
    #[allow(clippy::too_many_arguments)]
    fn new(device: &wgpu::Device, tex_bgl: &wgpu::BindGroupLayout, samp: &wgpu::Sampler, mip_bgl: &wgpu::BindGroupLayout,
           params: &wgpu::Buffer, format: wgpu::TextureFormat, samples: u32, w: u32, h: u32) -> Targets {
        let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
        let tex = |fmt, s: u32, mips: u32, usage| device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mirror"),
            size,
            mip_level_count: mips,
            sample_count: s,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        });
        let ra = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let msaa = tex(format, samples, 1, ra).create_view(&Default::default());
        let depth = tex(wgpu::TextureFormat::Depth32Float, samples, 1, ra).create_view(&Default::default());
        let levels = 32 - w.max(h).leading_zeros();
        let image = tex(format, 1, levels, ra | wgpu::TextureUsages::TEXTURE_BINDING);
        let level = |l: u32| image.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: l,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let mips = (1..levels).map(|l| {
            let src = level(l - 1);
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mirror mip"),
                layout: mip_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(samp) },
                ],
            });
            (level(l), bg)
        }).collect();
        let all = image.create_view(&Default::default());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mirror"),
            layout: tex_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&all) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(samp) },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
            ],
        });
        Targets { msaa, depth, resolve: level(0), mips, bg }
    }
}
