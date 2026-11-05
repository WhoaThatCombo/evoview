//! Per-pixel motion vectors (motion.wgsl), for TAA and DLSS.
//!
//! TAA used to reproject from depth and the previous camera only, with the
//! driven car approximated by one box and its wheels given no history at all;
//! replay cars had nothing. Here every pixel gets where its surface was last
//! frame: the camera's motion through the depth buffer for the scenery, and
//! for the cars (every mesh that is not track geometry) a second draw with
//! this and last frame's bone table and instance matrices - body, wheel spin,
//! steering, suspension and other cars all come out exact.
//!
//! Last frame's bones and instance matrices are GPU copies taken at the end
//! of each frame, after the pass that read them (queue writes of the next
//! frame land before its commands). EVOVIEW_MOTION=0 turns it off; TAA falls
//! back to its depth + box reprojection.

use glam::Mat4;

/// xy: last frame's position minus this frame's, render pixels, unjittered;
/// z: last frame's view depth; w: 1 on a moving object
pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// the copy DLSS reads (its documented RG16_FLOAT)
pub const RG_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct U {
    vp_jit: [[f32; 4]; 4],
    inv_jit: [[f32; 4]; 4],
    vp: [[f32; 4]; 4],
    prev_vp: [[f32; 4]; 4],
    eye: [f32; 4],
    rsize: [f32; 4],
}

pub fn wanted() -> bool {
    crate::envc::var("EVOVIEW_MOTION").map_or(true, |v| v != "0")
}

pub struct Motion {
    cam_pipe: wgpu::RenderPipeline,
    obj_pipe: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    ubuf: wgpu::Buffer,
    prev_bones: wgpu::Buffer,
    /// the wheel bones (set_spin)
    spin_buf: wgpu::Buffer,
    /// last frame's instance rows, per mesh index (cars only)
    prev_inst: Vec<Option<wgpu::Buffer>>,
    pub view: wgpu::TextureView,
    pub tex: wgpu::Texture,
    /// DLSS's copy (RG16F, the format it takes): xy only
    pub rg: Option<(wgpu::Texture, wgpu::TextureView)>,
    size: (u32, u32),
    prev_vp: Option<Mat4>,
    /// last frame's copies are this frame's data (first frame, reset)
    primed: bool,
}

impl Motion {
    /// `rg`: also write the RG16F copy DLSS reads.
    /// `bone_bytes`: the size of the renderer's bone buffer (all its blocks)
    pub fn new(device: &wgpu::Device, rw: u32, rh: u32, samples: u32, rg: bool, bone_bytes: u64) -> Motion {
        // EVOVIEW_WHEEL_MV_PX: a wheel spinning more than this many render
        // px a frame moves with its centre only (0 / EVOVIEW_WHEEL_MV=0: off)
        let wheel_px = if crate::envc::var("EVOVIEW_WHEEL_MV").map_or(false, |v| v == "0") { 0.0 }
            else { crate::envc::var("EVOVIEW_WHEEL_MV_PX").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(3.0) };
        let src = include_str!("motion.wgsl")
            .replace("WHEEL_MV_PX", &format!("{wheel_px:.3}"))
            .replace("DEPTH_TEXTURE", if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" })
            .replace("MV_OUT", if rg {
                "struct MvOut { @location(0) mv: vec4<f32>, @location(1) rg: vec2<f32> };
                 fn mv_out(v: vec4<f32>) -> MvOut { return MvOut(v, v.xy); }"
            } else {
                "struct MvOut { @location(0) mv: vec4<f32> };
                 fn mv_out(v: vec4<f32>) -> MvOut { return MvOut(v); }"
            });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let ubo = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let sto = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("motion bgl"),
            entries: &[
                ubo(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: samples > 1 },
                    count: None,
                },
                sto(2),
                sto(3),
                ubo(4),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("motion layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let t = |f| Some(wgpu::ColorTargetState { format: f, blend: None, write_mask: wgpu::ColorWrites::ALL });
        let target_rg = [t(FORMAT), t(RG_FORMAT)];
        let target: &[Option<wgpu::ColorTargetState>] = if rg { &target_rg } else { &target_rg[..1] };
        let cam_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("motion camera"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_full", compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_cam", compilation_options: Default::default(), targets: target }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let fpv = (crate::render::FLOATS_PER_VERTEX * 4) as u64;
        let obj_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("motion objects"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_obj",
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: fpv,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        // render.rs's vertex: pos3 nrm3 uv2 tan4 boneidx4 bonewgt4
                        attributes: &[
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 4 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 64, shader_location: 5 },
                        ],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 80,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Float32x4, 10 => Float32x4],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 80,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![11 => Float32x4, 12 => Float32x4, 13 => Float32x4, 14 => Float32x4],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_obj", compilation_options: Default::default(), targets: target }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion"),
            size: std::mem::size_of::<U>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let prev_bones = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion prev bones"),
            size: bone_bytes.max((crate::render::MAX_BONES * 64) as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // per bone: xyz its rest origin in mesh space, w 1 = it spins (a
        // wheel); x of the first row also carries the px threshold
        let spin_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion spin bones"),
            size: 256 * 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (tex, view) = Self::target(device, rw, rh, FORMAT);
        let rg = rg.then(|| Self::target(device, rw, rh, RG_FORMAT));
        Motion { cam_pipe, obj_pipe, bgl, ubuf, prev_bones, spin_buf, prev_inst: Vec::new(), view, tex, rg, size: (rw, rh), prev_vp: None, primed: false }
    }

    /// The bones that spin with a wheel, with their rest origin (the wheel
    /// centre) in mesh space. A fast-spinning wheel's own rotation is taken
    /// out of its vectors (see motion.wgsl fs_obj).
    pub fn set_spin(&self, queue: &wgpu::Queue, bones: &[(usize, glam::Vec3)]) {
        let mut t = vec![[0.0f32; 4]; 256];
        for &(b, c) in bones {
            if b < t.len() {
                t[b] = [c.x, c.y, c.z, 1.0];
            }
        }
        queue.write_buffer(&self.spin_buf, 0, bytemuck::cast_slice(&t));
    }

    fn target(device: &wgpu::Device, w: u32, h: u32, format: wgpu::TextureFormat) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("motion vectors"),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    }

    pub fn resize(&mut self, device: &wgpu::Device, rw: u32, rh: u32) {
        if (rw, rh) == self.size {
            return;
        }
        let (tex, view) = Self::target(device, rw, rh, FORMAT);
        self.tex = tex;
        self.view = view;
        if self.rg.is_some() {
            self.rg = Some(Self::target(device, rw, rh, RG_FORMAT));
        }
        self.size = (rw, rh);
        self.prev_vp = None;
    }

    /// The camera last frame (unjittered), None after a reset; and the
    /// target's size. For other passes drawing into `view` after `run`.
    pub fn prev_vp(&self) -> Option<Mat4> {
        self.prev_vp
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Forget last frame (camera cut, teleport, car swap).
    pub fn reset(&mut self) {
        self.prev_vp = None;
        self.primed = false;
    }

    /// Write this frame's vectors. `vp_jit` is the projection the main pass
    /// drew with, `vp` the same without jitter; `dynamic` names the meshes
    /// that move (the cars), drawn with `bones` for their skinning.
    #[allow(clippy::too_many_arguments)]
    pub fn run(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder,
               depth: &wgpu::TextureView, bones: &wgpu::Buffer, meshes: &[crate::render::GpuMesh],
               dynamic: &dyn Fn(usize) -> bool, opaque: &dyn Fn(usize) -> bool,
               vp_jit: Mat4, vp: Mat4, eye: glam::Vec3) {
        let (rw, rh) = self.size;
        let u = U {
            vp_jit: vp_jit.to_cols_array_2d(),
            inv_jit: vp_jit.inverse().to_cols_array_2d(),
            vp: vp.to_cols_array_2d(),
            prev_vp: self.prev_vp.unwrap_or(vp).to_cols_array_2d(),
            eye: [eye.x, eye.y, eye.z, 1.0],
            rsize: [rw as f32, rh as f32, 1.0 / rw.max(1) as f32, 1.0 / rh.max(1) as f32],
        };
        queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&u));
        if self.prev_inst.len() < meshes.len() {
            self.prev_inst.resize_with(meshes.len(), || None);
        }
        let dyns: Vec<usize> = (0..meshes.len()).filter(|&i| dynamic(i) && meshes[i].instance_count > 0).collect();
        for &i in &dyns {
            let sz = meshes[i].instances.size();
            if self.prev_inst[i].as_ref().map_or(true, |b| b.size() != sz) {
                self.prev_inst[i] = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("motion prev instances"),
                    size: sz,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.primed = false;
            }
        }
        // no last frame yet: it is this one (zero motion)
        if !self.primed {
            self.copy_prev(enc, bones, meshes, &dyns);
        }
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motion bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: bones.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.prev_bones.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.spin_buf.as_entire_binding() },
            ],
        });
        let att = |v| Some(wgpu::RenderPassColorAttachment {
            view: v,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
        });
        let atts = [att(&self.view), att(self.rg.as_ref().map_or(&self.view, |r| &r.1))];
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("motion vectors"),
                color_attachments: &atts[..if self.rg.is_some() { 2 } else { 1 }],
                ..Default::default()
            });
            pass.set_bind_group(0, &bg, &[]);
            pass.set_pipeline(&self.cam_pipe);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&self.obj_pipe);
            for &i in &dyns {
                let m = &meshes[i];
                let Some(prev) = self.prev_inst[i].as_ref() else { continue };
                pass.set_vertex_buffer(0, m.vertices.slice(..));
                pass.set_vertex_buffer(1, m.instances.slice(..));
                pass.set_vertex_buffer(2, prev.slice(..));
                pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                for &(start, count, mi) in &m.draws {
                    if opaque(mi) {
                        pass.draw_indexed(start..start + count, 0, 0..m.instance_count);
                    }
                }
            }
        }
        self.copy_prev(enc, bones, meshes, &dyns);
        self.primed = true;
        self.prev_vp = Some(vp);
    }

    /// Last frame's instance rows for mesh `i`, from the caller: carcull.rs
    /// compacts a moving car mesh's instances to the visible cars, so this
    /// frame's row k need not be last frame's row k (a GPU copy would pair
    /// one car's matrix with another's). Written before this frame's `run`.
    pub fn set_prev(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, meshes: &[crate::render::GpuMesh],
                    i: usize, rows: &[f32]) {
        if self.prev_inst.len() < meshes.len() {
            self.prev_inst.resize_with(meshes.len(), || None);
        }
        let sz = meshes[i].instances.size();
        if self.prev_inst[i].as_ref().map_or(true, |b| b.size() != sz) {
            self.prev_inst[i] = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("motion prev instances"),
                size: sz,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if let Some(b) = self.prev_inst[i].as_ref() {
            let n = (rows.len() * 4).min(sz as usize) / 4;
            if n > 0 {
                queue.write_buffer(b, 0, bytemuck::cast_slice(&rows[..n]));
            }
        }
    }

    /// Last frame's bones for bone blocks from `first` (of 256 bones): a
    /// replay car posed this frame but not the last (carcull.rs skips the
    /// rigs of cars off screen) takes this frame's pose, not a stale one -
    /// stale, its first frame blurred along a huge false vector.
    pub fn set_prev_bones(&self, queue: &wgpu::Queue, first: usize, bones: &[[f32; 16]]) {
        let off = (first * crate::render::MAX_BONES * 64) as u64;
        if off + (bones.len() * 64) as u64 <= self.prev_bones.size() {
            queue.write_buffer(&self.prev_bones, off, bytemuck::cast_slice(bones));
        }
    }

    fn copy_prev(&self, enc: &mut wgpu::CommandEncoder, bones: &wgpu::Buffer, meshes: &[crate::render::GpuMesh], dyns: &[usize]) {
        enc.copy_buffer_to_buffer(bones, 0, &self.prev_bones, 0, self.prev_bones.size().min(bones.size()));
        for &i in dyns {
            if let Some(p) = self.prev_inst[i].as_ref() {
                enc.copy_buffer_to_buffer(&meshes[i].instances, 0, p, 0, p.size());
            }
        }
    }
}
