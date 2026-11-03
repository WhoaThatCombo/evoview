//! Splat-layer maps bound as ONE `texture_2d_array` per kind.
//!
//! EVO's UberVehicle shaders give each splat layer (Red / Green / Blue) its
//! own colour and roughness maps. Three more 2D bindings per kind would push
//! the car shader past the sampled-texture limit of stricter adapters (it is
//! at 20 already, counting group 0), so each kind is one array instead:
//!   binding 15  layer colour maps     (sRGB)
//!   binding 7   layer roughness maps  (linear)
//! and the material uniform carries the array layer of each splat layer
//! (-1 = that layer has no map).
//!
//! Building an array costs nothing in the common case:
//!   0 maps  a shared 1x1 white array
//!   1 map   a D2Array VIEW of the texture already uploaded (no copy)
//!   2-3     resampled on the GPU into an RGBA8 array (sizes and formats
//!           differ between layers - a 2048 BC7 leather and a 512 BC1 noise),
//!           at the largest source size capped to 1024, with a full mip chain.
//!           Cached by source set; only a handful of materials have two
//!           mapped layers (stock: Green oil + Blue noise on exhaust parts).

use std::collections::HashMap;
use std::rc::Rc;

const BLIT_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<uniform> p: vec4<f32>;
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, smp, pos.xy * p.xy, p.z);
}
"#;

const MAX_EDGE: u32 = 1024;

pub struct LayerArrays {
    white: Rc<wgpu::Texture>,
    sampler: wgpu::Sampler,
    bgl: wgpu::BindGroupLayout,
    shader: wgpu::ShaderModule,
    pipes: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    /// (source texture identities, srgb) -> resampled array
    cache: HashMap<(Vec<usize>, bool), Rc<wgpu::Texture>>,
    /// arrays resampled, for the log
    pub resampled: u32,
    pub bytes: u64,
}

impl LayerArrays {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        use wgpu::util::DeviceExt;
        let white = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("layer white"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            Default::default(),
            &[255, 255, 255, 255],
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("layer blit"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layer blit"),
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
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layer blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        LayerArrays {
            white: Rc::new(white),
            sampler,
            bgl,
            shader,
            pipes: HashMap::new(),
            cache: HashMap::new(),
            resampled: 0,
            bytes: 0,
        }
    }

    /// The 1x1 white array every material without layer maps binds.
    pub fn white_view(&self) -> wgpu::TextureView {
        array_view(&self.white, 1)
    }

    /// One array holding `maps` in order (layer i = maps[i]).
    pub fn build(&mut self, device: &wgpu::Device, queue: &wgpu::Queue,
                 maps: &[Rc<wgpu::Texture>], srgb: bool) -> wgpu::TextureView {
        match maps.len() {
            0 => self.white_view(),
            1 => array_view(&maps[0], 1),
            n => {
                let key = (maps.iter().map(|t| Rc::as_ptr(t) as usize).collect::<Vec<_>>(), srgb);
                if let Some(t) = self.cache.get(&key) {
                    return array_view(t, n as u32);
                }
                let t = Rc::new(self.resample(device, queue, maps, srgb));
                self.cache.insert(key, t.clone());
                array_view(&t, n as u32)
            }
        }
    }

    fn pipe(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) -> &wgpu::RenderPipeline {
        let (bgl, shader) = (&self.bgl, &self.shader);
        self.pipes.entry(format).or_insert_with(|| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("layer blit"),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("layer blit"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: "vs",
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: "fs",
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        })
    }

    fn resample(&mut self, device: &wgpu::Device, queue: &wgpu::Queue,
                maps: &[Rc<wgpu::Texture>], srgb: bool) -> wgpu::Texture {
        use wgpu::util::DeviceExt;
        let w = maps.iter().map(|t| t.width()).max().unwrap_or(1).clamp(1, MAX_EDGE);
        let h = maps.iter().map(|t| t.height()).max().unwrap_or(1).clamp(1, MAX_EDGE);
        let mips = 32 - w.max(h).leading_zeros();
        // roughness maps are single-channel data: R8, a quarter of the memory
        let format = if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::R8Unorm };
        let texel = if srgb { 4 } else { 1 };
        let n = maps.len() as u32;
        let dst = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("layer array"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: n },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let mut bytes = 0u64;
        let mut enc = device.create_command_encoder(&Default::default());
        for (layer, src) in maps.iter().enumerate() {
            let src_view = src.create_view(&Default::default());
            for mip in 0..mips {
                let (dw, dh) = ((w >> mip).max(1), (h >> mip).max(1));
                bytes += (dw * dh * texel) as u64;
                // the source mip whose texel density matches this level
                let lod = ((src.width() as f32 / dw as f32).max(src.height() as f32 / dh as f32))
                    .log2().max(0.0);
                let p = [1.0 / dw as f32, 1.0 / dh as f32, lod, 0.0];
                let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("layer blit"),
                    contents: bytemuck::cast_slice(&p),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("layer blit"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        wgpu::BindGroupEntry { binding: 2, resource: ub.as_entire_binding() },
                    ],
                });
                let target = dst.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: layer as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let pipe = self.pipe(device, format);
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("layer blit"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rp.set_pipeline(pipe);
                rp.set_bind_group(0, &bg, &[]);
                rp.draw(0..3, 0..1);
            }
        }
        queue.submit(Some(enc.finish()));
        self.resampled += 1;
        self.bytes += bytes;
        dst
    }
}

fn array_view(t: &wgpu::Texture, layers: u32) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        base_array_layer: 0,
        array_layer_count: Some(layers),
        ..Default::default()
    })
}
