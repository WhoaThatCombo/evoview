//! A 2D overlay: speed, rpm, gear, pedals and a timeline.
//!
//! evoview has no text renderer and does not want one. Everything here is
//! axis-aligned coloured rectangles, and digits are drawn as seven segments -
//! so there is no font, no atlas, no glyph packing and no text shaping, and
//! the whole overlay is one vertex buffer and one three-line shader.
//!
//! ponytail: seven-segment means digits only. If this ever needs words, that
//! is the point to add a real bitmap font, not sooner.

use wgpu::util::DeviceExt;

/// What a click on the overlay asks for.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Action {
    PlayPause,
    Slower,
    Faster,
    Restart,
    Camera,
    PrevCar,
    NextCar,
    /// Somewhere on the timeline, as a fraction of the replay.
    Seek(f32),
}

/// Screen-space quads, built fresh each frame. A HUD is a few hundred
/// vertices; rebuilding beats tracking what changed.
pub struct Hud {
    /// drawn over the finished frame (`overlay`): in the main pass, TAA and
    /// the post chain jittered, fogged and sharpened it
    pipeline: wgpu::RenderPipeline,
    buffer: wgpu::Buffer,
    verts: Vec<f32>,
    capacity: usize,
    /// Clickable regions for this frame, in the same 0..1 screen space as the
    /// drawing calls. Rebuilt with the geometry so the two cannot disagree.
    hits: Vec<(f32, f32, f32, f32, Action)>,
    /// text: glyphs rasterised on first use at the exact pixel size asked
    /// for, packed into one coverage atlas (plus a disc for round corners)
    atlas: wgpu::Texture,
    bind: wgpu::BindGroup,
    cpu: Vec<u8>,
    dirty: bool,
    /// shelf packer: pen x, shelf top, shelf height
    shelf: (u32, u32, u32),
    glyphs: std::collections::HashMap<(Weight, char, u16), Glyph>,
    /// Segoe UI regular + semibold (ACECM's face), from Windows' own fonts;
    /// empty = the 5x7 pixel font
    fonts: Vec<ab_glyph::FontVec>,
    /// the target's size in pixels (`begin`)
    sw: f32,
    sh: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Weight {
    Regular,
    Semibold,
}

/// A rasterised glyph: atlas rect (u0 v0 u1 v1) and its pixel box from the
/// pen on the baseline.
#[derive(Clone, Copy)]
struct Glyph {
    uv: [f32; 4],
    w: f32,
    h: f32,
    ox: f32,
    oy: f32,
}

/// x, y in NDC, rgba, then the atlas uv (u < 0: a flat colour).
const FLOATS_PER_VERT: usize = 8;
const ATLAS: u32 = 1024;
/// the anti-aliased disc in the atlas's top-left corner, for round corners
const DISC: u32 = 64;

/// ACECM's skin (acecm/web/skin.css), as linear colours for the sRGB target.
pub mod theme {
    pub fn hex(rgb: u32, a: f32) -> [f32; 4] {
        let c = |v: u32| {
            let s = (v & 0xff) as f32 / 255.0;
            if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
        };
        [c(rgb >> 16), c(rgb >> 8), c(rgb), a]
    }
    pub fn panel(a: f32) -> [f32; 4] { hex(0x222731, a) }
    pub fn panel2(a: f32) -> [f32; 4] { hex(0x2a303b, a) }
    pub fn well(a: f32) -> [f32; 4] { hex(0x1d2129, a) }
    pub fn line(a: f32) -> [f32; 4] { hex(0x353c48, a) }
    pub fn fg(a: f32) -> [f32; 4] { hex(0xe4e7ec, a) }
    pub fn dim(a: f32) -> [f32; 4] { hex(0x97a0ad, a) }
    pub fn dim2(a: f32) -> [f32; 4] { hex(0x727b88, a) }
    pub fn accent(a: f32) -> [f32; 4] { hex(0xd9342b, a) }
    pub fn gold(a: f32) -> [f32; 4] { hex(0xe2b04a, a) }
    pub fn red(a: f32) -> [f32; 4] { hex(0xe5534b, a) }
    pub fn ok(a: f32) -> [f32; 4] { hex(0x57b37c, a) }
}

/// Which of the seven segments each digit lights, in the order
/// top, top-left, top-right, middle, bottom-left, bottom-right, bottom.
const DIGITS: [u8; 10] = [
    0b1110111, 0b0010010, 0b1011101, 0b1011011, 0b0111010,
    0b1101011, 0b1101111, 0b1010010, 0b1111111, 0b1111011,
];

/// The round-corner disc: coverage of a circle filling DISC x DISC, 4x4
/// supersampled.
fn disc(cpu: &mut [u8]) {
    let r = DISC as f32 * 0.5;
    for y in 0..DISC {
        for x in 0..DISC {
            let mut n = 0;
            for sy in 0..4 {
                for sx in 0..4 {
                    let dx = x as f32 + (sx as f32 + 0.5) / 4.0 - r;
                    let dy = y as f32 + (sy as f32 + 0.5) / 4.0 - r;
                    if dx * dx + dy * dy <= r * r {
                        n += 1;
                    }
                }
            }
            cpu[(y * ATLAS + x) as usize] = (n * 255 / 16) as u8;
        }
    }
}

impl Hud {
    /// `format`: the screen's.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Hud {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hud"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) colour: vec4<f32>,
    @location(1) uv: vec2<f32>,
};
@group(0) @binding(0) var atlas: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@vertex
fn vs(@location(0) p: vec2<f32>, @location(1) c: vec4<f32>, @location(2) uv: vec2<f32>) -> VsOut {
    var o: VsOut;
    o.pos = vec4<f32>(p, 0.0, 1.0);
    o.colour = c;
    o.uv = uv;
    return o;
}
@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    if (i.uv.x < 0.0) { return i.colour; }
    // coverage blended in linear light reads heavy for light-on-dark text
    let cov = pow(textureSampleLevel(atlas, samp, i.uv, 0.0).r, 1.3);
    return vec4<f32>(i.colour.rgb, i.colour.a * cov);
}
"#
                .into(),
            ),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hud"),
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
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hud"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: (FLOATS_PER_VERT * 4) as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4, 2 => Float32x2],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hud atlas"),
            size: wgpu::Extent3d { width: ATLAS, height: ATLAS, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hud"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas.create_view(&Default::default())),
                },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let dir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into());
        let fonts: Vec<ab_glyph::FontVec> = ["segoeui.ttf", "seguisb.ttf"]
            .iter()
            .filter_map(|f| {
                let b = std::fs::read(std::path::Path::new(&dir).join("Fonts").join(f)).ok()?;
                ab_glyph::FontVec::try_from_vec(b).ok()
            })
            .collect();
        let fonts = if fonts.len() == 2 { fonts } else { Vec::new() };
        let mut cpu = vec![0u8; (ATLAS * ATLAS) as usize];
        disc(&mut cpu);
        let capacity = 8192;
        Hud {
            pipeline,
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hud"),
                size: (capacity * FLOATS_PER_VERT * 4) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            verts: Vec::new(),
            capacity,
            hits: Vec::new(),
            atlas,
            bind,
            cpu,
            dirty: true,
            shelf: (DISC + 1, 0, DISC + 1),
            glyphs: Default::default(),
            fonts,
            sw: 1920.0,
            sh: 1080.0,
        }
    }

    /// Start a frame on a `w` x `h` pixel target.
    pub fn begin(&mut self, w: u32, h: u32) {
        self.sw = w.max(1) as f32;
        self.sh = h.max(1) as f32;
        self.clear();
    }

    /// A quad sampling the atlas (`uv` = u0 v0 u1 v1), 0..1 screen space.
    fn quad(&mut self, x: f32, y: f32, w: f32, h: f32, uv: [f32; 4], c: [f32; 4]) {
        let (x0, x1) = (x * 2.0 - 1.0, (x + w) * 2.0 - 1.0);
        let (y0, y1) = (1.0 - y * 2.0, 1.0 - (y + h) * 2.0);
        let (u0, v0, u1, v1) = (uv[0], uv[1], uv[2], uv[3]);
        for (px, py, u, v) in [(x0, y0, u0, v0), (x1, y0, u1, v0), (x1, y1, u1, v1),
                               (x0, y0, u0, v0), (x1, y1, u1, v1), (x0, y1, u0, v1)] {
            self.verts.extend_from_slice(&[px, py, c[0], c[1], c[2], c[3], u, v]);
        }
    }

    /// Snap a 0..1 x / y to the pixel grid (crisp 1 px edges).
    pub fn snap_x(&self, x: f32) -> f32 {
        (x * self.sw).round() / self.sw
    }
    pub fn snap_y(&self, y: f32) -> f32 {
        (y * self.sh).round() / self.sh
    }
    /// `n` pixels as a fraction of the width / height.
    pub fn px_x(&self, n: f32) -> f32 {
        n / self.sw
    }
    pub fn px_y(&self, n: f32) -> f32 {
        n / self.sh
    }

    /// A rectangle with corners rounded by `r` pixels, pixel-snapped.
    pub fn round_rect(&mut self, x: f32, y: f32, w: f32, h: f32, r: f32, c: [f32; 4]) {
        let (x0, y0) = (self.snap_x(x), self.snap_y(y));
        let (x1, y1) = (self.snap_x(x + w), self.snap_y(y + h));
        let r = r.min((x1 - x0) * self.sw * 0.5).min((y1 - y0) * self.sh * 0.5).max(0.0);
        if r < 0.5 {
            self.rect(x0, y0, x1 - x0, y1 - y0, c);
            return;
        }
        let (rx, ry) = (r / self.sw, r / self.sh);
        self.rect(x0 + rx, y0, x1 - x0 - 2.0 * rx, y1 - y0, c);
        self.rect(x0, y0 + ry, rx, y1 - y0 - 2.0 * ry, c);
        self.rect(x1 - rx, y0 + ry, rx, y1 - y0 - 2.0 * ry, c);
        let t = |v: f32| v / ATLAS as f32;
        let (a, m, b) = (t(0.5), t(DISC as f32 * 0.5), t(DISC as f32 - 0.5));
        self.quad(x0, y0, rx, ry, [a, a, m, m], c);
        self.quad(x1 - rx, y0, rx, ry, [m, a, b, m], c);
        self.quad(x0, y1 - ry, rx, ry, [a, m, m, b], c);
        self.quad(x1 - rx, y1 - ry, rx, ry, [m, m, b, b], c);
    }

    /// A panel: fill with a 1 px `edge`, corners `r` px.
    pub fn panel(&mut self, x: f32, y: f32, w: f32, h: f32, r: f32, fill: [f32; 4], edge: [f32; 4]) {
        self.round_rect(x, y, w, h, r, edge);
        let (px, py) = (self.px_x(1.0), self.px_y(1.0));
        self.round_rect(x + px, y + py, w - 2.0 * px, h - 2.0 * py, (r - 1.0).max(0.0), fill);
    }

    /// The font size `size` (em, a fraction of the screen height) in whole
    /// pixels.
    fn em_px(&self, size: f32) -> u16 {
        (size * self.sh).round().clamp(6.0, 200.0) as u16
    }

    fn scale(&self, wt: Weight, px: u16) -> ab_glyph::PxScale {
        use ab_glyph::Font;
        let f = &self.fonts[wt as usize];
        let upem = f.units_per_em().unwrap_or(2048.0);
        ab_glyph::PxScale::from(px as f32 * f.height_unscaled() / upem)
    }

    fn glyph(&mut self, wt: Weight, ch: char, px: u16) -> Glyph {
        use ab_glyph::Font;
        if let Some(g) = self.glyphs.get(&(wt, ch, px)) {
            return *g;
        }
        let scale = self.scale(wt, px);
        let f = &self.fonts[wt as usize];
        let id = f.glyph_id(ch);
        let mut g = Glyph { uv: [0.0; 4], w: 0.0, h: 0.0, ox: 0.0, oy: 0.0 };
        if let Some(o) = f.outline_glyph(id.with_scale_and_position(scale, ab_glyph::point(0.0, 0.0))) {
            let b = o.px_bounds();
            let (w, h) = (b.width() as u32, b.height() as u32);
            // shelf pack with a texel of gutter; a full atlas starts over
            let (mut sx, mut sy, mut sh) = self.shelf;
            if sx + w + 1 > ATLAS {
                sx = 0;
                sy += sh;
                sh = 0;
            }
            if sy + h + 1 > ATLAS {
                self.glyphs.clear();
                self.cpu.iter_mut().for_each(|v| *v = 0);
                disc(&mut self.cpu);
                (sx, sy, sh) = (DISC + 1, 0, DISC + 1);
            }
            let cpu = &mut self.cpu;
            o.draw(|x, y, c| {
                let i = ((sy + y) * ATLAS + sx + x) as usize;
                if let Some(p) = cpu.get_mut(i) {
                    *p = (c.clamp(0.0, 1.0) * 255.0) as u8;
                }
            });
            let t = |v: u32| v as f32 / ATLAS as f32;
            g = Glyph { uv: [t(sx), t(sy), t(sx + w), t(sy + h)], w: w as f32, h: h as f32, ox: b.min.x, oy: b.min.y };
            self.shelf = (sx + w + 1, sy, sh.max(h + 1));
            self.dirty = true;
        }
        self.glyphs.insert((wt, ch, px), g);
        g
    }

    /// Lay out `s`; with `c` draw it from (x, y) = the top of the line.
    /// Returns its width (a fraction of the screen width).
    fn layout(&mut self, s: &str, x: f32, y: f32, size: f32, wt: Weight, c: Option<[f32; 4]>) -> f32 {
        use ab_glyph::{Font, ScaleFont};
        if self.fonts.is_empty() {
            // no Segoe UI: the 5x7 pixel font, its caps ~ the face's
            let h = size * 0.72;
            let aspect = self.sw / self.sh;
            if let Some(c) = c {
                pixel_label(self, &s.to_uppercase(), x, y + size * 0.2, h, aspect, c);
            }
            return pixel_label_w(s, h, aspect);
        }
        let px = self.em_px(size);
        let scale = self.scale(wt, px);
        let ascent = self.fonts[wt as usize].as_scaled(scale).ascent();
        let x0 = (x * self.sw).round();
        let base = (y * self.sh).round() + ascent.round();
        let mut pen = x0;
        let mut prev = None;
        for ch in s.chars() {
            let (adv, kern, id) = {
                let f = self.fonts[wt as usize].as_scaled(scale);
                let id = f.glyph_id(ch);
                (f.h_advance(id), prev.map_or(0.0, |p| f.kern(p, id)), id)
            };
            pen += kern;
            if let Some(c) = c {
                let g = self.glyph(wt, ch, px);
                if g.w > 0.0 {
                    let gx = (pen + g.ox).round();
                    let gy = base + g.oy;
                    self.quad(gx / self.sw, gy / self.sh, g.w / self.sw, g.h / self.sh, g.uv, c);
                }
            }
            pen += adv;
            prev = Some(id);
        }
        (pen - x0) / self.sw
    }

    /// Text in ACECM's face from (x, y) = the top of the line; `size` = the
    /// font size as a fraction of the screen height. Returns its width.
    pub fn label(&mut self, s: &str, x: f32, y: f32, size: f32, wt: Weight, c: [f32; 4]) -> f32 {
        self.layout(s, x, y, size, wt, Some(c))
    }

    /// The width `label` would draw.
    pub fn label_w(&mut self, s: &str, size: f32, wt: Weight) -> f32 {
        self.layout(s, 0.0, 0.0, size, wt, None)
    }

    /// The height of a line of `size` text (ascent to descent).
    pub fn line_h(&self, size: f32) -> f32 {
        size * 1.33
    }

    pub fn clear(&mut self) {
        self.verts.clear();
        self.hits.clear();
    }

    /// The action under a cursor given in 0..1 screen space, if any.
    pub fn hit(&self, x: f32, y: f32) -> Option<Action> {
        for &(bx, by, bw, bh, a) in &self.hits {
            if x >= bx && x <= bx + bw && y >= by && y <= by + bh {
                // The timeline reports where along it the click landed.
                return Some(match a {
                    Action::Seek(_) => Action::Seek(((x - bx) / bw).clamp(0.0, 1.0)),
                    other => other,
                });
            }
        }
        None
    }

    /// A triangle, for the transport icons. Points are 0..1 screen space.
    pub fn tri(&mut self, p: [(f32, f32); 3], c: [f32; 4]) {
        for (x, y) in p {
            self.verts
                .extend_from_slice(&[x * 2.0 - 1.0, 1.0 - y * 2.0, c[0], c[1], c[2], c[3], -1.0, 0.0]);
        }
    }

    /// A clickable button with an icon drawn inside it.
    ///
    /// `hot` highlights it (the cursor is over it), `on` marks it active.
    /// ⚠ `bw`/`bh` are both screen FRACTIONS, and the screen is wider than it
    /// is tall, so equal values do not make a square. The caller scales height
    /// by the aspect ratio.
    pub fn button(&mut self, x: f32, y: f32, bw: f32, bh: f32, a: Action,
                  hot: bool, on: bool) {
        let bg = if hot { 0.34 } else { 0.18 };
        self.rect(x, y, bw, bh, [1.0, 1.0, 1.0, bg]);
        let fg = if on { [1.0, 0.85, 0.25, 0.95] } else { [1.0, 1.0, 1.0, 0.9] };
        // Icons are laid out in the middle 56% of the button.
        let (ix, iy, w, h) = (x + bw * 0.22, y + bh * 0.22, bw * 0.56, bh * 0.56);
        let bar = w * 0.26;
        match a {
            Action::PlayPause if on => {
                // Pause: two bars, shown while playing.
                self.rect(ix, iy, bar, h, fg);
                self.rect(ix + w - bar, iy, bar, h, fg);
            }
            Action::PlayPause => {
                self.tri([(ix, iy), (ix + w, iy + h * 0.5), (ix, iy + h)], fg);
            }
            Action::Slower | Action::PrevCar => {
                self.tri([(ix + w, iy), (ix, iy + h * 0.5), (ix + w, iy + h)], fg);
            }
            Action::Faster | Action::NextCar => {
                self.tri([(ix, iy), (ix + w, iy + h * 0.5), (ix, iy + h)], fg);
            }
            Action::Restart => {
                self.rect(ix, iy, bar, h, fg);
                self.tri([(ix + w, iy), (ix + bar, iy + h * 0.5), (ix + w, iy + h)], fg);
            }
            Action::Camera => {
                self.rect(ix, iy + h * 0.22, w, h * 0.56, fg);
                self.rect(ix + w * 0.3, iy, w * 0.3, h * 0.22, fg);
            }
            Action::Seek(_) => {}
        }
        self.hits.push((x, y, bw, bh, a));
    }

    /// Register a clickable strip without drawing anything - the timeline
    /// already draws itself as a bar.
    pub fn hotspot(&mut self, x: f32, y: f32, w: f32, h: f32, a: Action) {
        self.hits.push((x, y, w, h, a));
    }

    /// A rectangle, in a 0..1 screen space with the origin top-left.
    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, c: [f32; 4]) {
        let (x0, x1) = (x * 2.0 - 1.0, (x + w) * 2.0 - 1.0);
        let (y0, y1) = (1.0 - y * 2.0, 1.0 - (y + h) * 2.0);
        for (px, py) in [(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)] {
            self.verts.extend_from_slice(&[px, py, c[0], c[1], c[2], c[3], -1.0, 0.0]);
        }
    }

    /// A left-to-right filled bar with a dim background.
    pub fn bar(&mut self, x: f32, y: f32, w: f32, h: f32, v: f32, c: [f32; 4]) {
        self.rect(x, y, w, h, [1.0, 1.0, 1.0, 0.12]);
        self.rect(x, y, w * v.clamp(0.0, 1.0), h, c);
    }

    /// One seven-segment digit. `w`/`h` are the digit box; `t` the stroke.
    pub fn digit(&mut self, d: u8, x: f32, y: f32, w: f32, h: f32, t: f32, c: [f32; 4]) {
        self.segments(DIGITS[(d % 10) as usize], x, y, w, h, (t, t), c);
    }

    /// Digits, '.', '-' and the few letters seven segments can spell
    /// (A b c d E F L o P S t), left to right from `x`. Returns the right
    /// edge. `t` = stroke (width, height): equal fractions of a wide screen
    /// are not equal pixels.
    pub fn text(&mut self, s: &str, mut x: f32, y: f32, w: f32, h: f32, t: (f32, f32), c: [f32; 4]) -> f32 {
        let gap = w * 0.28;
        for ch in s.chars() {
            let seg = match ch {
                '0'..='9' => DIGITS[ch as usize - '0' as usize],
                'A' => 0b1111110,
                'b' => 0b0101111,
                'c' => 0b0001101,
                'd' => 0b0011111,
                'E' => 0b1101101,
                'F' => 0b1101100,
                'L' => 0b0100101,
                'o' => 0b0001111,
                'P' => 0b1111100,
                'S' => 0b1101011,
                't' => 0b0101101,
                '-' => 0b0001000,
                '.' => {
                    self.rect(x - gap * 0.5, y + h - t.1, t.0, t.1, c);
                    x += t.0 + gap * 0.5;
                    continue;
                }
                _ => 0,
            };
            self.segments(seg, x, y, w, h, t, c);
            x += w + gap;
        }
        x - gap
    }

    fn segments(&mut self, seg: u8, x: f32, y: f32, w: f32, h: f32, t: (f32, f32), c: [f32; 4]) {
        let (tx, ty) = t;
        let mid = y + h * 0.5 - ty * 0.5;
        let on = |i: u8| seg & (1 << (6 - i)) != 0;
        if on(0) { self.rect(x, y, w, ty, c); }
        if on(1) { self.rect(x, y, tx, h * 0.5, c); }
        if on(2) { self.rect(x + w - tx, y, tx, h * 0.5, c); }
        if on(3) { self.rect(x, mid, w, ty, c); }
        if on(4) { self.rect(x, mid, tx, h * 0.5, c); }
        if on(5) { self.rect(x + w - tx, mid, tx, h * 0.5, c); }
        if on(6) { self.rect(x, y + h - ty, w, ty, c); }
    }

    /// A whole number, right-aligned so it does not jitter as it changes width.
    pub fn number(&mut self, mut n: u32, right: f32, y: f32, w: f32, h: f32,
                  t: f32, c: [f32; 4]) {
        let gap = w * 0.28;
        let mut x = right - w;
        loop {
            self.digit((n % 10) as u8, x, y, w, h, t, c);
            n /= 10;
            if n == 0 {
                break;
            }
            x -= w + gap;
        }
    }

    /// Push this frame's geometry to the GPU. Split from `render` so the
    /// mutable upload and the borrow the render pass holds do not overlap.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
        if self.dirty {
            self.dirty = false;
            queue.write_texture(
                wgpu::ImageCopyTexture { texture: &self.atlas, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &self.cpu,
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(ATLAS), rows_per_image: Some(ATLAS) },
                wgpu::Extent3d { width: ATLAS, height: ATLAS, depth_or_array_layers: 1 },
            );
        }
        let n = self.verts.len() / FLOATS_PER_VERT;
        if n == 0 {
            return 0;
        }
        if n > self.capacity {
            self.capacity = n.next_power_of_two();
            self.buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("hud"),
                contents: bytemuck::cast_slice(&self.verts),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.verts));
        }
        n as u32
    }

    /// Over the finished frame `view`, in a pass of its own.
    pub fn overlay(&self, enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, n: u32) {
        if n == 0 {
            return;
        }
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hud"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.set_vertex_buffer(0, self.buffer.slice(..));
        pass.draw(0..n, 0..1);
    }
}

// --------------------------------------------------------------- 5x7 font --
// The fallback when Windows' Segoe UI is missing. Pixels are hud rects,
// one per lit run of a glyph row.

fn glyph(ch: char) -> [u8; 7] {
    const G: &[(char, [u8; 7])] = &[
        ('A', [14, 17, 17, 31, 17, 17, 17]), ('B', [30, 17, 17, 30, 17, 17, 30]), ('C', [14, 17, 16, 16, 16, 17, 14]),
        ('D', [28, 18, 17, 17, 17, 18, 28]), ('E', [31, 16, 16, 30, 16, 16, 31]), ('F', [31, 16, 16, 30, 16, 16, 16]),
        ('G', [14, 17, 16, 23, 17, 17, 15]), ('H', [17, 17, 17, 31, 17, 17, 17]), ('I', [14, 4, 4, 4, 4, 4, 14]),
        ('J', [7, 2, 2, 2, 2, 18, 12]), ('K', [17, 18, 20, 24, 20, 18, 17]), ('L', [16, 16, 16, 16, 16, 16, 31]),
        ('M', [17, 27, 21, 21, 17, 17, 17]), ('N', [17, 17, 25, 21, 19, 17, 17]), ('O', [14, 17, 17, 17, 17, 17, 14]),
        ('P', [30, 17, 17, 30, 16, 16, 16]), ('Q', [14, 17, 17, 17, 21, 18, 13]), ('R', [30, 17, 17, 30, 20, 18, 17]),
        ('S', [15, 16, 16, 14, 1, 1, 30]), ('T', [31, 4, 4, 4, 4, 4, 4]), ('U', [17, 17, 17, 17, 17, 17, 14]),
        ('V', [17, 17, 17, 17, 17, 10, 4]), ('W', [17, 17, 17, 21, 21, 21, 10]), ('X', [17, 17, 10, 4, 10, 17, 17]),
        ('Y', [17, 17, 10, 4, 4, 4, 4]), ('Z', [31, 1, 2, 4, 8, 16, 31]),
        ('0', [14, 17, 19, 21, 25, 17, 14]), ('1', [4, 12, 4, 4, 4, 4, 14]), ('2', [14, 17, 1, 2, 4, 8, 31]),
        ('3', [31, 2, 4, 2, 1, 17, 14]), ('4', [2, 6, 10, 18, 31, 2, 2]), ('5', [31, 16, 30, 1, 1, 17, 14]),
        ('6', [6, 8, 16, 30, 17, 17, 14]), ('7', [31, 1, 2, 4, 8, 8, 8]), ('8', [14, 17, 17, 14, 17, 17, 14]),
        ('9', [14, 17, 17, 15, 1, 2, 12]),
        ('.', [0, 0, 0, 0, 0, 12, 12]), (':', [0, 12, 12, 0, 12, 12, 0]), ('-', [0, 0, 0, 31, 0, 0, 0]),
        ('+', [0, 4, 4, 31, 4, 4, 0]), ('/', [1, 2, 2, 4, 8, 8, 16]), ('%', [24, 25, 2, 4, 8, 19, 3]),
        ('<', [2, 4, 8, 16, 8, 4, 2]), ('>', [8, 4, 2, 1, 2, 4, 8]), ('(', [2, 4, 8, 8, 8, 4, 2]),
        (')', [8, 4, 2, 2, 2, 4, 8]), ('=', [0, 0, 31, 0, 31, 0, 0]), ('!', [4, 4, 4, 4, 4, 0, 4]),
        ('?', [14, 17, 1, 2, 4, 0, 4]), (',', [0, 0, 0, 0, 12, 4, 8]),
    ];
    let ch = ch.to_ascii_uppercase();
    G.iter().find(|g| g.0 == ch).map_or([0; 7], |g| g.1)
}

/// Width of `s` at cell height `h` (screen fractions; `aspect` = w / h).
fn pixel_label_w(s: &str, h: f32, aspect: f32) -> f32 {
    let px = h / 7.0 / aspect;
    (s.chars().count() as f32 * 6.0 - 1.0).max(0.0) * px
}

/// `s` from (x, y) top-left, cell height `h`, square pixels.
fn pixel_label(hud: &mut Hud, s: &str, x: f32, y: f32, h: f32, aspect: f32, c: [f32; 4]) {
    let py = h / 7.0;
    let px = py / aspect;
    for (i, ch) in s.chars().enumerate() {
        let g = glyph(ch);
        let x0 = x + i as f32 * 6.0 * px;
        for (row, bits) in g.iter().enumerate() {
            let mut col = 0;
            while col < 5 {
                if bits & (16 >> col) != 0 {
                    let start = col;
                    while col < 5 && bits & (16 >> col) != 0 {
                        col += 1;
                    }
                    hud.rect(x0 + start as f32 * px, y + row as f32 * py, (col - start) as f32 * px, py, c);
                } else {
                    col += 1;
                }
            }
        }
    }
}
