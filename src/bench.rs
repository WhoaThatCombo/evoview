//! `--bench <seconds>`: render a fixed camera orbit, then print frame and
//! load statistics and exit.
//!
//! It drives the REAL interactive renderer (render::run), not a copy, so any
//! pass a feature adds there is measured automatically. Bench mode only
//! changes: the window size, the present mode (no vsync), the camera (a fixed
//! orbit), and it asks for timestamp queries when the adapter has them.
//!
//! The last line printed is `BENCH_JSON {...}` for tools/bench.py.

use std::time::Instant;

pub struct Opts {
    pub seconds: f32,
    pub width: u32,
    pub height: u32,
    pub name: String,
    /// `--bench-shot out.png`: save the FIRST frame (camera at its start
    /// angle), to check that a speed-up did not change the picture.
    pub shot: Option<std::path::PathBuf>,
}

/// Warm-up excluded from the statistics: pipeline first-use, residency.
const WARMUP_S: f32 = 1.0;
/// Readback slots for timestamp results, so the CPU never waits on the GPU.
const SLOTS: usize = 4;
/// Timestamps per frame: begin, up to MARKS - 2 `mark`s, end.
const MARKS: usize = 32;

struct Slot {
    readback: wgpu::Buffer,
    /// Some(rx) while a map is in flight.
    pending: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    /// timestamps written into it
    n: usize,
    /// each mark written into it, in the order the GPU runs them: (its
    /// timestamp's index in the slot, label). A segment runs from the mark
    /// before it (or the frame's start, index 0) to its own.
    labels: Vec<(usize, &'static str)>,
}

struct Gpu {
    set: std::sync::Arc<wgpu::QuerySet>,
    resolve: wgpu::Buffer,
    slots: Vec<Slot>,
    period_ns: f32,
    /// slot written this frame, if any
    current: Option<usize>,
}

/// The frame being encoded, for `mark` on any thread: the passes are
/// recorded on several threads, each into its own encoder (render.rs), so a
/// mark takes the next free timestamp wherever it is called and remembers
/// it per thread; the render thread strings the threads' marks together in
/// the order their command buffers are submitted (`marks_take` /
/// `marks_append`).
static TS: std::sync::Mutex<Option<(std::sync::Arc<wgpu::QuerySet>, usize)>> = std::sync::Mutex::new(None);
/// timestamps written this frame so far
static WRITTEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// this frame's marks in GPU order (see `Slot::labels`)
static ORDER: std::sync::Mutex<Vec<(usize, &'static str)>> = std::sync::Mutex::new(Vec::new());

thread_local! {
    /// marks this thread wrote and has not handed over yet
    static PENDING: std::cell::RefCell<Vec<(usize, &'static str)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The marks this thread wrote since its last take, in order: a worker
/// returns them with its command buffer.
pub fn marks_take() -> Vec<(usize, &'static str)> {
    PENDING.with(|p| std::mem::take(&mut *p.borrow_mut()))
}

/// Marks of a command buffer submitted after everything appended so far.
pub fn marks_append(m: Vec<(usize, &'static str)>) {
    if !m.is_empty() {
        if let Ok(mut o) = ORDER.lock() {
            o.extend(m);
        }
    }
}

/// This thread's marks so far go before whatever is appended next.
pub fn marks_flush() {
    marks_append(marks_take());
}

thread_local! {
    /// The bench's timestamp state while a frame is being encoded, so a pass
    /// inside another module (taa.rs, ...) can `bench::mark` its own steps.
    static GPU: std::cell::RefCell<Option<Gpu>> = const { std::cell::RefCell::new(None) };
}

/// Track draw calls and triangles submitted per view (track.rs
/// draw_phase_view: 0 main, 1 mirror, 2 probe, 3 rear camera), summed over
/// the run: a count the GPU sharing with a game cannot blur.
static DRAWS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
static TRIS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

pub fn count_draws(view: usize, draws: u64, tris: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if let (Some(d), Some(t)) = (DRAWS.get(view), TRIS.get(view)) {
        d.fetch_add(draws, Relaxed);
        t.fetch_add(tris, Relaxed);
    }
}

/// Things counted per frame under a name (`count("car draws main", n)`,
/// once per pass - not per draw), printed per frame.
static COUNTS: std::sync::Mutex<Vec<(&'static str, u64)>> = std::sync::Mutex::new(Vec::new());

pub fn count(label: &'static str, n: u64) {
    if let Ok(mut v) = COUNTS.lock() {
        match v.iter_mut().find(|e| e.0 == label) {
            Some(e) => e.1 += n,
            None => v.push((label, n)),
        }
    }
}

/// CPU time spent in named scopes over the run (`let _t = cpu("x");`),
/// printed per frame. Cheap enough to leave in: one Instant per scope.
static CPU: std::sync::Mutex<Vec<(&'static str, f64)>> = std::sync::Mutex::new(Vec::new());

pub struct CpuScope(&'static str, Instant);

pub fn cpu(label: &'static str) -> CpuScope {
    CpuScope(label, Instant::now())
}

impl Drop for CpuScope {
    fn drop(&mut self) {
        let ms = self.1.elapsed().as_secs_f64() * 1e3;
        if let Ok(mut v) = CPU.lock() {
            match v.iter_mut().find(|e| e.0 == self.0) {
                Some(e) => e.1 += ms,
                None => v.push((self.0, ms)),
            }
        }
    }
}

/// EVOVIEW_BENCH_HITCH=ms: each frame slower than that is printed with the
/// CPU scopes that grew most in it (what a hitch was spent on).
fn hitch(frame: u64, t: f32, ms: f64) {
    static LIMIT: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    static PREV: std::sync::Mutex<Vec<(&'static str, f64)>> = std::sync::Mutex::new(Vec::new());
    let Some(limit) = *LIMIT.get_or_init(|| crate::envc::var("EVOVIEW_BENCH_HITCH").ok().and_then(|v| v.parse().ok())) else { return };
    let now = CPU.lock().map(|v| v.clone()).unwrap_or_default();
    let mut prev = PREV.lock().unwrap();
    if ms > limit {
        let mut d: Vec<(&str, f64)> = now.iter()
            .map(|(l, v)| (*l, v - prev.iter().find(|p| p.0 == *l).map_or(0.0, |p| p.1)))
            .filter(|x| x.1 > 1.0).collect();
        d.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top: Vec<String> = d.iter().take(16).map(|(l, v)| format!("{l} {v:.1}")).collect();
        println!("hitch: frame {frame} at {t:.2} s: {ms:.1} ms [{}]", top.join(", "));
    }
    *prev = now;
}

/// GPU time since the previous mark (or the frame start) is reported as
/// `label`. A no-op outside --bench or without timestamp queries. Labels
/// are matched by name, so a mark may be skipped on some frames.
pub fn mark(enc: &mut wgpu::CommandEncoder, label: &'static str) {
    use std::sync::atomic::Ordering::Relaxed;
    let Ok(ts) = TS.lock() else { return };
    let Some((set, i)) = ts.as_ref() else { return };
    let k = WRITTEN.fetch_add(1, Relaxed);
    if k >= MARKS - 1 {
        WRITTEN.fetch_sub(1, Relaxed);
        return;
    }
    enc.write_timestamp(set, (i * MARKS + k) as u32);
    PENDING.with(|p| p.borrow_mut().push((k, label)));
}

/// Drop the timestamp queries and readback buffers while the device is
/// alive: left in the thread-local they are freed by the TLS destructors at
/// process exit, after the Vulkan driver is gone (segfault, exit code 139).
pub fn release() {
    let _ = GPU.try_with(|g| g.borrow_mut().take());
    if let Ok(mut ts) = TS.lock() {
        *ts = None;
    }
}

impl Drop for Bench {
    fn drop(&mut self) {
        release();
    }
}

pub struct Bench {
    pub opts: Opts,
    start: Option<Instant>,
    last: Option<Instant>,
    cpu_ms: Vec<f64>,
    gpu_ms: Vec<f64>,
    /// per `mark` segment: (label, ms samples)
    seg_ms: Vec<(&'static str, Vec<f64>)>,
    vram_peak: u64,
    vram_t: Option<Instant>,
    pub load_s: f64,
    pub info: Vec<(&'static str, f64)>,
    /// heaviest meshes by instanced triangles, printed (not in the JSON)
    pub top: Vec<(f64, String)>,
    frame: u64,
    in_flight: std::collections::VecDeque<wgpu::SubmissionIndex>,
    readback: Option<(wgpu::Buffer, u32, u32, u32, bool)>,
    shot_index: Option<u64>,
}

/// Features bench mode would like. Normal runs request none of these.
pub fn features(adapter: &wgpu::Adapter) -> wgpu::Features {
    let want = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    if adapter.features().contains(want) {
        want
    } else {
        wgpu::Features::empty()
    }
}

/// Uncapped presentation, so frame time measures the renderer, not vsync.
pub fn present_mode(caps: &wgpu::SurfaceCapabilities) -> wgpu::PresentMode {
    [wgpu::PresentMode::Immediate, wgpu::PresentMode::Mailbox]
        .into_iter()
        .find(|m| caps.present_modes.contains(m))
        .unwrap_or(wgpu::PresentMode::Fifo)
}

impl Bench {
    pub fn new(opts: Opts, device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let gpu = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
            .then(|| {
                let size = SLOTS as u64 * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;
                Gpu {
                    set: std::sync::Arc::new(device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("bench ts"),
                        ty: wgpu::QueryType::Timestamp,
                        count: (SLOTS * MARKS) as u32,
                    })),
                    resolve: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("bench resolve"),
                        size,
                        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                    slots: (0..SLOTS)
                        .map(|_| Slot {
                            readback: device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("bench readback"),
                                size: (MARKS * 8) as u64,
                                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                                mapped_at_creation: false,
                            }),
                            pending: None,
                            n: 0,
                            labels: Vec::new(),
                        })
                        .collect(),
                    period_ns: queue.get_timestamp_period(),
                    current: None,
                }
            });
        println!(
            "bench: {}s at {}x{}, GPU timing {}",
            opts.seconds, opts.width, opts.height,
            if gpu.is_some() { "via timestamp queries" } else { "unavailable (CPU frame time only)" });
        GPU.with(|g| *g.borrow_mut() = gpu);
        Bench {
            opts,
            start: None,
            last: None,
            cpu_ms: Vec::new(),
            gpu_ms: Vec::new(),
            seg_ms: Vec::new(),
            vram_peak: 0,
            vram_t: None,
            load_s: 0.0,
            info: Vec::new(),
            top: Vec::new(),
            frame: 0,
            in_flight: Default::default(),
            readback: None,
            shot_index: None,
        }
    }

    /// Seconds since the first frame; drives the camera.
    pub fn t(&self) -> f32 {
        // frame 0 is the --bench-shot frame; pin it to the start angle
        if self.frame == 0 {
            return 0.0;
        }
        self.start.map_or(0.0, |s| s.elapsed().as_secs_f32())
    }

    /// The orbit angle at time t: one full turn every 8 s.
    pub fn yaw(&self, base: f32) -> f32 {
        // EVOVIEW_BENCH_YAW_STEP=deg: that many degrees a frame instead, so
        // two runs reach the same angle at the same frame (A/B captures in
        // motion, whatever the frame rate)
        if let Some(d) = crate::envc::var("EVOVIEW_BENCH_YAW_STEP").ok().and_then(|v| v.parse::<f32>().ok()) {
            return base + (self.frame as f32 * d).to_radians();
        }
        base + self.t() * std::f32::consts::TAU / 8.0
    }

    /// Call right after the frame's command encoder is created.
    pub fn begin(&mut self, enc: &mut wgpu::CommandEncoder) {
        let now = Instant::now();
        if self.start.is_none() {
            self.start = Some(now);
            // time to first frame: process start to the first encoder
            self.load_s = crate::perf::since_start();
            crate::perf::stamp("first frame");
        }
        if let Some(l) = self.last {
            if self.t() > WARMUP_S {
                let ms = now.duration_since(l).as_secs_f64() * 1e3;
                self.cpu_ms.push(ms);
                hitch(self.frame, self.t(), ms);
            }
        }
        self.last = Some(now);
        GPU.with(|g| {
            let mut g = g.borrow_mut();
            let Some(g) = g.as_mut() else { return };
            g.current = g.slots.iter().position(|s| s.pending.is_none());
            let _ = marks_take();
            if let Ok(mut o) = ORDER.lock() {
                o.clear();
            }
            if let Some(i) = g.current {
                enc.write_timestamp(&g.set, (i * MARKS) as u32);
                g.slots[i].labels.clear();
                WRITTEN.store(1, std::sync::atomic::Ordering::Relaxed);
            }
            if let Ok(mut ts) = TS.lock() {
                *ts = g.current.map(|i| (g.set.clone(), i));
            }
        });
    }

    /// GPU time since the previous mark (or the frame start) is reported
    /// as `label`. Call between passes.
    pub fn mark(&mut self, enc: &mut wgpu::CommandEncoder, label: &'static str) {
        mark(enc, label);
    }

    /// Call just before `enc.finish()`.
    pub fn end(&mut self, enc: &mut wgpu::CommandEncoder, device: &wgpu::Device, tex: &wgpu::Texture) {
        // EVOVIEW_BENCH_SHOT_FRAME=N: save frame N instead of the first (TAA
        // and animation only show after some frames)
        let at = crate::envc::var("EVOVIEW_BENCH_SHOT_FRAME").ok().and_then(|v| v.parse().ok()).unwrap_or(0u64);
        // EVOVIEW_BENCH_SHOT_COUNT=K: K frames in a row from there, saved as
        // <shot>_0.png .. (flicker checks)
        let count = crate::envc::var("EVOVIEW_BENCH_SHOT_COUNT").ok().and_then(|v| v.parse().ok()).unwrap_or(1u64);
        if self.frame >= at && self.frame < at + count && self.opts.shot.is_some() {
            self.shot_index = if count > 1 { Some(self.frame - at) } else { None };
            let (w, h) = (tex.width(), tex.height());
            let padded = (w * 4).div_ceil(256) * 256;
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bench shot"),
                size: (padded * h) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            enc.copy_texture_to_buffer(
                tex.as_image_copy(),
                wgpu::ImageCopyBuffer {
                    buffer: &buf,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(padded),
                        rows_per_image: Some(h),
                    },
                },
                tex.size(),
            );
            let bgra = matches!(tex.format(),
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
            self.readback = Some((buf, w, h, padded, bgra));
        }
        self.frame += 1;
        if matches!(self.frame, 1 | 2 | 10 | 100) {
            crate::perf::vram_note(&format!("frame {}", self.frame));
        }
        GPU.with(|g| {
            let mut g = g.borrow_mut();
            let Some(g) = g.as_mut() else { return };
            let Some(i) = g.current else { return };
            // no mark after this one: the frame's timestamps are counted
            if let Ok(mut ts) = TS.lock() {
                *ts = None;
            }
            marks_flush();
            if let Ok(mut o) = ORDER.lock() {
                g.slots[i].labels = std::mem::take(&mut *o);
            }
            let n = WRITTEN.load(std::sync::atomic::Ordering::Relaxed);
            enc.write_timestamp(&g.set, (i * MARKS + n) as u32);
            let q = (i * MARKS) as u32;
            let off = i as u64 * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;
            enc.resolve_query_set(&g.set, q..q + n as u32 + 1, &g.resolve, off);
            enc.copy_buffer_to_buffer(&g.resolve, off, &g.slots[i].readback, 0, (n as u64 + 1) * 8);
            g.slots[i].n = n + 1;
        });
    }

    /// Writes the --bench-shot PNG once its frame is submitted. Blocks once.
    pub fn after_present(&mut self, device: &wgpu::Device) {
        let Some((buf, w, h, padded, bgra)) = self.readback.take() else { return };
        let Some(mut path) = self.opts.shot.clone() else { return };
        if let Some(k) = self.shot_index {
            let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            path.set_file_name(format!("{stem}_{k}.png"));
        }
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let mut px = Vec::with_capacity((w * h * 4) as usize);
        for row in 0..h as usize {
            let r = &data[row * padded as usize..row * padded as usize + w as usize * 4];
            for p in r.chunks_exact(4) {
                if bgra {
                    px.extend_from_slice(&[p[2], p[1], p[0], 255]);
                } else {
                    px.extend_from_slice(&[p[0], p[1], p[2], 255]);
                }
            }
        }
        drop(data);
        buf.unmap();
        let ok = std::fs::File::create(&path).ok().and_then(|f| {
            let mut e = png::Encoder::new(std::io::BufWriter::new(f), w, h);
            e.set_color(png::ColorType::Rgba);
            e.set_depth(png::BitDepth::Eight);
            e.write_header().ok()?.write_image_data(&px).ok()
        });
        match ok {
            Some(()) => println!("bench shot: {}", path.display()),
            None => eprintln!("bench shot: cannot write {}", path.display()),
        }
        // the blocking readback must not count as a frame
        self.last = Some(Instant::now());
    }

    /// Call after `queue.submit`. Returns true once the run is over.
    ///
    /// Offscreen there is no swapchain to hold the CPU back, so it would run
    /// unboundedly ahead of the GPU and pile up command buffers (GBs within
    /// seconds). At most two frames stay in flight, like a no-vsync swapchain.
    pub fn after_submit(&mut self, device: &wgpu::Device, submitted: wgpu::SubmissionIndex) -> bool {
        self.in_flight.push_back(submitted);
        let wait = cpu("gpu wait");
        while self.in_flight.len() > 2 {
            let oldest = self.in_flight.pop_front().unwrap();
            device.poll(wgpu::Maintain::WaitForSubmissionIndex(oldest));
        }
        drop(wait);
        let warm = self.start.map_or(false, |s| s.elapsed().as_secs_f32() > WARMUP_S);
        let mut done_frames: Vec<(f64, Vec<(&'static str, f64)>)> = Vec::new();
        GPU.with(|g| {
            let mut g = g.borrow_mut();
            let Some(g) = g.as_mut() else { return };
            if let Some(i) = g.current.take() {
                let (tx, rx) = std::sync::mpsc::channel();
                g.slots[i]
                    .readback
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| {
                        let _ = tx.send(r);
                    });
                g.slots[i].pending = Some(rx);
            }
            device.poll(wgpu::Maintain::Poll);
            let period = g.period_ns as f64;
            for s in g.slots.iter_mut() {
                let done = match s.pending.as_ref().map(|rx| rx.try_recv()) {
                    Some(Ok(r)) => Some(r.is_ok()),
                    _ => None,
                };
                if let Some(ok) = done {
                    if ok {
                        let ts: Vec<u64> = {
                            let d = s.readback.slice(..).get_mapped_range();
                            (0..s.n).map(|k| u64::from_le_bytes(d[k * 8..k * 8 + 8].try_into().unwrap())).collect()
                        };
                        s.readback.unmap();
                        let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * period / 1e6;
                        if warm && ts.len() >= 2 && ts[ts.len() - 1] > ts[0] {
                            // each segment ends at its mark; the last runs to the end ("rest")
                            let mut segs = Vec::with_capacity(s.labels.len() + 1);
                            let mut at = 0;
                            for &(k, l) in s.labels.iter().filter(|m| m.0 < ts.len()) {
                                segs.push((l, ms(ts[at], ts[k])));
                                at = k;
                            }
                            segs.push(("rest", ms(ts[at], ts[ts.len() - 1])));
                            done_frames.push((ms(ts[0], ts[ts.len() - 1]), segs));
                        }
                    }
                    s.pending = None;
                }
            }
        });
        for (total, segs) in done_frames {
            self.gpu_ms.push(total);
            // per frame, summed by label (a label may appear more than once)
            let mut frame: Vec<(&'static str, f64)> = Vec::new();
            for (l, v) in segs {
                match frame.iter_mut().find(|f| f.0 == l) {
                    Some(f) => f.1 += v,
                    None => frame.push((l, v)),
                }
            }
            for (l, v) in frame {
                match self.seg_ms.iter_mut().find(|f| f.0 == l) {
                    Some(f) => f.1.push(v),
                    None => self.seg_ms.push((l, vec![v])),
                }
            }
        }
        // The OS counter costs ~1 ms; sample it once a second, not per frame.
        if self.vram_t.map_or(true, |t| t.elapsed().as_secs_f32() >= 1.0) {
            self.vram_t = Some(Instant::now());
            if let Some(v) = crate::perf::vram_bytes() {
                self.vram_peak = self.vram_peak.max(v);
            }
        }
        // a frame sequence (EVOVIEW_BENCH_SHOT_COUNT > 1) ends the run once saved
        let seq = crate::envc::var("EVOVIEW_BENCH_SHOT_COUNT").ok().and_then(|v| v.parse::<u64>().ok()).filter(|&c| c > 1).map(|c| {
            c + crate::envc::var("EVOVIEW_BENCH_SHOT_FRAME").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0)
        });
        self.t() >= self.opts.seconds + WARMUP_S || seq.map_or(false, |end| self.frame > end + 2)
    }

    pub fn report(&self) {
        let stats = |v: &[f64]| -> Option<(f64, f64, f64)> {
            if v.is_empty() {
                return None;
            }
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = s.iter().sum::<f64>() / s.len() as f64;
            let pct = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
            Some((mean, pct(0.5), pct(0.99)))
        };
        let (ws, commit, ws_now) = crate::perf::ram();
        let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
        println!("---- bench: {} ----", self.opts.name);
        println!("load: {:.2}s wall", self.load_s);
        for (n, s, c) in crate::perf::phases() {
            println!("  {n:24} {:8.1} ms  ({c} call(s))", s * 1e3);
        }
        for (n, b) in crate::perf::bytes() {
            println!("  {n:24} {:8.1} MB", b as f64 / 1e6);
        }
        let line = |label: &str, v: Option<(f64, f64, f64)>| match v {
            Some((m, p50, p99)) => println!(
                "{label}: mean {m:.3} ms  p50 {p50:.3}  p99 {p99:.3}"),
            None => println!("{label}: n/a"),
        };
        line(&format!("cpu frame ({} frames)", self.cpu_ms.len()), stats(&self.cpu_ms));
        // hitches (late joins, streaming): the worst frame and how many missed 30 fps
        let worst = self.cpu_ms.iter().copied().fold(0.0f64, f64::max);
        let over = |ms: f64| self.cpu_ms.iter().filter(|&&v| v > ms).count();
        println!("cpu frame worst {worst:.1} ms; {} frame(s) > 33 ms, {} > 50 ms, {} > 100 ms", over(33.3), over(50.0), over(100.0));
        line(&format!("gpu frame ({} frames)", self.gpu_ms.len()), stats(&self.gpu_ms));
        let nf = self.gpu_ms.len().max(1) as f64;
        for (l, v) in &self.seg_ms {
            // per frame: a pass that skips frames counts them as 0
            let pf = v.iter().sum::<f64>() / nf;
            match stats(v) {
                Some((m, p50, p99)) => println!("  gpu {l:>14}: per frame {pf:.3} ms  (when run: mean {m:.3}  p50 {p50:.3}  p99 {p99:.3}, {} frames)", v.len()),
                None => println!("  gpu {l:>14}: n/a"),
            }
        }
        if let Ok(v) = CPU.lock() {
            let frames = self.frame.max(1) as f64;
            for (l, ms) in v.iter() {
                println!("  cpu {l:>14}: per frame {:.3} ms", ms / frames);
            }
        }
        {
            use std::sync::atomic::Ordering::Relaxed;
            let frames = self.frame.max(1) as f64;
            for (k, name) in ["main", "mirror", "probe", "rear cam"].iter().enumerate() {
                let d = DRAWS[k].load(Relaxed) as f64 / frames;
                let t = TRIS[k].load(Relaxed) as f64 / frames;
                if d > 0.0 {
                    println!("  track draws {name:>9}: {d:8.0} draws/frame  {:8.2} Mtri/frame", t / 1e6);
                }
            }
        }
        if let Ok(v) = COUNTS.lock() {
            let frames = self.frame.max(1) as f64;
            for (l, n) in v.iter() {
                println!("  count {l:>18}: per frame {:.1}", *n as f64 / frames);
            }
        }
        println!("vram: {:.0} MB   peak RAM: {:.0} MB working set, {:.0} MB commit; {:.0} MB working set while rendering",
                 mb(self.vram_peak), mb(ws), mb(commit), mb(ws_now));
        for (k, v) in &self.info {
            println!("  {k}: {v}");
        }
        println!("heaviest meshes:");
        for (t, n) in &self.top {
            println!("  {:7.2} Mtri  {n}", t / 1e6);
        }

        let js = |v: Option<(f64, f64, f64)>| match v {
            Some((m, a, b)) => format!("{{\"mean\":{m:.4},\"p50\":{a:.4},\"p99\":{b:.4}}}"),
            None => "null".into(),
        };
        let phases: Vec<String> = crate::perf::phases()
            .iter()
            .map(|(n, s, _)| format!("\"{n}\":{:.4}", s))
            .collect();
        let info: Vec<String> = self.info.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
        // per-pass GPU means, in pass order
        let segs: Vec<String> = self.seg_ms.iter()
            .map(|(l, v)| format!("\"{l}\":{:.4}", v.iter().sum::<f64>() / self.gpu_ms.len().max(1) as f64))
            .collect();
        println!(
            "BENCH_JSON {{\"name\":\"{}\",\"size\":\"{}x{}\",\"seconds\":{},\"load_s\":{:.4},\
             \"phases_s\":{{{}}},\"frames\":{},\"cpu_ms\":{},\"gpu_ms\":{},\"gpu_pass_ms\":{{{}}},\
             \"cpu_worst_ms\":{:.2},\"cpu_over33\":{},\"vram_mb\":{:.1},\"peak_ws_mb\":{:.1},\"peak_commit_mb\":{:.1},\"ws_mb\":{:.1},\"info\":{{{}}}}}",
            self.opts.name.replace('\\', "/").replace('"', ""),
            self.opts.width, self.opts.height, self.opts.seconds, self.load_s,
            phases.join(","), self.cpu_ms.len(), js(stats(&self.cpu_ms)),
            js(stats(&self.gpu_ms)), segs.join(","), worst, over(33.3), mb(self.vram_peak), mb(ws), mb(commit), mb(ws_now),
            info.join(","));
    }
}

/// EVOVIEW_DET_LOG=1 (see render.rs): a hash of `v`'s bits joins this
/// frame's DET line, so two runs' logs show which state parted first.
pub fn det(label: &str, v: &[f32]) {
    if det_on() {
        let h = v.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3));
        if let Ok(mut s) = DET.lock() {
            s.push_str(&format!(" {label} {h:016x}"));
        }
    }
}

pub fn det_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::envc::var_os("EVOVIEW_DET_LOG").is_some())
}

static DET: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// What `det` gathered since the last call.
pub fn det_take() -> String {
    DET.lock().map(|mut s| std::mem::take(&mut *s)).unwrap_or_default()
}

/// FNV over bytes, eight at a time (the DET / DS lines).
pub fn det_bytes(d: &[u8]) -> u64 {
    d.chunks(8).fold(0xcbf29ce484222325u64, |h, c| {
        let mut b = [0u8; 8];
        b[..c.len()].copy_from_slice(c);
        (h ^ u64::from_le_bytes(b)).wrapping_mul(0x100000001b3)
    })
}
