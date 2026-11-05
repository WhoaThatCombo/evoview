//! Lap timing for the track drive: the current / last / best lap, sector
//! splits, the live delta to the best lap, lap count and invalid laps, a lap
//! panel and a leaderboard in the HUD's look, personal bests per car + track
//! + layout in %APPDATA%/evoview/laps.json, and the numbers the cars' own
//! dashes bind (dash.rs).
//!
//! TRACK DATA (content\tracks\<id>\containers\layout_<layout>.scene, the
//! container the track scene places with ContainerActorData f121): Spline
//! actors (f108 {1 {3 [ {1 transform {1 pos}} ]}}) named
//! `track_center_spline`, `track_limits_left`, `track_limits_right` (mods:
//! `<prefix>_track_center_spline` ...), plus the ideal / pit splines.
//! - The finish line is the centre spline's first point: every shipped
//!   layout's centre, limits and ideal line start together there (Imola: x
//!   222 on the main straight). No shipped layout has its own timing line;
//!   the only "RaceTimingLines" zone (ZoneActorData f112, 3 gates) is in
//!   imola\containers\sro.scene, an old SRO layout the track scene no longer
//!   places - it is read when present (EVOVIEW_TIMING_GATES=1) but not used
//!   by default.
//! - No sector lines were found in any track file (no "sector" string under
//!   content\tracks): S1 / S2 / S3 are thirds of the centre spline.
//! - Without a centre spline (some mods) the racing line stands in: its
//!   first point is the line.
//! TRACK LIMITS: a wheel is off when its centre is past the limit spline by
//!   EVOVIEW_CUT_MARGIN m (1.2: the game's own racing line runs up to 1.0 m past the limit lines at Tamburello / Spa, so 0.3 voided every lap); all four off
//!   for EVOVIEW_CUT_SECS (0.1 s) invalidates the lap. Without limit splines
//!   grass / sand under all four wheels (SurfaceDefinitionData surface_type
//!   3 / 4) does.
//! LAPS: progress is the car's projection on the centre spline. A lap ends on
//!   a forward crossing of the line; a reverse crossing is owed back first, a
//!   jump of > 150 m (Home to the pits, a reset) drops to an out lap. The out
//!   lap (from the pits, a reset, or crossing the line in the pit lane) is not
//!   timed; a standing start on the line is (the spawn is there).
//!
//! Timer is per car and the leaderboard takes a list of entries, so AI cars
//! can be timed and listed the same way.

use crate::proto::{fields, text};
use glam::{Mat4, Vec3};
use std::collections::HashMap;

// ------------------------------------------------------------ polylines --

/// A closed line in XZ with its running length.
pub struct Poly {
    pub pts: Vec<Vec3>,
    cum: Vec<f32>,
    pub len: f32,
    step: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Proj {
    pub seg: usize,
    /// distance along from the first point
    pub s: f32,
    /// signed distance sideways (+ = left of the line's direction)
    pub lat: f32,
    d2: f32,
}

impl Poly {
    pub fn new(mut pts: Vec<Vec3>) -> Option<Poly> {
        pts.dedup_by(|a, b| (*a - *b).length_squared() < 0.0025);
        if pts.len() < 4 {
            return None;
        }
        if pts[0].distance(*pts.last().unwrap()) > 0.05 {
            pts.push(pts[0]);
        }
        let mut cum = vec![0.0f32];
        for w in pts.windows(2) {
            let d = Vec3::new(w[1].x - w[0].x, 0.0, w[1].z - w[0].z).length();
            cum.push(cum.last().unwrap() + d);
        }
        let len = *cum.last().unwrap();
        let step = len / (pts.len() - 1) as f32;
        (len > 50.0).then_some(Poly { pts, cum, len, step })
    }

    fn segs(&self) -> usize {
        self.pts.len() - 1
    }

    fn seg(&self, i: usize, p: Vec3) -> Proj {
        let (a, b) = (self.pts[i], self.pts[i + 1]);
        let (dx, dz) = (b.x - a.x, b.z - a.z);
        let l2 = (dx * dx + dz * dz).max(1e-6);
        let t = (((p.x - a.x) * dx + (p.z - a.z) * dz) / l2).clamp(0.0, 1.0);
        let (qx, qz) = (a.x + t * dx, a.z + t * dz);
        let dy = (p.y - (a.y + t * (b.y - a.y))).abs();
        // another level (a bridge, Suzuka's crossover) costs extra
        let d2 = (p.x - qx).powi(2) + (p.z - qz).powi(2) + ((dy - 4.0).max(0.0) * 3.0).powi(2);
        let lat = (dx * (p.z - a.z) - dz * (p.x - a.x)) / l2.sqrt();
        Proj { seg: i, s: self.cum[i] + t * (self.cum[i + 1] - self.cum[i]), lat, d2 }
    }

    /// The nearest point, searched around `hint` first (the last answer).
    pub fn project(&self, p: Vec3, hint: Option<usize>) -> Proj {
        let n = self.segs();
        let better = |b: Option<Proj>, q: Proj| if b.map_or(true, |b| q.d2 < b.d2) { Some(q) } else { b };
        let mut best: Option<Proj> = None;
        if let Some(h) = hint {
            let w = ((250.0 / self.step.max(0.1)).ceil() as usize).clamp(8, n);
            for k in 0..w {
                best = better(best, self.seg((h + n - w / 4 + k) % n, p));
            }
            if let Some(b) = best.filter(|b| b.d2 < 40.0 * 40.0) {
                return b;
            }
        }
        for i in 0..n {
            best = better(best, self.seg(i, p));
        }
        best.unwrap()
    }

    /// Where the segment a-b crosses the line (XZ): its distance along.
    fn crossing(&self, a: Vec3, b: Vec3) -> Option<f32> {
        for i in 0..self.segs() {
            let (p, q) = (self.pts[i], self.pts[i + 1]);
            let r = (q.x - p.x, q.z - p.z);
            let s = (b.x - a.x, b.z - a.z);
            let den = r.0 * s.1 - r.1 * s.0;
            if den.abs() < 1e-9 {
                continue;
            }
            let (wx, wz) = (a.x - p.x, a.z - p.z);
            let t = (wx * s.1 - wz * s.0) / den;
            let u = (wx * r.1 - wz * r.0) / den;
            if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
                return Some(self.cum[i] + t * (self.cum[i + 1] - self.cum[i]));
            }
        }
        None
    }
}

// ------------------------------------------------------------ track data --

/// A track's (or a mod folder's) files, keys relative to the track folder,
/// lowercase with '\'.
struct Src {
    pkg: Option<crate::pkg::Package>,
    pre: String,
    loose: HashMap<String, std::path::PathBuf>,
}

impl Src {
    fn open(pkg: &std::path::Path, track: &str) -> Option<Src> {
        let pre = format!("content\\tracks\\{}\\", track.to_lowercase());
        if let Ok(p) = crate::pkg::Package::open_filtered(pkg, &[pre.as_str()]) {
            if !p.by_path.is_empty() {
                return Some(Src { pkg: Some(p), pre, loose: HashMap::new() });
            }
        }
        // a loose folder, or a mod id under Saved Games/ACE/mods
        let dir = std::path::Path::new(track);
        let dir = if dir.is_dir() {
            dir.to_path_buf()
        } else {
            std::path::Path::new(&crate::envc::var_os("USERPROFILE")?).join("Saved Games").join("ACE").join("mods")
                .join("content").join("tracks").join(track)
        };
        let mut loose = HashMap::new();
        for sub in ["", "containers", "layouts"] {
            let Ok(rd) = std::fs::read_dir(dir.join(sub)) else { continue };
            for e in rd.flatten() {
                if e.path().is_file() {
                    let name = e.file_name().to_string_lossy().to_lowercase();
                    let k = if sub.is_empty() { name } else { format!("{sub}\\{name}") };
                    loose.insert(k, e.path());
                }
            }
        }
        (!loose.is_empty()).then_some(Src { pkg: None, pre, loose })
    }

    fn keys(&self) -> Vec<String> {
        let mut k: Vec<String> = match &self.pkg {
            Some(p) => p.by_path.keys().filter_map(|k| k.strip_prefix(&self.pre)).map(str::to_string).collect(),
            None => self.loose.keys().cloned().collect(),
        };
        k.sort();
        k
    }

    fn read(&self, k: &str) -> Option<Vec<u8>> {
        match &self.pkg {
            Some(p) => p.by_path.get(&format!("{}{k}", self.pre)).map(|e| p.read(*e)),
            None => std::fs::read(self.loose.get(k)?).ok(),
        }
    }
}

/// Each actor of a scene: its name, node matrix and ActorData (f50) range.
fn actors(d: &[u8], mut f: impl FnMut(&str, Mat4, (usize, usize))) {
    fields(d, 0, d.len(), |fno, v| {
        let (2, Some((s, e))) = (fno, v.range()) else { return };
        let (mut name, mut node, mut body) = (String::new(), Mat4::IDENTITY, None);
        fields(d, s, e, |g, w| match (g, w.range()) {
            (1, Some(r)) => name = text(d, r),
            (4, Some((a, b))) => {
                let (p, r, sc) = crate::load::parse_transform(d, a, b);
                node = crate::load::trs(p, r, sc);
            }
            (50, Some(r)) => body = Some(r),
            _ => {}
        });
        if let Some(r) = body {
            f(&name, node, r);
        }
    });
}

fn sub(d: &[u8], (a, b): (usize, usize), want: u32) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    fields(d, a, b, |f, v| {
        if let (true, Some(r)) = (f == want, v.range()) {
            out.push(r);
        }
    });
    out
}

/// Where a transform list's points sit: [{1 transform {1 pos}}] -> world.
fn points(d: &[u8], list: &[(usize, usize)], m: Mat4) -> Vec<Vec3> {
    list.iter()
        .filter_map(|&r| sub(d, r, 1).first().copied())
        .map(|(a, b)| m.transform_point3(Vec3::from(crate::load::parse_transform(d, a, b).0)))
        .collect()
}

/// The finish line and sectors of the track being driven.
pub struct TimingTrack {
    pub center: Poly,
    /// track limits (left, right) and which side of each is outside (+1 / -1 lat)
    pub limits: Option<(Poly, f32, Poly, f32)>,
    /// distance along the centre of the finish line
    pub start: f32,
    /// S1 / S2 ends, from the line
    pub sectors: [f32; 2],
    /// what the line and the sectors came from (the console, the report)
    pub source: String,
    pub layout: String,
}

impl TimingTrack {
    pub fn load(pkg: &std::path::Path, track: &str) -> Option<TimingTrack> {
        let lay = crate::envc::var("EVOVIEW_LAYOUT").ok().map(|l| l.to_lowercase()).or_else(crate::load::loaded_layout);
        let src = Src::open(pkg, track);
        let mut found: Option<(String, HashMap<String, Vec<Vec3>>)> = None;
        let mut gates: Vec<Vec<Vec3>> = Vec::new();
        if let Some(src) = src.as_ref() {
            let keys = src.keys();
            // the track scene places the containers (one level, like pits.rs)
            let base = |k: &str| k.rsplit(['\\', '/']).next().unwrap_or(k).to_lowercase();
            let mut placed: HashMap<String, Mat4> = HashMap::new();
            for k in keys.iter().filter(|k| !k.contains('\\') && k.ends_with(".scene")) {
                let Some(d) = src.read(k) else { continue };
                actors(&d, |_, node, r| {
                    for c in sub(&d, r, 121) {
                        if let Some(p) = sub(&d, c, 1).first() {
                            placed.insert(base(&text(&d, *p)), node);
                        }
                    }
                });
            }
            let mut lays: Vec<&String> = keys.iter()
                .filter(|k| k.starts_with("containers\\layout_") && k.ends_with(".scene"))
                .collect();
            lays.sort_by_key(|k| {
                let want = lay.as_ref().map_or(false, |l| **k == format!("containers\\layout_{l}.scene"));
                (!want, !k.contains("gp"), k.len())
            });
            for k in lays {
                let Some(d) = src.read(k) else { continue };
                let cont = placed.get(&base(k)).copied().unwrap_or(Mat4::IDENTITY);
                let mut sp: HashMap<String, Vec<Vec3>> = HashMap::new();
                actors(&d, |name, node, r| {
                    let n = name.to_lowercase();
                    let Some(kind) = ["track_center_spline", "track_limits_left", "track_limits_right"].iter().find(|s| n.ends_with(*s)) else { return };
                    for s108 in sub(&d, r, 108) {
                        for s1 in sub(&d, s108, 1) {
                            let pts = points(&d, &sub(&d, s1, 3), cont * node);
                            if pts.len() > 3 {
                                sp.insert(kind.to_string(), pts);
                            }
                        }
                    }
                });
                if sp.contains_key("track_center_spline") {
                    found = Some((k.clone(), sp));
                    break;
                }
            }
            // a RaceTimingLines zone (only Imola's unplaced sro.scene has one)
            if crate::envc::var_os("EVOVIEW_TIMING_GATES").is_some() {
                for k in keys.iter().filter(|k| k.starts_with("containers\\") && k.ends_with(".scene")) {
                    let Some(d) = src.read(k) else { continue };
                    if !d.windows(6).any(|w| w == b"Timing") {
                        continue;
                    }
                    let cont = placed.get(&base(k)).copied().unwrap_or(Mat4::IDENTITY);
                    actors(&d, |name, node, r| {
                        if !name.contains("Timing") {
                            return;
                        }
                        for z in sub(&d, r, 112) {
                            for zd in sub(&d, z, 1000) {
                                for g in sub(&d, zd, 1) {
                                    gates.push(points(&d, &sub(&d, g, 1), cont * node));
                                }
                            }
                        }
                    });
                    println!("laps: {} timing gate(s) in {k}{}", gates.len(), if placed.contains_key(&base(k)) { "" } else { " (not placed by the track scene)" });
                }
            }
        }
        let (center, limits, mut source, layout) = match found {
            Some((k, mut sp)) => {
                // ⚠ Spa's splines sit 100 m below its road (y 19 vs 119):
                // level them to the racing line where both start
                if let Some(r0) = crate::trackdrive::racing_line(pkg, track).first().copied() {
                    let c = &sp["track_center_spline"];
                    if let Some(q) = c.iter().min_by(|a, b| {
                        let d = |p: &&Vec3| (p.x - r0.x).powi(2) + (p.z - r0.z).powi(2);
                        d(a).partial_cmp(&d(b)).unwrap()
                    }) {
                        let dy = r0.y - q.y;
                        if (q.x - r0.x).hypot(q.z - r0.z) < 30.0 && dy.abs() > 3.0 {
                            println!("laps: the layout's splines are {dy:+.1} m off the racing line in height - moved");
                            for v in sp.values_mut() {
                                v.iter_mut().for_each(|p| p.y += dy);
                            }
                        }
                    }
                }
                let c = Poly::new(sp.remove("track_center_spline").unwrap())?;
                let side = |p: &Poly| -> f32 {
                    // outside = away from the centre: the centre's points
                    // sit on the inside of each limit line
                    let mut sum = 0.0;
                    for q in c.pts.iter().step_by((c.pts.len() / 40).max(1)) {
                        sum += p.project(*q, None).lat.signum();
                    }
                    -sum.signum()
                };
                let lim = match (sp.remove("track_limits_left").and_then(Poly::new), sp.remove("track_limits_right").and_then(Poly::new)) {
                    (Some(l), Some(r)) => {
                        let (sl, sr) = (side(&l), side(&r));
                        Some((l, sl, r, sr))
                    }
                    _ => None,
                };
                let layout = k.trim_start_matches("containers\\layout_").trim_end_matches(".scene").to_string();
                (c, lim, format!("finish = start of track_center_spline ({k})"), layout)
            }
            None => {
                let line = crate::trackdrive::racing_line(pkg, track);
                let c = Poly::new(line)?;
                (c, None, "finish = start of the racing line (no track_center_spline)".to_string(), lay.clone().unwrap_or_default())
            }
        };
        let len = center.len;
        let mut start = 0.0;
        let mut sectors = [len / 3.0, len * 2.0 / 3.0];
        let mut sec_src = "sectors = thirds of the lap (no sector lines in the track data)".to_string();
        // gates: each a polyline across the track; the one nearest the
        // spline's start is the line, the others the sector ends
        let mut at: Vec<f32> = gates.iter()
            .filter_map(|g| g.windows(2).find_map(|w| center.crossing(w[0], w[1])))
            .collect();
        if at.len() == 3 {
            at.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("laps: gates at {:.0} / {:.0} / {:.0} m of {len:.0}", at[0], at[1], at[2]);
            let fi = (0..3).min_by(|&a, &b| {
                let d = |s: f32| s.min(len - s);
                d(at[a]).partial_cmp(&d(at[b])).unwrap()
            }).unwrap();
            start = at[fi];
            let rel: Vec<f32> = (1..3).map(|k| (at[(fi + k) % 3] - start).rem_euclid(len)).collect();
            sectors = [rel[0].min(rel[1]), rel[0].max(rel[1])];
            source = format!("finish + sectors = the RaceTimingLines zone (gate at {start:.0} m)");
            sec_src.clear();
        }
        if let Ok(v) = crate::envc::var("EVOVIEW_LAP_START") {
            if let Ok(m) = v.parse::<f32>() {
                start = m.rem_euclid(len);
                source = format!("finish = {m:.0} m along the centre (EVOVIEW_LAP_START)");
            }
        }
        let p0 = center.pts[center.project_s(start)];
        println!("laps: {source}; {sec_src}{}lap {len:.0} m, line at ({:.0}, {:.0}, {:.0}), S1 {:.0} m, S2 {:.0} m, limits {}",
                 if sec_src.is_empty() { "" } else { "; " }, p0.x, p0.y, p0.z, sectors[0], sectors[1],
                 if limits.is_some() { "yes" } else { "no (grass / sand instead)" });
        let tt = TimingTrack { center, limits, start, sectors, source: format!("{source}; {sec_src}"), layout };
        // EVOVIEW_LAPS_LOG: how far the racing line itself runs past the limits
        if let (Some((l, sl, r, sr)), true) = (tt.limits.as_ref(), crate::envc::var_os("EVOVIEW_LAPS_LOG").is_some()) {
            let mut worst: Vec<(f32, Vec3)> = crate::trackdrive::racing_line(pkg, track).iter().step_by(4)
                .map(|p| ((l.project(*p, None).lat * sl).max(r.project(*p, None).lat * sr), *p)).collect();
            worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for (e, p) in worst.iter().take(8) {
                println!("LAPSLOG ideal line past the limits by {e:+.2} m at ({:.0}, {:.0}) {:.0} m", p.x, p.z, tt.rel(tt.center.project(*p, None).s));
            }
        }
        Some(tt)
    }

    /// Distance from the line along the lap, 0..len.
    fn rel(&self, s: f32) -> f32 {
        (s - self.start).rem_euclid(self.center.len)
    }

    pub fn len(&self) -> f32 {
        self.center.len
    }
}

impl Poly {
    /// The point index at distance `s` along.
    fn project_s(&self, s: f32) -> usize {
        self.cum.partition_point(|c| *c < s).min(self.pts.len() - 1)
    }
}

// ----------------------------------------------------------------- timer --

#[derive(Clone, Copy, Debug)]
pub struct Lap {
    /// 1-based lap number
    pub n: u32,
    pub time: f32,
    pub sectors: [Option<f32>; 3],
    pub valid: bool,
    /// ended in the pit lane
    pub pit: bool,
}

/// How a split compares: personal best ever, better than this session's
/// best, slower.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mark {
    Purple,
    Green,
    Yellow,
}

/// Seconds since each wheel / all wheels went off.
const TRACE_M: f32 = 10.0;

pub struct Timer {
    hint: Option<usize>,
    lim_hint: [Option<usize>; 2],
    /// distance along the lap last step, None before the first
    s: Option<f32>,
    /// timing a lap (false: the out lap)
    pub timing: bool,
    /// the running lap's time (the out lap's too)
    pub t: f32,
    /// how far into this lap (can dip below 0 after backing over the line)
    dist: f32,
    /// reverse crossings owed back before a forward one ends a lap
    owed: i32,
    pub sector: usize,
    pub splits: [Option<f32>; 3],
    pub invalid: bool,
    off_t: f32,
    /// laps finished (timed or not)
    pub count: u32,
    pub laps: Vec<Lap>,
    pub best: Option<Lap>,
    /// this session's best per sector, and the all-time ones (laps.json)
    pub best_sectors: [Option<f32>; 3],
    pub pb_sectors: [Option<f32>; 3],
    /// the all-time best lap time and its time every TRACE_M m
    pub pb: Option<f32>,
    pub pb_trace: Vec<f32>,
    trace: Vec<f32>,
    best_trace: Vec<f32>,
    /// the marks of this lap's finished sectors
    pub marks: [Option<Mark>; 3],
    /// the lap just finished and seconds since (HUD pop-up, dash)
    pub event: Option<(Lap, f32, [Option<Mark>; 3], f32)>,
    /// all four wheels are off right now
    pub off: bool,
    /// lap 1 of a standing start: the next forward crossing is the grid's
    grid: bool,
}

impl Default for Timer {
    fn default() -> Self {
        Timer {
            hint: None,
            lim_hint: [None; 2],
            s: None,
            timing: false,
            t: 0.0,
            dist: 0.0,
            owed: 0,
            sector: 0,
            splits: [None; 3],
            invalid: false,
            off_t: 0.0,
            count: 0,
            laps: Vec::new(),
            best: None,
            best_sectors: [None; 3],
            pb_sectors: [None; 3],
            pb: None,
            pb_trace: Vec::new(),
            trace: Vec::new(),
            best_trace: Vec::new(),
            marks: [None; 3],
            event: None,
            off: false,
            grid: false,
        }
    }
}

fn env_f(k: &str, d: f32) -> f32 {
    crate::envc::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// The time at `d` m into a lap from a trace (linear between samples).
fn trace_at(tr: &[f32], d: f32) -> Option<f32> {
    if tr.len() < 2 || d < 0.0 {
        return None;
    }
    let x = d / TRACE_M;
    let i = x.floor() as usize;
    if i + 1 >= tr.len() {
        return None;
    }
    Some(tr[i] + (tr[i + 1] - tr[i]) * (x - i as f32))
}

impl Timer {
    /// Back to an out lap (a reset, Home to the pits).
    /// A standing start from a grid behind the line: lap 1 is timed from
    /// the lights, and the first crossing (just after the start) is not a
    /// lap - the distance before the line counts as negative.
    pub fn race_start(&mut self, tt: &TimingTrack) {
        let len = tt.len();
        self.timing = true;
        self.t = 0.0;
        self.owed = 0;
        self.restart_lap();
        self.dist = self.s.map_or(0.0, |s| if s > len * 0.5 { s - len } else { s });
        self.grid = self.dist < 0.0;
    }

    pub fn out_lap(&mut self) {
        self.timing = false;
        self.grid = false;
        self.t = 0.0;
        self.dist = 0.0;
        self.owed = 0;
        self.restart_lap();
    }

    fn restart_lap(&mut self) {
        self.sector = 0;
        self.splits = [None; 3];
        self.marks = [None; 3];
        self.invalid = false;
        self.off_t = 0.0;
        self.trace.clear();
        self.trace.push(0.0);
    }

    /// The lap number on the board: the one being driven.
    pub fn lap_no(&self) -> u32 {
        self.count + 1
    }

    pub fn last(&self) -> Option<&Lap> {
        self.laps.last()
    }

    /// The live delta to the best lap (this session's, else the all-time
    /// one), seconds; + = slower.
    pub fn delta(&self) -> Option<f32> {
        if !self.timing {
            return None;
        }
        // the fastest reference: the all-time best (this session's when it
        // set a new one, pb_trace follows it)
        let tr = if self.pb_trace.len() > 1 { &self.pb_trace } else { &self.best_trace };
        trace_at(tr, self.dist).map(|b| self.t - b)
    }

    /// The reference best: the all-time one, else this session's.
    pub fn ref_best(&self) -> Option<f32> {
        match (self.pb, self.best.map(|b| b.time)) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    fn mark(&self, i: usize, v: f32) -> Mark {
        if self.pb_sectors[i].map_or(true, |p| v <= p) {
            Mark::Purple
        } else if self.best_sectors[i].map_or(true, |b| v < b) {
            Mark::Green
        } else {
            Mark::Yellow
        }
    }

    /// One step: `pos` the car, `wheels_off` all four wheels off the track
    /// (wheels_off below), `in_pit` in the pit lane. Returns a finished lap.
    pub fn update(&mut self, tt: &TimingTrack, pos: Vec3, wheels_off: bool, in_pit: bool, dt: f32) -> Option<Lap> {
        if let Some(e) = self.event.as_mut() {
            e.1 += dt;
        }
        let pr = tt.center.project(pos, self.hint);
        if pr.d2 > 80.0 * 80.0 {
            // nowhere near the track (a garage far off, the fly cam)
            self.hint = None;
            return None;
        }
        self.hint = Some(pr.seg);
        let len = tt.len();
        let s = tt.rel(pr.s);
        if crate::envc::var_os("EVOVIEW_LAPS_LOG").is_some() && (self.t / 5.0).floor() != ((self.t + dt) / 5.0).floor() {
            eprintln!("LAPSLOG t {:.1} s {s:.0} off {:.1} m timing {} dist {:.0} owed {}", self.t, pr.d2.sqrt(), self.timing, self.dist, self.owed);
        }
        let Some(prev) = self.s else {
            // a standing start on (just past) the line is timed
            self.s = Some(s);
            self.restart_lap();
            if s < 40.0 && !in_pit {
                self.timing = true;
                self.dist = s;
            }
            return None;
        };
        let mut ds = s - prev;
        if ds > len * 0.5 {
            ds -= len;
        } else if ds < -len * 0.5 {
            ds += len;
        }
        self.s = Some(s);
        if ds.abs() > 150.0 {
            println!("laps: moved {ds:.0} m at once - out lap");
            self.out_lap();
            return None;
        }
        let fwd = ds > 0.0 && s < prev;
        let back = ds < 0.0 && s > prev;
        self.t += dt;
        if self.timing {
            let d0 = self.dist;
            self.dist += ds;
            // sector ends (forward through them)
            for (i, &b) in tt.sectors.iter().enumerate() {
                if self.sector == i && d0 < b && self.dist >= b {
                    let at = self.t - dt * (self.dist - b) / ds.max(1e-4);
                    let before: f32 = self.splits[..i].iter().flatten().sum();
                    let v = at - before;
                    self.splits[i] = Some(v);
                    self.marks[i] = Some(self.mark(i, v));
                    self.sector = i + 1;
                }
            }
            // the time every TRACE_M m, for the delta
            while self.dist >= self.trace.len() as f32 * TRACE_M && ds > 0.0 {
                let mark = self.trace.len() as f32 * TRACE_M;
                self.trace.push(self.t - dt * (self.dist - mark) / ds.max(1e-4));
            }
            // track limits
            if wheels_off {
                self.off_t += dt;
                if self.off_t >= env_f("EVOVIEW_CUT_SECS", 0.1) && !self.invalid {
                    self.invalid = true;
                    println!("laps: lap {} invalid (all four wheels off at {:.0} m)", self.lap_no(), self.dist);
                }
            } else {
                self.off_t = 0.0;
            }
        }
        self.off = wheels_off;
        if back {
            self.owed += 1;
            return None;
        }
        if !fwd {
            return None;
        }
        if self.owed > 0 {
            self.owed -= 1;
            return None;
        }
        // across the line: when, within this step
        let over = dt * (s / ds.max(1e-4)).clamp(0.0, 1.0);
        if !self.timing {
            // the out lap ends; crossing in the pit lane keeps it going
            if !in_pit {
                self.timing = true;
                self.t = over;
                self.dist = s;
                self.restart_lap();
                println!("laps: out lap done, lap {} timed from here", self.lap_no());
            }
            return None;
        }
        if self.grid {
            // the race start's crossing: lap 1 goes on, measured from here
            self.grid = false;
            self.dist = s;
            return None;
        }
        if self.dist < len * 0.5 {
            // not a lap's worth of track: a projection jump, not a lap
            self.out_lap();
            return None;
        }
        let time = self.t - over;
        let prev_best = self.ref_best();
        let mut sectors = self.splits;
        if let (Some(a), Some(b)) = (sectors[0], sectors[1]) {
            sectors[2] = Some(time - a - b);
            self.marks[2] = Some(self.mark(2, time - a - b));
        }
        self.count += 1;
        let lap = Lap { n: self.count, time, sectors, valid: !self.invalid && !in_pit, pit: in_pit };
        self.laps.push(lap);
        if lap.valid {
            for i in 0..3 {
                if let Some(v) = sectors[i] {
                    if self.best_sectors[i].map_or(true, |b| v < b) {
                        self.best_sectors[i] = Some(v);
                    }
                }
            }
            if self.best.map_or(true, |b| time < b.time) {
                self.best = Some(lap);
                let mut tr = std::mem::take(&mut self.trace);
                tr.push(time);
                self.best_trace = tr;
            }
            for i in 0..3 {
                if let Some(v) = sectors[i] {
                    if self.pb_sectors[i].map_or(true, |p| v < p) {
                        self.pb_sectors[i] = Some(v);
                    }
                }
            }
            if self.pb.map_or(true, |p| time < p) {
                self.pb = Some(time);
                self.pb_trace = self.best_trace.clone();
            }
        }
        let delta = prev_best.map_or(f32::NAN, |b| time - b);
        self.event = Some((lap, 0.0, self.marks, delta));
        // in the pit lane: the in lap; the next one starts from the pits
        if in_pit {
            self.out_lap();
        } else {
            self.t = over;
            self.dist = s;
            self.restart_lap();
        }
        Some(lap)
    }

    /// Which wheels are off the track: past the limit splines by the margin,
    /// else on grass / sand.
    pub fn wheels_off(&mut self, tt: &TimingTrack, ground: &crate::trackdrive::Ground, wheels: &[Vec3; 4]) -> bool {
        let margin = env_f("EVOVIEW_CUT_MARGIN", 1.2);
        match tt.limits.as_ref() {
            Some((l, sl, r, sr)) => {
                let mut all = true;
                for w in wheels {
                    let pl = l.project(*w, self.lim_hint[0]);
                    let pr = r.project(*w, self.lim_hint[1]);
                    self.lim_hint = [Some(pl.seg), Some(pr.seg)];
                    let out = pl.lat * sl > margin || pr.lat * sr > margin;
                    if crate::envc::var_os("EVOVIEW_LAPS_LOG").is_some() && out {
                        eprintln!("LAPSLOG wheel ({:.1}, {:.1}) past left {:+.2} right {:+.2}", w.x, w.z, pl.lat * sl, pr.lat * sr);
                    }
                    all &= out;
                }
                all
            }
            None => wheels.iter().all(|w| {
                ground.probe(w.x, w.z, w.y + 1.5).map_or(false, |h| matches!(ground.kind(h.surf), 3 | 4))
            }),
        }
    }
}

/// "1:57.812" (decimals 1..3); under a minute "57.812".
pub fn fmt(t: f32, decimals: usize) -> String {
    if !t.is_finite() {
        return "-".into();
    }
    let p = 10f32.powi(decimals as i32);
    let t = (t.abs() * p).round() / p;
    let m = (t / 60.0).floor() as u32;
    let s = t - m as f32 * 60.0;
    if m > 0 {
        format!("{m}:{s:0w$.decimals$}", w = decimals + 3)
    } else {
        format!("{s:.decimals$}")
    }
}

/// "+0.32" / "-1.05".
pub fn fmt_delta(d: f32) -> String {
    if !d.is_finite() {
        return "-".into();
    }
    format!("{}{:.2}", if d >= 0.0 { "+" } else { "-" }, d.abs())
}

// ---------------------------------------------------------------- store --

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct Saved {
    pub ms: u32,
    #[serde(default)]
    pub sectors: [Option<u32>; 3],
    #[serde(default)]
    pub date: String,
    #[serde(default)]
    pub lap: u32,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, Default)]
pub struct Record {
    /// the best laps, fastest first
    pub laps: Vec<Saved>,
    /// the best of each sector over every saved lap
    #[serde(default)]
    pub best_sectors: [Option<u32>; 3],
    /// the fastest lap's time (ms) every 10 m, for the delta
    #[serde(default)]
    pub trace: Vec<u32>,
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
pub struct Store {
    #[serde(default)]
    pub records: std::collections::BTreeMap<String, Record>,
}

/// EVOVIEW_LAPS_FILE, else %APPDATA%/evoview/laps.json.
fn store_path() -> Option<std::path::PathBuf> {
    if let Some(p) = crate::envc::var_os("EVOVIEW_LAPS_FILE") {
        return Some(p.into());
    }
    crate::envc::var_os("APPDATA").map(|a| std::path::Path::new(&a).join("evoview").join("laps.json"))
}

impl Store {
    pub fn load() -> Store {
        store_path().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
    }

    fn save(&self) {
        let Some(p) = store_path() else { return };
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        if let Ok(s) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(p, s);
        }
    }
}

/// Today as "2026-11-01" (civil from days, no date crate).
fn today() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let z = secs.div_euclid(86400) + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{d:02}")
}

// ------------------------------------------------------------ leaderboard --

/// One row of the leaderboard. The board draws rows in the order given;
/// `standings` sorts timers by best lap.
#[derive(Clone, Debug, Default)]
pub struct Entry {
    pub name: String,
    pub best: Option<f32>,
    pub last: Option<f32>,
    /// behind the leader, seconds (None: the leader / no time)
    pub gap: Option<f32>,
    pub laps: u32,
    /// the player's row (highlighted)
    pub you: bool,
}

/// Rows for several timed cars, fastest best lap first.
pub fn standings(cars: &[(&str, &Timer, bool)]) -> Vec<Entry> {
    let mut v: Vec<Entry> = cars.iter().map(|(n, t, you)| Entry {
        name: n.to_string(),
        best: t.best.map(|b| b.time),
        last: t.last().map(|l| l.time),
        gap: None,
        laps: t.count,
        you: *you,
    }).collect();
    v.sort_by(|a, b| a.best.unwrap_or(f32::MAX).partial_cmp(&b.best.unwrap_or(f32::MAX)).unwrap());
    let lead = v.first().and_then(|e| e.best);
    for e in v.iter_mut().skip(1) {
        e.gap = e.best.zip(lead).map(|(a, b)| a - b);
    }
    v
}

// --------------------------------------------------------------- session --

/// The player's lap timing on one car + track + layout.
pub struct Session {
    pub track: TimingTrack,
    pub timer: Timer,
    pub store: Store,
    key: String,
    /// write laps.json (not for scripted drivers)
    save: bool,
    /// the leaderboard: on by its key, and for a while after each lap
    pub board_on: bool,
    pub board_t: f32,
    /// the AI's rows (another agent's race order), drawn instead of the
    /// personal history when not empty
    pub others: Vec<Entry>,
    /// the AI's timers (race.rs `ai[k]`), never saved
    pub ai_timers: Vec<Timer>,
}

/// How long the lap-end pop-up and the leaderboard stay up.
pub const POP_SECS: f32 = 5.0;
pub const BOARD_SECS: f32 = 8.0;

impl Session {
    pub fn load(pkg: &std::path::Path, track: &str, car: &str, save: bool) -> Option<Session> {
        if crate::envc::var("EVOVIEW_LAPS").map_or(false, |v| v == "0") {
            return None;
        }
        let tt = TimingTrack::load(pkg, track)?;
        let key = format!("{}|{}|{}", car.to_lowercase(), track.to_lowercase(), tt.layout);
        let store = Store::load();
        let mut timer = Timer::default();
        if let Some(r) = store.records.get(&key) {
            timer.pb = r.laps.first().map(|l| l.ms as f32 / 1000.0);
            timer.pb_sectors = r.best_sectors.map(|s| s.map(|v| v as f32 / 1000.0));
            timer.pb_trace = r.trace.iter().map(|v| *v as f32 / 1000.0).collect();
            println!("laps: personal best {} ({} saved lap(s), {key})", timer.pb.map_or("-".into(), |t| fmt(t, 3)), r.laps.len());
        }
        let save = save || crate::envc::var_os("EVOVIEW_LAPS_FILE").is_some();
        // EVOVIEW_LAPS_BOARD=1: the leaderboard up from the start (captures)
        let board_on = crate::envc::var_os("EVOVIEW_LAPS_BOARD").is_some();
        Some(Session { track: tt, timer, store, key, save, board_on, board_t: f32::INFINITY, others: Vec::new(), ai_timers: Vec::new() })
    }

    /// After the car's physics step.
    pub fn step(&mut self, td: &crate::trackdrive::TrackDrive, in_pit: bool, dt: f32) {
        // the hubs in the world (TrackDrive.wheels)
        let wheels = td.pose().wheels;
        let off = self.timer.wheels_off(&self.track, &td.ground, &wheels);
        self.board_t += dt;
        if let Some(lap) = self.timer.update(&self.track, td.pos, off, in_pit, dt) {
            let s = |v: Option<f32>| v.map_or("-".to_string(), |v| fmt(v, 3));
            println!("LAPTIME lap {} {}  S1 {} S2 {} S3 {}{}{}", lap.n, fmt(lap.time, 3), s(lap.sectors[0]), s(lap.sectors[1]),
                     s(lap.sectors[2]), if lap.valid { "" } else { "  (invalid)" }, if lap.pit { "  (pit)" } else { "" });
            self.board_t = 0.0;
            if lap.valid {
                self.record(&lap);
            }
        }
    }

    /// A valid lap into laps.json: the top 10 per car + track + layout.
    fn record(&mut self, lap: &Lap) {
        let ms = |v: f32| (v * 1000.0).round() as u32;
        let r = self.store.records.entry(self.key.clone()).or_default();
        let fastest = r.laps.first().map_or(true, |f| ms(lap.time) < f.ms);
        r.laps.push(Saved { ms: ms(lap.time), sectors: lap.sectors.map(|s| s.map(ms)), date: today(), lap: lap.n });
        r.laps.sort_by_key(|l| l.ms);
        r.laps.truncate(10);
        for i in 0..3 {
            if let Some(v) = lap.sectors[i].map(ms) {
                if r.best_sectors[i].map_or(true, |b| v < b) {
                    r.best_sectors[i] = Some(v);
                }
            }
        }
        if fastest {
            r.trace = self.timer.best_trace.iter().map(|t| ms(*t)).collect();
            println!("laps: new personal best {}", fmt(lap.time, 3));
        }
        if self.save {
            self.store.save();
        }
    }

    /// The opponents' laps (race.rs), after they stepped. They are not
    /// judged on track limits.
    pub fn step_ai(&mut self, cars: &[&crate::trackdrive::TrackDrive], dt: f32) {
        self.ai_timers.resize_with(cars.len(), Timer::default);
        for (t, td) in self.ai_timers.iter_mut().zip(cars) {
            t.update(&self.track, td.pos, false, false, dt);
        }
    }

    /// The race leaderboard from race.rs's order (car 0 = the player, k =
    /// ai[k - 1]; progress in laps): each car's timer for best / last /
    /// laps, the gap as the distance behind the leader at a lap's pace.
    pub fn race_rows(&self, order: &[(usize, f32)], name: impl Fn(usize) -> String) -> Vec<Entry> {
        let pace = self.timer.best.map(|b| b.time).or(self.timer.pb).unwrap_or(120.0);
        let lead = order.first().map_or(0.0, |o| o.1);
        order.iter().enumerate().map(|(i, &(k, p))| {
            let t = if k == 0 { Some(&self.timer) } else { self.ai_timers.get(k - 1) };
            Entry {
                name: name(k),
                best: t.and_then(|t| t.best.map(|b| b.time)),
                last: t.and_then(|t| t.last().map(|l| l.time)),
                gap: (i > 0).then(|| (lead - p) * pace),
                laps: t.map_or(0, |t| t.count),
                you: k == 0,
            }
        }).collect()
    }

    /// The race went from a standing start: every car's lap 1 from now.
    pub fn race_start(&mut self) {
        self.timer.race_start(&self.track);
        for t in self.ai_timers.iter_mut() {
            t.race_start(&self.track);
        }
    }

    pub fn toggle_board(&mut self) {
        self.board_on = !self.board_on;
    }

    /// The numbers the cars' dashes bind (dash.rs).
    pub fn dash(&self) -> crate::fxstate::LapDash {
        let t = &self.timer;
        let ms = |v: f32| (v * 1000.0).round() as f64;
        let delta = t.delta();
        crate::fxstate::LapDash {
            current_ms: ms(t.t),
            last_ms: t.last().map_or(0.0, |l| ms(l.time)),
            best_ms: t.best.map_or(0.0, |b| ms(b.time)),
            delta_ms: delta.map_or(0.0, ms),
            predicted_ms: delta.zip(t.ref_best()).map_or(0.0, |(d, b)| ms(b + d)),
            laps: t.count,
            since_lap: t.event.map_or(f32::INFINITY, |e| e.1),
            last_is_best: t.last().zip(t.best).map_or(false, |(l, b)| l.time == b.time),
        }
    }

    /// The board's rows: the AI's when there are any, else this session and
    /// the saved personal bests.
    pub fn board(&self) -> (String, Vec<Entry>) {
        if !self.others.is_empty() {
            return ("Race".into(), self.others.clone());
        }
        let t = &self.timer;
        let mut v = vec![Entry {
            name: "You (this session)".into(),
            best: t.best.map(|b| b.time),
            last: t.last().map(|l| l.time),
            gap: None,
            laps: t.count,
            you: true,
        }];
        if let Some(r) = self.store.records.get(&self.key) {
            let today = today();
            let ms = |v: f32| (v * 1000.0).round() as u32;
            for l in &r.laps {
                // this session's own saves are its row already
                if l.date == today && t.laps.iter().any(|s| s.valid && ms(s.time) == l.ms && s.n == l.lap) {
                    continue;
                }
                v.push(Entry {
                    name: format!("You  {}", l.date),
                    best: Some(l.ms as f32 / 1000.0),
                    last: None,
                    gap: None,
                    laps: l.lap,
                    you: false,
                });
            }
        }
        v.sort_by(|a, b| a.best.unwrap_or(f32::MAX).partial_cmp(&b.best.unwrap_or(f32::MAX)).unwrap());
        let lead = v.first().and_then(|e| e.best);
        for e in v.iter_mut().skip(1) {
            e.gap = e.best.zip(lead).map(|(a, b)| a - b);
        }
        v.truncate(9);
        let car = self.key.split('|').next().unwrap_or("");
        (format!("Personal bests · {car} · {}", self.track.layout), v)
    }
}

// ------------------------------------------------------------------- HUD --

fn purple(a: f32) -> [f32; 4] {
    crate::hud::theme::hex(0xb072f0, a)
}

fn mark_col(m: Mark) -> [f32; 4] {
    use crate::hud::theme as t;
    match m {
        Mark::Purple => purple(1.0),
        Mark::Green => t::ok(1.0),
        Mark::Yellow => t::gold(1.0),
    }
}

/// The lap panel's box at the top right (the pit sign goes left of it).
pub fn panel_rect(aspect: f32) -> (f32, f32, f32, f32) {
    let (w, h) = (0.25 / aspect, 0.2);
    (1.0 - w - 0.02 / aspect, 0.025, w, h)
}

impl Session {
    /// The lap panel (top right), the lap-end pop-up and the leaderboard.
    pub fn draw(&self, hud: &mut crate::hud::Hud, aspect: f32) {
        use crate::hud::{theme as t, Weight};
        let tm = &self.timer;
        let u = |v: f32| v / aspect;
        let (x, y, w, h) = panel_rect(aspect);
        hud.panel(x, y, w, h, 10.0, t::panel(0.42), t::line(0.45));
        let pad = 0.013;
        let (ix, iy, iw) = (x + u(pad), y + pad, w - u(pad) * 2.0);
        let ls = 0.0105;
        let right = |hud: &mut crate::hud::Hud, s: &str, yy: f32, size: f32, wt: Weight, c: [f32; 4]| {
            let tw = hud.label_w(s, size, wt);
            hud.label(s, ix + iw - tw, yy, size, wt, c);
        };
        // header: the lap, its state
        hud.label(&format!("Lap {}", tm.lap_no()), ix, iy, ls, Weight::Semibold, t::dim(1.0));
        let (state, sc) = if !tm.timing {
            ("Out lap", t::dim(1.0))
        } else if tm.invalid {
            ("Invalid", t::red(1.0))
        } else if tm.off {
            ("Off track", t::gold(1.0))
        } else {
            ("", t::dim(1.0))
        };
        right(hud, state, iy, ls, Weight::Semibold, sc);
        // the running time, big, and the delta beside it
        let big = 0.036;
        let ty = iy + hud.line_h(ls) - 0.004;
        let cur = if tm.timing { fmt(tm.t, 1) } else { "-:--.-".into() };
        hud.label(&cur, ix, ty, big, Weight::Semibold, if tm.invalid { t::dim(1.0) } else { t::fg(1.0) });
        if let Some(d) = tm.delta() {
            let ds = 0.02;
            let c = if d <= 0.0 { t::ok(1.0) } else { t::red(1.0) };
            right(hud, &fmt_delta(d), ty + (hud.line_h(big) - hud.line_h(ds)) * 0.62, ds, Weight::Semibold, c);
        }
        // sectors: three cells, coloured like a timing screen
        let sy = ty + hud.line_h(big) + 0.002;
        let gap = u(0.004);
        let cw = (iw - gap * 2.0) / 3.0;
        let ch = hud.line_h(ls) * 2.0 + 0.004;
        // a lap just finished shows its own splits for a few seconds
        let showing_last = tm.event.filter(|e| e.1 < POP_SECS && tm.sector == 0);
        let (splits, marks) = match showing_last {
            Some((lap, _, m, _)) => (lap.sectors, m),
            None => (tm.splits, tm.marks),
        };
        for i in 0..3 {
            let cx = ix + i as f32 * (cw + gap);
            let live = showing_last.is_none() && tm.timing && i == tm.sector;
            let (fill, txt) = match (splits[i], marks[i]) {
                (Some(_), Some(m)) => (mark_col(m), t::hex(0x15181e, 1.0)),
                _ => (t::well(if live { 0.9 } else { 0.6 }), t::dim(1.0)),
            };
            hud.round_rect(cx, sy, cw, ch, 4.0, fill);
            hud.label(&format!("S{}", i + 1), cx + u(0.005), sy + 0.002, ls, Weight::Regular, txt);
            let v = match splits[i] {
                Some(v) => fmt(v, 3),
                None if live => {
                    let before: f32 = tm.splits[..i].iter().flatten().sum();
                    fmt(tm.t - before, 1)
                }
                None => "-".into(),
            };
            hud.label(&v, cx + u(0.005), sy + hud.line_h(ls) - 0.001, ls * 1.15, Weight::Semibold,
                      if splits[i].is_some() { txt } else { t::fg(0.9) });
        }
        // last and best
        let ry = sy + ch + 0.008;
        let row = |hud: &mut crate::hud::Hud, yy: f32, label: &str, v: String, c: [f32; 4]| {
            hud.label(label, ix, yy, ls, Weight::Regular, t::dim(1.0));
            right(hud, &v, yy - 0.002, ls * 1.35, Weight::Semibold, c);
        };
        let last = tm.last();
        row(hud, ry, "Last", last.map_or("-".into(), |l| fmt(l.time, 3)),
            last.map_or(t::fg(1.0), |l| if !l.valid { t::red(1.0) } else if tm.pb == Some(l.time) { purple(1.0) } else { t::fg(1.0) }));
        let lh = hud.line_h(ls * 1.35);
        row(hud, ry + lh, "Best", tm.best.map_or("-".into(), |b| fmt(b.time, 3)), t::fg(1.0));
        row(hud, ry + lh * 2.0, "Personal best", tm.pb.map_or("-".into(), |b| fmt(b, 3)), purple(1.0));

        // the lap-end pop-up, top centre under the pit chips
        if let Some((lap, age, _, delta)) = tm.event.filter(|e| e.1 < POP_SECS) {
            let a = ((POP_SECS - age) / 0.5).clamp(0.0, 1.0);
            let big = 0.05;
            let label = format!("Lap {}", lap.n);
            let v = fmt(lap.time, 3);
            let note = if lap.pit {
                "In lap".to_string()
            } else if !lap.valid {
                "Invalid".to_string()
            } else if tm.pb == Some(lap.time) {
                "Personal best".to_string()
            } else {
                fmt_delta(delta)
            };
            let nc = if lap.pit || !lap.valid {
                t::red(a)
            } else if tm.pb == Some(lap.time) {
                purple(a)
            } else if delta <= 0.0 {
                t::ok(a)
            } else {
                t::red(a)
            };
            let small = big * 0.42;
            let lw = hud.label_w(&label, small, Weight::Regular);
            let vw = hud.label_w(&v, big, Weight::Semibold);
            let nw = if delta.is_finite() || lap.pit || !lap.valid || tm.pb == Some(lap.time) { hud.label_w(&note, small * 1.3, Weight::Semibold) } else { 0.0 };
            let gap = u(0.014);
            let (padx, pady) = (u(0.022), 0.012);
            let pw = lw + gap + vw + if nw > 0.0 { gap + nw } else { 0.0 } + padx * 2.0;
            let ph = hud.line_h(big) + pady * 2.0;
            let (px, py) = (0.5 - pw * 0.5, 0.14);
            hud.panel(px, py, pw, ph, 8.0, t::panel(0.9 * a), t::line(a));
            let mid = py + pady + hud.line_h(big) * 0.5;
            hud.label(&label, px + padx, mid - hud.line_h(small) * 0.42, small, Weight::Regular, t::dim(a));
            let mut vc = t::fg(1.0);
            vc[3] = a;
            hud.label(&v, px + padx + lw + gap, py + pady, big, Weight::Semibold, vc);
            if nw > 0.0 {
                hud.label(&note, px + padx + lw + gap + vw + gap, mid - hud.line_h(small * 1.3) * 0.45, small * 1.3, Weight::Semibold, nc);
            }
        }
        if self.board_on || self.board_t < BOARD_SECS {
            let a = if self.board_on { 1.0 } else { ((BOARD_SECS - self.board_t) / 0.5).clamp(0.0, 1.0) };
            let (title, rows) = self.board();
            draw_board(hud, &title, &rows, aspect, a);
        }
    }
}

/// The leaderboard, top left: position, driver, best, last, gap, laps.
pub fn draw_board(hud: &mut crate::hud::Hud, title: &str, rows: &[Entry], aspect: f32, a: f32) {
    use crate::hud::{theme as t, Weight};
    let u = |v: f32| v / aspect;
    let fs = 0.0125;
    let lh = hud.line_h(fs) + 0.006;
    let pad = 0.013;
    let w = u(0.5);
    let h = pad * 2.0 + hud.line_h(fs) + 0.006 + lh * (rows.len().max(1) + 1) as f32;
    let (x, y) = (u(0.02), 0.025);
    hud.panel(x, y, w, h, 10.0, t::panel(0.6 * a), t::line(0.45 * a));
    let (ix, iw) = (x + u(pad), w - u(pad) * 2.0);
    hud.label(title, ix, y + pad, fs, Weight::Semibold, t::fg(a));
    // columns: right edges as fractions of the inner width
    let cols = [(0.0, "Pos", false), (0.08, "Driver", false), (0.69, "Best", true), (0.83, "Last", true), (0.93, "Gap", true), (1.0, "Laps", true)];
    let hy = y + pad + hud.line_h(fs) + 0.006;
    let cell = |hud: &mut crate::hud::Hud, i: usize, s: &str, yy: f32, wt: Weight, c: [f32; 4]| {
        let (at, _, r) = cols[i];
        if r {
            let tw = hud.label_w(s, fs, wt);
            hud.label(s, ix + iw * at - tw, yy, fs, wt, c);
        } else {
            hud.label(s, ix + iw * at, yy, fs, wt, c);
        }
    };
    for (i, (_, name, _)) in cols.iter().enumerate() {
        cell(hud, i, name, hy, Weight::Regular, t::dim(a));
    }
    if rows.is_empty() {
        cell(hud, 1, "No laps yet", hy + lh, Weight::Regular, t::dim2(a));
    }
    let fastest_last = rows.iter().filter_map(|r| r.last).fold(f32::MAX, f32::min);
    for (k, r) in rows.iter().enumerate() {
        let yy = hy + lh * (k + 1) as f32;
        if r.you {
            hud.round_rect(x + u(0.005), yy - 0.003, w - u(0.01), lh, 5.0, t::panel2(0.9 * a));
            hud.round_rect(x + u(0.005), yy - 0.003, hud.px_x(3.0), lh, 1.5, t::accent(a));
        }
        let fg = t::fg(a);
        cell(hud, 0, &format!("{}", k + 1), yy, Weight::Semibold, fg);
        cell(hud, 1, &r.name, yy, if r.you { Weight::Semibold } else { Weight::Regular }, fg);
        cell(hud, 2, &r.best.map_or("-".into(), |b| fmt(b, 3)), yy, Weight::Semibold, if k == 0 && r.best.is_some() { purple(a) } else { fg });
        cell(hud, 3, &r.last.map_or("-".into(), |b| fmt(b, 3)), yy, Weight::Regular,
             if r.last == Some(fastest_last) && rows.len() > 1 { t::ok(a) } else { t::dim(a) });
        cell(hud, 4, &r.gap.map_or(if k == 0 { "".into() } else { "-".into() }, |g| format!("+{:.3}", g)), yy, Weight::Regular, t::dim(a));
        cell(hud, 5, &format!("{}", r.laps), yy, Weight::Regular, t::dim(a));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oval() -> TimingTrack {
        // a 400 m x 200 m loop, counter-clockwise seen from +y
        let mut pts = Vec::new();
        for i in 0..200 {
            let a = i as f32 / 200.0 * std::f32::consts::TAU;
            pts.push(Vec3::new(200.0 * a.cos(), 0.0, 100.0 * a.sin()));
        }
        let c = Poly::new(pts).unwrap();
        let len = c.len;
        TimingTrack { center: c, limits: None, start: 0.0, sectors: [len / 3.0, len * 2.0 / 3.0], source: String::new(), layout: String::new() }
    }

    fn at(tt: &TimingTrack, s: f32) -> Vec3 {
        let s = s.rem_euclid(tt.len());
        let i = tt.center.project_s(s).max(1);
        let (a, b) = (tt.center.pts[i - 1], tt.center.pts[i]);
        let (c0, c1) = (tt.center.cum[i - 1], tt.center.cum[i]);
        a + (b - a) * ((s - c0) / (c1 - c0).max(1e-6))
    }

    /// drive at v m/s from s0 for secs
    fn drive(t: &mut Timer, tt: &TimingTrack, s0: f32, v: f32, secs: f32, pit: bool) -> (f32, Vec<Lap>) {
        let dt = 1.0 / 120.0;
        let mut s = s0;
        let mut laps = Vec::new();
        for _ in 0..(secs / dt) as usize {
            s += v * dt;
            if let Some(l) = t.update(tt, at(tt, s), false, pit, dt) {
                laps.push(l);
            }
        }
        (s, laps)
    }

    #[test]
    fn laps_time_and_reverse() {
        let tt = oval();
        let len = tt.len();
        let mut t = Timer::default();
        // standing start on the line, 40 m/s: a lap takes len / 40
        let (s, laps) = drive(&mut t, &tt, 1.0, 40.0, len / 40.0 * 2.0 + 1.0, false);
        assert_eq!(laps.len(), 2, "{laps:?}");
        assert!((laps[1].time - len / 40.0).abs() < 0.02, "{} vs {}", laps[1].time, len / 40.0);
        let sec = laps[1].sectors;
        assert!((sec[0].unwrap() - len / 120.0).abs() < 0.02);
        assert!(t.delta().map_or(false, |d| d.abs() < 0.05));
        // back over the line and forward again: not a lap
        let (s, _) = drive(&mut t, &tt, s, -5.0, (s - (s / len).floor() * len + 20.0) / 5.0, false);
        let (_, l2) = drive(&mut t, &tt, s, 5.0, 10.0, false);
        assert!(l2.is_empty(), "{l2:?}");
    }

    #[test]
    fn out_lap_from_pits() {
        let tt = oval();
        let len = tt.len();
        let mut t = Timer::default();
        // start mid-lap (the pits): untimed until the line
        let (s, laps) = drive(&mut t, &tt, len * 0.5, 40.0, len * 0.5 / 40.0 + 0.5, true);
        assert!(laps.is_empty());
        assert!(!t.timing, "crossing in the pit lane keeps the out lap");
        let (_, laps) = drive(&mut t, &tt, s, 40.0, len / 40.0 + 0.5, false);
        assert!(laps.is_empty(), "the out lap is not timed");
        assert!(t.timing);
    }

    #[test]
    fn formats() {
        assert_eq!(fmt(117.8124, 3), "1:57.812");
        assert_eq!(fmt(57.8, 3), "57.800");
        assert_eq!(fmt(61.05, 1), "1:01.1");
        assert_eq!(fmt_delta(-0.316), "-0.32");
    }
}
