//! Racing against AI cars: `--ai N` (EVOVIEW_AI=N) puts N opponents on the
//! track with the driven car.
//!
//! - Each opponent is its own TrackDrive (the simple model, or the rig car
//!   with EVOVIEW_AI_RIG=1) driven by a trackdrive::LineDriver with its own
//!   pace, line offset and braking point, so they do not drive nose to tail.
//! - Traffic: a car ahead in the driver's path caps its speed (follow), or,
//!   with asphalt beside it and the speed to do it, the driver moves its
//!   line over to pass. A car alongside pushes the line away from it (room
//!   for the player too).
//! - Contact: every pair of cars as oriented boxes; overlapping ones are
//!   pushed apart and trade the speed along the contact normal (a little
//!   bounce, some friction) - nobody drives through anybody.
//! - Start: the grid (spawnpoints_grid.scene, else slots off the racing
//!   line), a standing start with five red lights (EVOVIEW_AI_START=standing,
//!   the default), or rolling (=rolling: a formation 200 m back, capped to
//!   90 km/h until the leader crosses the line).
//! - Order: every car's laps + fraction along the racing line (race_order);
//!   lap timing / the leaderboard live elsewhere (branch gfx-laps).
//!
//! Knobs: EVOVIEW_AI_CARS=id1,id2 (models, default the player's), EVOVIEW_AI_PACE=lo,hi
//! (share of the line's corner speeds, default 0.9,1.0), EVOVIEW_AI_GRID_POS=n
//! (the player's grid slot, 1 = pole; default the middle), EVOVIEW_AI_RIG,
//! EVOVIEW_AI_THREADS, EVOVIEW_AI_LOG=1 (contacts / stuck / resets to stdout).

use crate::trackdrive::{Controls, LineDriver, TrackDrive};
use glam::{Vec2, Vec3};

/// AI driver names, in grid order.
pub const NAMES: [&str; 24] = [
    "M. Rossetti", "J. Lindqvist", "A. Okafor", "T. Brandt", "L. Moreau", "K. Tanaka",
    "R. Alvarez", "S. Novak", "D. Fischer", "P. Costa", "E. Kowalski", "H. Nakamura",
    "F. Bianchi", "N. Petrov", "C. Dubois", "G. Hartmann", "I. Silva", "O. Jensen",
    "B. Marino", "V. Horvat", "W. Becker", "Y. Sato", "Z. Romano", "U. Larsen",
];

pub fn env_count() -> usize {
    std::env::var("EVOVIEW_AI").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0).min(NAMES.len())
}

/// EVOVIEW_AI_CARS, else the player's car, one model per opponent (cycled).
pub fn models(player: &str, n: usize) -> Vec<String> {
    let list: Vec<String> = std::env::var("EVOVIEW_AI_CARS").ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
        .filter(|l: &Vec<String>| !l.is_empty())
        .unwrap_or_else(|| vec![player.to_string()]);
    (0..n).map(|i| list[i % list.len()].clone()).collect()
}

/// The racing line with its running distance, for progress.
pub struct Line {
    pub pts: Vec<Vec3>,
    cum: Vec<f32>,
    pub len: f32,
    /// how far off the line a car may go here, m: the line's curvature
    /// over +-15 m (an offset in a corner changes its radius - inside, the
    /// driver's speed for the line is too fast)
    room: Vec<f32>,
}

impl Line {
    fn new(pts: Vec<Vec3>) -> Line {
        let n = pts.len();
        let mut cum = vec![0.0; n];
        for i in 1..n {
            cum[i] = cum[i - 1] + pts[i - 1].distance(pts[i]);
        }
        let len = if n > 1 { cum[n - 1] + pts[n - 1].distance(pts[0]) } else { 1.0 };
        let mut l = Line { pts, cum, len: len.max(1.0), room: Vec::new() };
        if n > 8 {
            l.room = (0..n).map(|i| {
                let (a, b) = (l.walk(i, -15.0), l.walk(i, 15.0));
                let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z).normalize_or_zero();
                let (u, v) = (flat(l.pts[i] - l.pts[a]), flat(l.pts[b] - l.pts[i]));
                let k = u.dot(v).clamp(-1.0, 1.0).acos() / 30.0;
                // R 180 m and wider: 4.5 m; R 60 m: 1.5 m
                (0.025 / k.max(1e-4)).clamp(0.5, 4.5)
            }).collect();
        }
        l
    }

    /// The room off the line over the `d` m from point `j`.
    fn room_ahead(&self, j: usize, d: f32) -> f32 {
        if self.room.is_empty() {
            return 4.5;
        }
        let n = self.pts.len();
        let (mut j, mut s, mut r) = (j, 0.0, f32::MAX);
        while s < d {
            r = r.min(self.room[j]);
            let k = (j + 1) % n;
            s += self.pts[j].distance(self.pts[k]);
            j = k;
        }
        r
    }

    fn left(&self, j: usize) -> Vec3 {
        let n = self.pts.len();
        let (p0, p1) = (self.pts[(j + n - 1) % n], self.pts[(j + 1) % n]);
        let t = Vec3::new(p1.x - p0.x, 0.0, p1.z - p0.z).normalize_or_zero();
        Vec3::new(t.z, 0.0, -t.x)
    }

    fn nearest(&self, p: Vec3, from: usize) -> usize {
        let n = self.pts.len();
        let mut best = (f32::MAX, from);
        for k in 0..80 {
            let j = (from + n - 10 + k) % n;
            let d = self.pts[j].distance_squared(p);
            if d < best.0 {
                best = (d, j);
            }
        }
        if best.0 > 20.0 * 20.0 {
            for j in 0..n {
                let d = self.pts[j].distance_squared(p);
                if d < best.0 {
                    best = (d, j);
                }
            }
        }
        best.1
    }

    /// The point `d` metres along the line from point `j` (negative: back).
    fn walk(&self, j: usize, d: f32) -> usize {
        let n = self.pts.len();
        let (mut j, mut left) = (j, d.abs());
        while left > 0.0 {
            let k = if d >= 0.0 { (j + 1) % n } else { (j + n - 1) % n };
            left -= self.pts[j].distance(self.pts[k]);
            j = k;
        }
        j
    }
}

/// Where a car is along the line: its nearest point, laps done.
#[derive(Clone, Copy, Debug)]
struct Progress {
    i: usize,
    laps: i32,
}

impl Progress {
    fn start(line: &Line, p: Vec3) -> Progress {
        let i = line.nearest(p, 0);
        // a grid slot behind the line has not started its first lap
        let laps = if line.cum[i] > line.len * 0.5 { -1 } else { 0 };
        Progress { i, laps }
    }

    fn update(&mut self, line: &Line, p: Vec3) {
        let j = line.nearest(p, self.i);
        let (a, b) = (line.cum[self.i] / line.len, line.cum[j] / line.len);
        if a > 0.75 && b < 0.25 {
            self.laps += 1;
        } else if a < 0.25 && b > 0.75 {
            self.laps -= 1;
        }
        self.i = j;
    }

    fn value(&self, line: &Line) -> f32 {
        self.laps as f32 + line.cum[self.i] / line.len
    }
}

pub struct Ai {
    pub td: TrackDrive,
    pub drv: LineDriver,
    pub name: &'static str,
    /// the render slot (Scene::slots index, 1..) and the instance in it
    pub slot: usize,
    pub inst: usize,
    /// the driver's own line offset, m
    base_off: f32,
    /// a pass under way: the offset it took and seconds it is held for
    pass: Option<(f32, f32)>,
    /// seconds without making 5 m of progress along the line, and the
    /// best progress (m) it is measured from
    slow: f32,
    best: f32,
    /// seconds more than 20 m off the line (a gravel trap, a field)
    far: f32,
    ctl: Controls,
    /// tyre smoke 0..1, eased toward TrackDrive::smoke_target
    pub smoke: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Start {
    Standing,
    Rolling,
}

/// One contact or stuck event, for the drive-test log.
#[derive(Default, Debug, Clone, Copy)]
pub struct Stats {
    pub contacts: u32,
    pub contacts_player: u32,
    /// pairs still overlapping more than 0.3 m after the push (pass-through)
    pub deep: u32,
    pub stuck: u32,
    pub resets: u32,
}

pub struct Race {
    pub ai: Vec<Ai>,
    pub line: Line,
    /// 0 the player, 1.. the AI
    prog: Vec<Progress>,
    /// seconds since the race was set up
    pub t: f32,
    pub start: Start,
    /// standing: the lights go out at this time; None until they have
    lights_out: f32,
    /// the race is on (lights out / green flag), at race time
    pub go: Option<f32>,
    pub stats: Stats,
    log: bool,
    threads: usize,
    /// contacts already logged, per pair, until this time
    quiet: std::collections::HashMap<(usize, usize), f32>,
    /// the radar: None = automatic (a car near), Some(on) forced
    pub radar: Option<bool>,
    /// online (mp.rs): the cars are other players, placed from the server
    /// each frame - nothing here drives them
    pub net: bool,
}

/// A value in 0..1 from an index and a salt, the same every run.
fn hash01(i: usize, salt: u32) -> f32 {
    let mut x = (i as u32).wrapping_mul(0x9E37_79B9) ^ salt.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^= x >> 12;
    x = x.wrapping_mul(0x297A_2D39);
    x ^= x >> 15;
    (x & 0xFFFF) as f32 / 65535.0
}

impl Race {
    /// `cars`: one TrackDrive per opponent with its render slot / instance.
    /// Places everyone (the player too) on the grid or in the formation.
    pub fn new(line: Vec<Vec3>, cars: Vec<(TrackDrive, usize, usize)>, grid: &[crate::pits::Spot], player: &mut TrackDrive) -> Race {
        let line = Line::new(line);
        let start = match std::env::var("EVOVIEW_AI_START").as_deref() {
            Ok("rolling") => Start::Rolling,
            _ => Start::Standing,
        };
        let (lo, hi) = std::env::var("EVOVIEW_AI_PACE").ok()
            .and_then(|v| {
                let mut it = v.split(',').filter_map(|x| x.trim().parse::<f32>().ok());
                let a = it.next()?;
                Some((a, it.next().unwrap_or(a)))
            })
            .unwrap_or((0.9, 1.0));
        let n = cars.len();
        let mut ai: Vec<Ai> = cars.into_iter().enumerate().map(|(i, (td, slot, inst))| {
            let mut drv = LineDriver::new(line.pts.clone());
            drv.pace = lo + (hi - lo) * hash01(i, 1);
            // EVOVIEW_AI_VARY=0: every driver on the line itself, braking alike
            let vary = std::env::var("EVOVIEW_AI_VARY").map_or(1.0, |v| v.parse::<f32>().unwrap_or(1.0));
            drv.brake_k = 1.0 + (0.15 * hash01(i, 2) - 0.1) * vary;
            let base_off = (hash01(i, 3) - 0.5) * 1.2 * vary;
            drv.offset = base_off;
            drv.offset_target = base_off;
            Ai { td, drv, name: NAMES[i % NAMES.len()], slot, inst, base_off, pass: None, slow: 0.0, best: f32::MIN, far: 0.0, ctl: Controls::default(), smoke: 0.0 }
        }).collect();
        // the player's slot on the grid, 0-based (default: the middle)
        let pp = std::env::var("EVOVIEW_AI_GRID_POS").ok().and_then(|v| v.parse::<usize>().ok())
            .map_or(n / 2, |p| p.max(1) - 1).min(n);
        let spots = Self::slots(&line, grid, n + 1, start);
        let mut k = 0;
        for (s, &(p, yaw)) in spots.iter().enumerate() {
            // a rolling start rolls: the formation at 80 km/h
            let roll = if start == Start::Rolling { 22.0 } else { 0.0 };
            if s == pp {
                player.place_from(p, yaw, 2.0);
                if roll > 0.0 {
                    player.roll(roll);
                }
                continue;
            }
            if let Some(a) = ai.get_mut(k) {
                if !a.td.place_from(p, yaw, 2.0) {
                    a.td.place(p, yaw);
                }
                if roll > 0.0 {
                    a.td.roll(roll);
                }
                a.drv.snap(a.td.pos);
                // start on the grid slot's own line, easing over to the
                // driver's (not all of them turning for the line at once)
                let o = lateral(&line, a.td.pos);
                a.drv.offset = o;
            }
            k += 1;
        }
        let mut prog = vec![Progress::start(&line, player.pos)];
        prog.extend(ai.iter().map(|a| Progress::start(&line, a.td.pos)));
        let threads = std::env::var("EVOVIEW_AI_THREADS").ok().and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(if n >= 4 { 3 } else { 1 }).clamp(1, 8);
        println!("race: {n} AI, {:?} start, player P{} of {}, pace {lo:.2}-{hi:.2}, {} model, {} thread(s)",
                 start, pp + 1, n + 1, if ai.first().map_or(false, |a| a.td.rig.is_some()) { "rig" } else { "simple" }, threads);
        for (i, a) in ai.iter().enumerate() {
            let aids = a.td.rig.as_ref().map_or(String::new(), |(rc, _)| format!("  bias {:.1} TC {:?} ABS {:?}", rc.brake_bias() * 100.0,
                crate::aids::level(rc, crate::aids::Knob::Tc), crate::aids::level(rc, crate::aids::Knob::Abs)));
            println!("  AI {:2} {:14} {}  pace {:.3} brake x{:.2} line {:+.2} m{aids}", i + 1, a.name, a.td.slot.model, a.drv.pace, a.drv.brake_k, a.base_off);
        }
        // the lights: 2 s to settle, one red a second, out 0.4..2.4 s after the fifth
        let lights_out = 2.0 + 5.0 + 0.4 + 2.0 * hash01(n, 7);
        Race {
            ai,
            line,
            prog,
            t: 0.0,
            start,
            lights_out,
            go: None,
            stats: Stats::default(),
            log: std::env::var_os("EVOVIEW_AI_LOG").is_some(),
            threads,
            quiet: Default::default(),
            radar: None,
            net: false,
        }
    }

    /// `n` starting places: the track's grid when it has enough slots (and
    /// the start is standing), else two staggered columns down the racing
    /// line behind its start (rolling: 200 m back).
    fn slots(line: &Line, grid: &[crate::pits::Spot], n: usize, start: Start) -> Vec<(Vec3, f32)> {
        // a grid only if its slots are on this line: Spa's sit 1,100 m
        // under the track (a container transform), the Nordschleife layout
        // gets the GP circuit's - cars spawned 300 m off and fell / reset
        let off = |p: Vec3| line.pts.iter().map(|q| q.distance(p)).fold(f32::MAX, f32::min);
        let worst = grid.iter().take(n).map(|s| off(s.pos)).fold(0.0f32, f32::max);
        if start == Start::Standing && grid.len() >= n && worst < 25.0 {
            println!("race: grid from the track ({} slots)", grid.len());
            return grid.iter().take(n).map(|s| (s.pos, s.yaw)).collect();
        }
        if start == Start::Standing && grid.len() >= n {
            println!("race: the track's grid is {worst:.0} m off the racing line - slots off the line instead");
        }
        let np = line.pts.len();
        let back = if start == Start::Rolling { 200.0 } else { 12.0 };
        (0..n).map(|k| {
            let j = line.walk(0, -(back + 8.0 * k as f32));
            let side = if k % 2 == 0 { 2.2 } else { -2.2 };
            let p = line.pts[j] + line.left(j) * side;
            let d = line.pts[(j + 3) % np] - line.pts[j];
            (p, d.x.atan2(d.z))
        }).collect()
    }

    /// Online (mp.rs): one more car (a late joiner's), drawn from render
    /// slot `slot` (1..), instance `inst`; returns its opponent index.
    pub fn add_net(&mut self, td: TrackDrive, slot: usize, inst: usize) -> usize {
        let i = self.ai.len();
        let p = Progress::start(&self.line, td.pos);
        let drv = LineDriver::new(self.line.pts.clone());
        self.ai.push(Ai { td, drv, name: NAMES[i % NAMES.len()], slot, inst, base_off: 0.0, pass: None, slow: 0.0,
                          best: f32::MIN, far: 0.0, ctl: Controls::default(), smoke: 0.0 });
        self.prog.push(p);
        i
    }

    /// The start-light state for the HUD: (lit 0..5, out). Standing only.
    pub fn lights(&self) -> Option<(u32, bool)> {
        if self.start != Start::Standing {
            return None;
        }
        match self.go {
            Some(g) if self.t - g > 3.0 => None,
            Some(_) => Some((0, true)),
            None => Some((((self.t - 2.0).floor() + 1.0).clamp(0.0, 5.0) as u32, false)),
        }
    }

    /// The rolling start's banner: true until the green flag (and 3 s after).
    pub fn rolling_banner(&self) -> Option<bool> {
        if self.start != Start::Rolling {
            return None;
        }
        match self.go {
            Some(g) if self.t - g > 3.0 => None,
            Some(_) => Some(true),
            None => Some(false),
        }
    }

    /// Whether the cars are held on the grid (standing start, lights on).
    pub fn held(&self) -> bool {
        self.go.is_none() && self.start == Start::Standing
    }

    /// Hold the player on the grid until the lights go out: the handbrake,
    /// no pedals (an automatic box takes the brake pedal at a standstill as
    /// reverse - the car would back into the one behind). Behind the
    /// rolling start's pace the throttle is governed to the field's 90 km/h
    /// (`kmh` = the player's speed), like the pit limiter.
    pub fn filter_player(&self, c: &mut Controls, kmh: f32) {
        if self.held() {
            c.throttle = 0.0;
            c.brake = 0.0;
            c.handbrake = true;
            c.handbrake_v = Some(1.0);
        }
        if self.go.is_none() && self.start == Start::Rolling {
            let target = 90.0 - 1.0;
            let cap = ((target - kmh.abs()) / 3.6 * 1.2 + 0.15).clamp(0.0, 1.0);
            c.throttle = c.throttle.min(cap);
        }
    }

    /// The frame the race went (lights out / green flag).
    pub fn just_started(&self) -> bool {
        self.go == Some(self.t)
    }

    /// Positions in race order: (car, laps done + fraction of this one);
    /// car 0 is the player, k the AI `ai[k - 1]`. For the leaderboard.
    pub fn race_order(&self) -> Vec<(usize, f32)> {
        let mut v: Vec<(usize, f32)> = self.prog.iter().enumerate().map(|(i, p)| (i, p.value(&self.line))).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    /// Car k's name ("You" for the player).
    pub fn name(&self, k: usize) -> &str {
        if k == 0 { "You" } else { self.ai.get(k - 1).map_or("?", |a| a.name) }
    }

    /// One frame, after the player's car has stepped: the opponents'
    /// driving, their physics, contacts with everyone, progress.
    pub fn step(&mut self, player: &mut TrackDrive, dt: f32) {
        let dt = dt.clamp(0.0, 0.1);
        self.t += dt;
        if self.net {
            return;
        }
        // the start
        if self.go.is_none() {
            match self.start {
                Start::Standing if self.t >= self.lights_out => {
                    self.go = Some(self.t);
                    println!("race: lights out at {:.1} s", self.t);
                }
                Start::Rolling if self.prog.iter().any(|p| p.laps >= 0 && self.line.cum[p.i] / self.line.len < 0.5) => {
                    self.go = Some(self.t);
                    println!("race: green flag at {:.1} s", self.t);
                }
                _ => {}
            }
        }
        self.traffic(player, dt);
        // controls and physics: each car alone, so in parallel
        let held = self.held();
        let work = |a: &mut Ai| {
            a.ctl = if held {
                a.drv.rest();
                Controls { handbrake: true, auto_box: true, ..Default::default() }
            } else {
                a.drv.controls(&a.td, dt)
            };
            // stalled (a spin): the starter, as a driver would
            a.ctl.starter = a.td.rig.as_ref().map_or(false, |(rc, _)| !rc.running);
            a.td.step(&a.ctl, dt);
            a.smoke += (a.td.smoke_target() - a.smoke) * (dt * 6.0).min(1.0);
        };
        let _t = crate::bench::cpu("ai physics");
        if self.threads <= 1 || self.ai.len() < 2 {
            self.ai.iter_mut().for_each(work);
        } else {
            let per = self.ai.len().div_ceil(self.threads);
            std::thread::scope(|s| {
                for chunk in self.ai.chunks_mut(per) {
                    s.spawn(move || chunk.iter_mut().for_each(work));
                }
            });
        }
        drop(_t);
        // EVOVIEW_AI_CONTACT=0: cars pass through each other (A/B)
        if std::env::var("EVOVIEW_AI_CONTACT").map_or(true, |v| v != "0") {
            self.contacts(player);
        }
        self.prog[0].update(&self.line, player.pos);
        for (k, a) in self.ai.iter().enumerate() {
            self.prog[k + 1].update(&self.line, a.td.pos);
        }
        self.unstick(player, dt);
    }

    /// Each AI's speed cap and line offset from the cars round it.
    fn traffic(&mut self, player: &TrackDrive, dt: f32) {
        // every car: position, velocity, progress (m), its offset off the line
        let cars: Vec<(Vec3, Vec3, f32, f32)> = std::iter::once(player)
            .chain(self.ai.iter().map(|a| &a.td))
            .zip(self.prog.iter())
            .map(|(td, p)| {
                let off = (td.pos - self.line.pts[p.i]).dot(self.line.left(p.i));
                (td.pos, td.vel(), p.value(&self.line) * self.line.len, off)
            })
            .collect();
        let rolling = self.go.is_none() && self.start == Start::Rolling;
        let len = self.line.len;
        for k in 0..self.ai.len() {
            let me = k + 1;
            let (pos, vel, s, _) = cars[me];
            let v = vel.length();
            let mut cap = if rolling { 25.0 } else { f32::INFINITY };
            let (cur, at, td) = (self.ai[k].drv.offset, self.ai[k].drv.progress(), &self.ai[k].td);
            let mut pass = self.ai[k].pass;
            if let Some((_, t)) = pass.as_mut() {
                *t -= dt;
            }
            if pass.map_or(false, |(_, t)| t <= 0.0) {
                pass = None;
            }
            let mut want = pass.map_or(self.ai[k].base_off, |p| p.0);
            let look = 10.0 + v * 1.4;
            for (b, &(bp, bv, bs, boff)) in cars.iter().enumerate() {
                if b == me || bp.distance_squared(pos) > (look + 20.0).powi(2) {
                    continue;
                }
                // gap along the line, wrapped to the near side
                let mut ds = bs - s;
                ds -= (ds / len).round() * len;
                let vb = bv.dot(vel.normalize_or_zero());
                if ds.abs() < 5.5 {
                    // alongside: keep a car's width of air between us
                    let dl = boff - cur;
                    if dl.abs() < 2.9 {
                        want = boff - dl.signum() * 3.0;
                    }
                } else if ds > 0.0 && ds < look {
                    // ahead and in the way of where we are heading
                    if (boff - want).abs() < 2.4 {
                        let faster = v > vb + 1.0;
                        let side = faster.then(|| {
                            let ahead = self.line.walk(at, ds + 10.0);
                            [boff + 3.2, boff - 3.2].into_iter()
                                .filter(|o| o.abs() < 4.5)
                                .filter(|o| room(&self.line, td, ahead, *o) && room(&self.line, td, at, *o))
                                .min_by(|x, y| (x - want).abs().total_cmp(&(y - want).abs()))
                        }).flatten();
                        match side {
                            // room to move over before reaching it
                            Some(o) if ds > 7.0 || (o - cur).abs() < 1.0 => {
                                want = o;
                                pass = Some((o, 2.0));
                            }
                            _ => {
                                // follow: close to a safe gap at its speed
                                let safe = 7.0 + v * 0.35;
                                cap = cap.min((vb + (ds - safe) * 0.5).max(0.0));
                            }
                        }
                    }
                }
            }
            // never off the asphalt to make room (a chicane's kerbs and
            // bollards): halve back toward the driver's own line until there
            // is road there, here and a little ahead
            let base = self.ai[k].base_off;
            let ahead = self.line.walk(at, 8.0 + v * 0.6);
            let lim = self.line.room_ahead(at, 15.0 + v * 1.2);
            let mut want = want.clamp(-lim, lim);
            for _ in 0..4 {
                if (want - base).abs() < 0.3 || (room(&self.line, td, at, want) && room(&self.line, td, ahead, want)) {
                    break;
                }
                want = base + (want - base) * 0.5;
            }
            let a = &mut self.ai[k];
            a.pass = pass;
            a.drv.offset_target = want;
            a.drv.vcap = cap;
        }
    }

    /// Cars as boxes: push overlapping pairs apart and trade speed.
    fn contacts(&mut self, player: &mut TrackDrive) {
        let n = self.ai.len() + 1;
        let names: Vec<&'static str> = std::iter::once("You").chain(self.ai.iter().map(|a| a.name)).collect();
        for i in 0..n {
            for j in i + 1..n {
                let (a, b): (&mut TrackDrive, &mut TrackDrive) = if i == 0 {
                    (player, &mut self.ai[j - 1].td)
                } else {
                    let (x, y) = self.ai.split_at_mut(j - 1);
                    (&mut x[i - 1].td, &mut y[0].td)
                };
                if a.pos.distance_squared(b.pos) > 8.0 * 8.0 || (a.pos.y - b.pos.y).abs() > 3.0 {
                    continue;
                }
                let Some((nrm, depth, cp)) = overlap(a, b) else { continue };
                let (ma, mb) = (a.mass_kg(), b.mass_kg());
                let (wa, wb) = (mb / (ma + mb), ma / (ma + mb));
                let rel = b.vel() - a.vel();
                let vn = rel.dot(nrm);
                let (mut dva, mut dvb, mut dra, mut drb) = (Vec3::ZERO, Vec3::ZERO, 0.0, 0.0);
                if vn < 0.0 {
                    // restitution 0.1, sliding friction 0.3 of the normal impulse
                    let inv = 1.0 / ma + 1.0 / mb;
                    let jn = -(1.1 * vn) / inv;
                    let vt = rel - nrm * vn;
                    let jt = (0.3 * jn).min(vt.length() / inv);
                    let imp = nrm * jn + vt.normalize_or_zero() * jt;
                    dva = -imp / ma;
                    dvb = imp / mb;
                    // a yaw kick from where it hit, half strength (the box is
                    // a coarse body, and a full kick spun cars at a tap)
                    let iz = |m: f32, td: &TrackDrive| m * (td.a + td.b).powi(2) * 0.28;
                    let kick = |td: &TrackDrive, j: Vec3| {
                        let r = cp - td.pos;
                        (r.z * j.x - r.x * j.z) * 0.5
                    };
                    dra = kick(a, -imp) / iz(ma, a);
                    drb = kick(b, imp) / iz(mb, b);
                    let key = (i, j);
                    if self.quiet.get(&key).map_or(true, |&t| self.t > t) {
                        self.quiet.insert(key, self.t + 1.0);
                        self.stats.contacts += 1;
                        if i == 0 {
                            self.stats.contacts_player += 1;
                        }
                        if self.log {
                            println!("CONTACT t {:6.2}  {} <-> {}  {:4.1} km/h closing, {:.2} m deep at ({:.0} {:.0})",
                                     self.t, names[i], names[j], -vn * 3.6, depth, cp.x, cp.z);
                        }
                    }
                }
                a.nudge(-nrm * depth * wa, dva, dra);
                b.nudge(nrm * depth * wb, dvb, drb);
                if let Some((_, d2, _)) = overlap(a, b) {
                    if d2 > 0.3 {
                        self.stats.deep += 1;
                        if self.log {
                            println!("DEEP t {:6.2}  {} <-> {} still {:.2} m inside", self.t, names[i], names[j], d2);
                        }
                    }
                }
            }
        }
    }

    /// A car that has not got 5 m further along the line in 8 s is stuck
    /// (logged: beached, in a wall, backing and filling); at 15 s it is put
    /// back on the line where nobody is.
    fn unstick(&mut self, player: &TrackDrive, dt: f32) {
        if self.go.is_none() {
            return;
        }
        let others: Vec<Vec3> = std::iter::once(player.pos).chain(self.ai.iter().map(|a| a.td.pos)).collect();
        for k in 0..self.ai.len() {
            let m = self.prog[k + 1].value(&self.line) * self.line.len;
            let a = &mut self.ai[k];
            if m > a.best + 5.0 {
                a.best = m;
                a.slow = 0.0;
            } else {
                a.slow += dt;
            }
            if a.slow > 8.0 && a.slow - dt <= 8.0 {
                self.stats.stuck += 1;
                let rig = a.td.rig.as_ref().map_or(String::new(), |(rc, _)| format!(" running {} clutch {:.2} manual {}", rc.running, rc.clutch_e, rc.manual));
                let near = others.iter().enumerate().filter(|(o, _)| *o != k + 1).map(|(_, p)| p.distance(a.td.pos)).fold(f32::MAX, f32::min);
                let wall = { let mut h = Vec::new(); a.td.ground.walls_near(a.td.pos, 4.0, a.td.pos.y + 0.1, a.td.pos.y + 1.2, &mut h); h.iter().map(|w| w.1).fold(f32::MAX, f32::min) };
                println!("STUCK t {:6.1}  {} at ({:.0} {:.0} {:.0}) car {near:.1} m wall {wall:.1} m on {} line {:.1} m v {:.1} gear {} rpm {:.0} ctl thr {:.2} brk {:.2} steer {:+.2} hb {} vcap {:.1} off {:+.1}->{:+.1}{rig}", self.t, a.name, a.td.pos.x, a.td.pos.y, a.td.pos.z,
                         a.td.ground.probe(a.td.pos.x, a.td.pos.z, a.td.pos.y + 1.0).map_or("nothing".to_string(), |h| format!("{}/{:.2}", a.td.ground.surf_name(h.surf), h.n.y)),
                         self.line.pts.iter().map(|p| Vec3::new(p.x - a.td.pos.x, 0.0, p.z - a.td.pos.z).length()).fold(f32::MAX, f32::min),
                         a.td.speed_kmh(),
                         a.td.gear, a.td.rpm, a.ctl.throttle, a.ctl.brake, a.ctl.steer, a.ctl.handbrake, a.drv.vcap, a.drv.offset, a.drv.offset_target);
            }
            // 20 m off the line for 6 s: it is not finding its way back
            let off = Vec3::new(a.td.pos.x - self.line.pts[self.prog[k + 1].i].x, 0.0, a.td.pos.z - self.line.pts[self.prog[k + 1].i].z).length();
            a.far = if off > 20.0 { a.far + dt } else { 0.0 };
            if a.slow > 10.0 || a.far > 6.0 {
                a.far = 0.0;
                // back on the line 30 m on, further while another car is near
                // (after 10 s: 15 left wedged cars out for a quarter lap)
                let mut j = self.line.walk(self.prog[k + 1].i, 30.0);
                for _ in 0..20 {
                    if others.iter().enumerate().all(|(o, p)| o == k + 1 || p.distance(self.line.pts[j]) > 12.0) {
                        break;
                    }
                    j = self.line.walk(j, 10.0);
                }
                let np = self.line.pts.len();
                let d = self.line.pts[(j + 3) % np] - self.line.pts[j];
                a.td.place_from(self.line.pts[j], d.x.atan2(d.z), 2.0);
                a.drv.snap(a.td.pos);
                a.slow = 0.0;
                a.best = f32::MIN;
                self.stats.resets += 1;
                println!("RESET t {:6.1}  {} back on the line at ({:.0} {:.0})", self.t, a.name, a.td.pos.x, a.td.pos.z);
            }
        }
    }

    /// Cars within `r` m of the player, in its frame: (x right, y forward)
    /// of the centre, heading (rad, 0 = same way, + = turned right), half
    /// width and half length.
    pub fn near(&self, player: &TrackDrive, r: f32) -> Vec<(Vec2, f32, f32, f32)> {
        self.near_of(player, None, r)
    }

    /// `near` around opponent `view` (online: the player the camera follows)
    /// when given: the other cars then include the player's own.
    pub fn near_of(&self, player: &TrackDrive, view: Option<usize>, r: f32) -> Vec<(Vec2, f32, f32, f32)> {
        let me = view.and_then(|k| self.ai.get(k)).map_or(player, |a| &a.td);
        let (f, l) = (me.forward(), me.left());
        let (_, pzf, pzr) = me.footprint();
        let pc = me.pos + f * (pzf + pzr) * 0.5;
        let others = self.ai.iter().enumerate().filter(|(k, _)| Some(*k) != view).map(|(_, a)| &a.td)
            .chain(view.is_some().then_some(player));
        others.filter_map(|o| {
            let (hw, zf, zr) = o.footprint();
            let c = o.pos + o.forward() * (zf + zr) * 0.5;
            let d = c - pc;
            if d.length_squared() > r * r || d.y.abs() > 4.0 {
                return None;
            }
            let af = o.forward();
            let head = (-af.dot(l)).atan2(af.dot(f));
            Some((Vec2::new(-d.dot(l), d.dot(f)), head, hw, (zf - zr) * 0.5))
        }).collect()
    }
}

/// How far `p` is to the left of the line (m) at its nearest point.
pub fn lateral(line: &Line, p: Vec3) -> f32 {
    let j = line.nearest(p, 0);
    (p - line.pts[j]).dot(line.left(j))
}

/// Asphalt (or kerb) at `off` m beside line point `j`, under the car's level.
fn room(line: &Line, td: &TrackDrive, j: usize, off: f32) -> bool {
    let p = line.pts[j] + line.left(j) * off;
    let road = td.ground.probe(p.x, p.z, p.y + 2.0).map_or(false, |h| (h.y - p.y).abs() < 1.0 && td.ground.grip(h.surf) > 0.9);
    // asphalt is not enough: Laguna's pit lane is asphalt too, the other
    // side of the pit wall. No wall between the line and a car's width past
    // the spot
    road && !td.ground.wall_across(line.pts[j], line.pts[j] + line.left(j) * (off + off.signum() * 1.2), p.y - 0.3, p.y + 1.0)
}

/// Two cars' footprints as oriented boxes (SAT): the normal from a to b,
/// the depth along it, and a contact point. None: apart.
fn overlap(a: &TrackDrive, b: &TrackDrive) -> Option<(Vec3, f32, Vec3)> {
    let bx = |td: &TrackDrive| {
        let (hw, zf, zr) = td.footprint();
        let (f, l) = (td.forward(), td.left());
        (td.pos + f * (zf + zr) * 0.5, f, l, (zf - zr) * 0.5, hw)
    };
    let (ca, fa, la, ha, wa) = bx(a);
    let (cb, fb, lb, hb, wb) = bx(b);
    let d = Vec3::new(cb.x - ca.x, 0.0, cb.z - ca.z);
    let mut best = (f32::MAX, Vec3::X);
    for ax in [fa, la, fb, lb] {
        let ra = ha * fa.dot(ax).abs() + wa * la.dot(ax).abs();
        let rb = hb * fb.dot(ax).abs() + wb * lb.dot(ax).abs();
        let dist = d.dot(ax);
        let o = ra + rb - dist.abs();
        if o <= 0.0 {
            return None;
        }
        if o < best.0 {
            best = (o, if dist >= 0.0 { ax } else { -ax });
        }
    }
    let n = best.1;
    // the contact: a's surface along the normal, toward b
    let ra = ha * fa.dot(n).abs() + wa * la.dot(n).abs();
    Some((n, best.0, Vec3::new(ca.x, a.pos.y, ca.z) + n * (ra - best.0 * 0.5)))
}

/// `--drive-test` with AI: the player's car on its own racing-line driver,
/// everyone stepped at 120 Hz for `secs`; prints the order every 10 s, laps,
/// contacts, deep overlaps and stuck cars.
pub fn test_run(race: &mut Race, player: &mut TrackDrive, mut drv: LineDriver, secs: f32) {
    let dt = 1.0 / 120.0;
    let mut t = 0.0;
    let mut next = 0.0;
    drv.snap(player.pos);
    drv.offset = lateral(&race.line, player.pos);
    drv.pace = std::env::var("EVOVIEW_PLAYER_PACE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.95);
    let mut laps_seen = vec![i32::MIN; race.ai.len() + 1];
    let mut prev_y: Vec<f32> = std::iter::once(player.pos.y).chain(race.ai.iter().map(|a| a.td.pos.y)).collect();
    let wall = std::time::Instant::now();
    let mut ai_ms = 0.0f64;
    // per AI: the most tyre smoke and seconds with some (> 0.2)
    let mut smoke = vec![(0.0f32, 0.0f32); race.ai.len()];
    while t < secs {
        if race.held() {
            drv.rest();
        }
        let mut c = drv.controls(player, dt);
        race.filter_player(&mut c, player.speed_kmh());
        player.step(&c, dt);
        let t0 = std::time::Instant::now();
        race.step(player, dt);
        ai_ms += t0.elapsed().as_secs_f64() * 1000.0;
        t += dt;
        // a car going up fast is a launch (walls, contacts)
        for (k, td) in std::iter::once(&*player).chain(race.ai.iter().map(|a| &a.td)).enumerate() {
            let vy = (td.pos.y - prev_y.get(k).copied().unwrap_or(td.pos.y)) / dt;
            if let Some(p) = prev_y.get_mut(k) { *p = td.pos.y; }
            // EVOVIEW_Y_LOG="k,t0,t1": car k's height every step in a window
            if let Some((yk, t0, t1)) = std::env::var("EVOVIEW_Y_LOG").ok().and_then(|v| {
                let p: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect();
                (p.len() == 3).then(|| (p[0] as usize, p[1], p[2]))
            }) {
                if yk == k && t >= t0 && t <= t1 {
                    let w = td.rig.as_ref().map_or(String::new(), |(rc, _)| format!(" cg {:.2} v ({:+.1} {:+.1} {:+.1}) w ({:+.1} {:+.1} {:+.1}) up {:.2} fz [{:.0} {:.0} {:.0} {:.0}]",
                        rc.cg.y, rc.v.x, rc.v.y, rc.v.z, rc.w.x, rc.w.y, rc.w.z, rc.up().y, rc.c[0].fz, rc.c[1].fz, rc.c[2].fz, rc.c[3].fz));
                    println!("YLOG t {t:7.3} y {:.3}{w}", td.pos.y);
                }
            }
            if vy > 10.0 {
                println!("LAUNCH t {t:6.2}  {} up {vy:.1} m/s at ({:.0} {:.1} {:.0}) {:.0} km/h", race.name(k), td.pos.x, td.pos.y, td.pos.z, td.speed_kmh());
            }
        }
        for (s, a) in smoke.iter_mut().zip(&race.ai) {
            s.0 = s.0.max(a.smoke);
            if a.smoke > 0.2 {
                s.1 += dt;
            }
        }
        for (k, p) in race.prog.iter().enumerate() {
            if laps_seen[k] != i32::MIN && p.laps > laps_seen[k] && p.laps >= 1 {
                println!("LAPDONE t {t:6.1}  {} completes lap {}", race.name(k), p.laps);
            }
            laps_seen[k] = p.laps;
        }
        // EVOVIEW_AI_TRACE=k: car k (0 the player) four times a second
        if let Some(k) = std::env::var("EVOVIEW_AI_TRACE").ok().and_then(|v| v.parse::<usize>().ok()).filter(|_| (t * 4.0).fract() < dt * 4.0) {
            let (td, c, extra) = if k == 0 { (&*player, c, String::new()) } else {
                let a = &race.ai[k - 1];
                (&a.td, a.ctl, format!(" vcap {:.1} off {:+.1}->{:+.1} smoke {:.2} slip_r {:.3} spin {:.2}", a.drv.vcap, a.drv.offset, a.drv.offset_target,
                                        a.smoke, a.td.snd.slip_r, a.td.snd.spin) + &{
                    let (st, b, r) = a.drv.recovery();
                    let cl = a.td.rig.as_ref().map_or(String::new(), |(rc, _)| format!(" ce {:.2} run {}", rc.clutch_e, rc.running));
                    format!(" rec {st:.1}/{b:.1}/{r:.1} vx {:.1}{cl}", a.td.vx)
                })
            };
            let line = race.line.pts.iter().map(|p| Vec3::new(p.x - td.pos.x, 0.0, p.z - td.pos.z).length()).fold(f32::MAX, f32::min);
            println!("TRACE t {t:6.2} {} ({:.1} {:.1} {:.1}) v {:.1} gear {} rpm {:.0} thr {:.2} brk {:.2} steer {:+.2} line {line:.1}{extra}", race.name(k), td.pos.x, td.pos.y, td.pos.z, td.speed_kmh(), td.gear, td.rpm, c.throttle, c.brake, c.steer);
        }
        if t >= next {
            next += 10.0;
            let order: Vec<String> = race.race_order().iter().map(|(k, v)| {
                let kmh = if *k == 0 { player.speed_kmh() } else { race.ai[k - 1].td.speed_kmh() };
                format!("{}:{:.2}@{:.0}", race.name(*k), v, kmh)
            }).collect();
            let pr = player.rig.as_ref().map_or(String::new(), |(rc, _)| format!(" running {} clutch {:.2} hb {:.2}", rc.running, rc.clutch_e, rc.hb));
            println!("ORDER t {t:5.1}  {}   [player gear {} rpm {:.0} thr {:.2} brk {:.2}{pr}]", order.join("  "), player.gear, player.rpm, c.throttle, c.brake);
        }
    }
    let s = race.stats;
    let min_laps = race.prog.iter().skip(1).map(|p| p.laps).min().unwrap_or(0);
    println!("RACE {secs:.0} s: contacts {} ({} with the player), deep overlaps {}, stuck {}, resets {}; slowest AI {} laps",
             s.contacts, s.contacts_player, s.deep, s.stuck, s.resets, min_laps);
    println!("RACE tyre smoke (max, s > 0.2): {}", race.ai.iter().zip(&smoke)
        .map(|(a, s)| format!("{} {:.2} {:.1}s", a.name, s.0, s.1)).collect::<Vec<_>>().join(", "));
    println!("RACE cpu: AI step {:.3} ms per 120 Hz step for {} cars ({:.1} s wall)", ai_ms / (secs as f64 * 120.0), race.ai.len(), wall.elapsed().as_secs_f32());
}

// ---- HUD: the start lights and the proximity radar (hud.rs look) ----

/// Radar range, m: cars inside it show, and the panel's half side (15 m
/// drew a car 12 px long on a 720 p screen).
const RADAR_M: f32 = 12.0;

impl Race {
    /// Every AI car's engine-sound feed, placed in the world (carsound.rs
    /// update_ai).
    pub fn sound_cars(&self) -> Vec<crate::carsound::AiCar<'_>> {
        self.ai.iter().map(|a| {
            let mut f = a.td.sound_feed(false);
            let p = a.td.pose();
            f.car_pos = (p.pos + p.up * 0.5).to_array();
            f.car_vel = a.td.vel().to_array();
            f.car_fwd = p.forward.to_array();
            f.car_up = p.up.to_array();
            crate::carsound::AiCar { name: a.name, model: &a.td.slot.model, feed: f }
        }).collect()
    }
}

impl Ai {
    /// The driver's brake pedal 0..1 (its brake lamps).
    pub fn brake(&self) -> f32 {
        self.ctl.brake
    }
}

impl Race {
    /// R: the radar automatic (a car within range) -> always on -> off.
    pub fn radar_key(&mut self) {
        self.radar = match self.radar {
            None => Some(true),
            Some(true) => Some(false),
            Some(false) => None,
        };
        println!("radar: {}", match self.radar { None => "automatic", Some(true) => "on", Some(false) => "off" });
    }

    pub fn draw_hud(&self, hud: &mut crate::hud::Hud, player: &TrackDrive, view: Option<usize>, aspect: f32) {
        use crate::hud::{theme as t, Weight};
        let u = |v: f32| v / aspect;
        // the start lights, top centre: five reds, then all out
        if let Some((lit, out)) = self.lights() {
            let (d, gap, pad) = (0.045, 0.012, 0.014);
            let w = u(d * 5.0 + gap * 4.0 + pad * 2.0);
            let (x, y) = (0.5 - w * 0.5, 0.06);
            hud.panel(x, y, w, d + pad * 2.0, 10.0, t::panel(0.55), t::line(0.5));
            for i in 0..5u32 {
                let c = if !out && i < lit { t::red(1.0) } else { t::well(0.9) };
                hud.round_rect(x + u(pad + i as f32 * (d + gap)), y + pad, u(d), d, 999.0, c);
            }
            if out {
                let s = "GO";
                let sz = 0.04;
                let tw = hud.label_w(s, sz, Weight::Semibold);
                hud.label(s, 0.5 - tw * 0.5, y + d + pad * 2.0 + 0.01, sz, Weight::Semibold, t::ok(1.0));
            }
        }
        if let Some(green) = self.rolling_banner() {
            let (s, c) = if green { ("GREEN FLAG", t::ok(1.0)) } else { ("ROLLING START - HOLD POSITION", t::gold(1.0)) };
            let sz = 0.03;
            let tw = hud.label_w(s, sz, Weight::Semibold);
            hud.panel(0.5 - tw * 0.5 - u(0.015), 0.06, tw + u(0.03), sz * 1.6, 8.0, t::panel(0.55), t::line(0.5));
            hud.label(s, 0.5 - tw * 0.5, 0.06 + sz * 0.25, sz, Weight::Semibold, c);
        }
        // the radar: bottom, left of the car in the chase view (the cluster and
        // the aids sit bottom right, the pit menu left), EVOVIEW_RADAR_POS=x,y
        let near: Vec<_> = self.near_of(player, view, RADAR_M * 1.5).into_iter()
            .filter(|(c, ..)| c.x.abs() < RADAR_M && c.y.abs() < RADAR_M).collect();
        let show = self.radar.unwrap_or(!near.is_empty());
        if !show {
            return;
        }
        let side = 0.2;
        let (w, h) = (u(side), side);
        let (x, y) = std::env::var("EVOVIEW_RADAR_POS").ok()
            .and_then(|v| v.split_once(',').and_then(|(a, b)| Some((a.trim().parse::<f32>().ok()?, b.trim().parse::<f32>().ok()?))))
            .map_or((0.5 - w * 0.5 - u(0.42), 1.0 - h - 0.025), |(a, b)| (a.clamp(0.0, 1.0 - w), b.clamp(0.0, 1.0 - h)));
        hud.panel(x, y, w, h, 10.0, t::panel(0.55), t::line(0.45));
        let (cx, cy) = (x + w * 0.5, y + h * 0.5);
        // metres -> screen: +x right, +y forward (up the screen)
        let k = (side * 0.5 - 0.012) / RADAR_M;
        let to = |p: Vec2| (cx + u(p.x * k), cy - p.y * k);
        // guides: the centre lines and the 5 / 10 m rings' ticks
        hud.rect(cx - hud.px_x(1.0), y + 0.01, hud.px_x(2.0), h - 0.02, t::dim2(0.25));
        hud.rect(x + u(0.01), cy - hud.px_y(1.0), w - u(0.02), hud.px_y(2.0), t::dim2(0.25));
        let boxq = |hud: &mut crate::hud::Hud, c: Vec2, head: f32, hw: f32, hl: f32, col: [f32; 4]| {
            let (s, co) = head.sin_cos();
            // heading + = turned right: forward (0, 1) -> (sin, cos)
            let f = Vec2::new(s, co);
            let r = Vec2::new(co, -s);
            let pts = [c + f * hl + r * hw, c + f * hl - r * hw, c - f * hl - r * hw, c - f * hl + r * hw].map(to);
            hud.tri([pts[0], pts[1], pts[2]], col);
            hud.tri([pts[0], pts[2], pts[3]], col);
        };
        let (phw, pzf, pzr) = player.footprint();
        let phl = (pzf - pzr) * 0.5;
        let mut left = None::<(f32, f32)>;
        let mut right = None::<(f32, f32)>;
        for &(c, head, hw, hl) in &near {
            if c.x.abs() > RADAR_M || c.y.abs() > RADAR_M {
                continue;
            }
            // alongside: overlapping the player's length, within a lane
            let beside = c.y.abs() < phl + hl && c.x.abs() < 4.5 && c.x.abs() > phw;
            if beside {
                let span = (c.y - hl, c.y + hl);
                let e = if c.x < 0.0 { &mut left } else { &mut right };
                *e = Some(e.map_or(span, |s: (f32, f32)| (s.0.min(span.0), s.1.max(span.1))));
            }
            // closer = stronger
            let a = (1.0 - (c.length() - 4.0) / RADAR_M).clamp(0.45, 1.0);
            boxq(hud, c, head, hw, hl, if beside { t::red(a) } else { t::gold(a) });
        }
        boxq(hud, Vec2::ZERO, 0.0, phw, phl, t::fg(0.95));
        // red brackets on the side a car is alongside, over its length
        for (side, span) in [(-1.0f32, left), (1.0, right)] {
            let Some((a, b)) = span else { continue };
            let (a, b) = (a.max(-RADAR_M), b.min(RADAR_M));
            let bx = if side < 0.0 { x + u(0.006) } else { x + w - u(0.006) - u(0.006) };
            let (_, y0) = to(Vec2::new(0.0, b));
            let (_, y1) = to(Vec2::new(0.0, a));
            hud.round_rect(bx, y0, u(0.006), (y1 - y0).max(0.01), 3.0, t::red(1.0));
        }
    }
}
