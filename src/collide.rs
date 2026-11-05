//! EVOVIEW_COLLIDE_LOG=1: the rigid-body car's contact events, for hunting
//! kerb launches and invisible walls (--drive-test). Each physics step:
//!
//! LAUNCH  the body's vertical speed over the road's own rise (median of
//!   five probes around it, ~30 ms smoothing) above EVOVIEW_LAUNCH_V m/s
//!   (1.0); one line per event (until it has stayed below 0.3 m/s for
//!   0.3 s) with how much higher the body got and the time on <= 1 wheel
//!   with its peak, air time, peak height and the surfaces under the wheels
//!   when it began.
//! RATE    body pitch or roll rate over 1.5 rad/s.
//! NJUMP   a wheel's contact normal turning > 20 deg in one 3 ms step.
//! HSTEP   a wheel's ground height jumping > 6 cm in one step.
//! PEN     tyre deflection over 6 cm (the radial spring bottoming).
//! SPIKE   a tyre's load passing 3x a quarter of the car's weight (once
//!   per excursion; printed with EVOVIEW_COLLIDE_LOG=2).
//! WALL    a wall contact closing faster than 0.5 m/s: where, the wall
//!   segment, its source mesh and the normal.
//!
//! Counts per lap come out through `lap_line` (trackdrive test_run).

use crate::rigcar::RigCar;
use crate::trackdrive::Ground;
use glam::Vec3;
use std::collections::HashMap;

#[derive(Default, Clone)]
pub struct Counts {
    pub launch: u32,
    pub launch_kerb: u32,
    pub rate: u32,
    pub njump: u32,
    pub hstep: u32,
    pub pen: u32,
    pub wall: u32,
    /// tyre load spikes over 3x a quarter of the car's weight
    pub spike: u32,
    pub wall_by: HashMap<String, u32>,
}

pub struct Events {
    pub n: Counts,
    pub total: Counts,
    launch_v: f32,
    /// an open launch: start t, peak v, peak height, surfaces at the start
    open: Option<(f32, f32, f32, String, Vec3, f32)>,
    prev_gn: [Vec3; 4],
    prev_gy: [f32; 4],
    prev_on: [bool; 4],
    spiking: [bool; 4],
    last_rate: f32,
    last_up: f32,
    road: f32,
    air: f32,
    h0: f32,
    last_njump: f32,
    last_hstep: f32,
    last_pen: f32,
    last_wall: HashMap<u32, f32>,
    /// verbose lines (EVOVIEW_COLLIDE_LOG=2 prints every NJUMP / HSTEP too)
    verbose: bool,
}

impl Events {
    pub fn from_env() -> Option<Box<Events>> {
        let v = crate::envc::var("EVOVIEW_COLLIDE_LOG").ok()?;
        Some(Box::new(Events {
            n: Counts::default(),
            total: Counts::default(),
            launch_v: crate::envc::var("EVOVIEW_LAUNCH_V").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            open: None,
            prev_gn: [Vec3::Y; 4],
            prev_gy: [0.0; 4],
            prev_on: [false; 4],
            spiking: [false; 4],
            last_rate: -9.0,
            last_up: -9.0,
            road: f32::NAN,
            air: 0.0,
            h0: 0.0,
            last_njump: -9.0,
            last_hstep: -9.0,
            last_pen: -9.0,
            last_wall: HashMap::new(),
            verbose: v == "2",
        }))
    }

    fn bump(&mut self, f: impl Fn(&mut Counts)) {
        f(&mut self.n);
        f(&mut self.total);
    }

    /// After each 3 ms step.
    pub fn step(&mut self, rc: &RigCar, ground: &Ground) {
        let t = rc.t();
        let (fw, up) = (rc.forward(), rc.up());
        let lat = up.cross(fw);
        // the road under the body: the median of five probes around its
        // middle (a kerb under one of them does not move it), smoothed over
        // ~30 ms; the launch speed is the body's vertical speed over the
        // road's own rise
        let o = rc.origin();
        let mut ys: Vec<f32> = [Vec3::ZERO, fw * 1.2, -fw * 1.2, lat * 0.8, -lat * 0.8].iter()
            .filter_map(|d| ground.probe(o.x + d.x, o.z + d.z, o.y + 1.0).map(|h| h.y)).collect();
        ys.sort_by(|a, b| a.total_cmp(b));
        let road = ys.get(ys.len() / 2).copied().unwrap_or(o.y);
        let dt = crate::rigcar::DT;
        let prev = self.road;
        self.road = if prev.is_nan() { road } else { prev + (road - prev) * (dt / 0.03).min(1.0) };
        let vroad = if prev.is_nan() { 0.0 } else { (self.road - prev) / dt };
        let vn = rc.v.y - vroad;
        let _ = up;
        let off = rc.c.iter().filter(|c| !c.on_ground).count();
        let kmh = rc.speed() * 3.6;
        let surfs = || (0..4).map(|k| {
            let c = &rc.c[k];
            if c.on_ground { format!("{}:{}", ground.surf_name(c.surf), ground.kind(c.surf)) } else { "air".into() }
        }).collect::<Vec<_>>().join(" ");
        let height = o.y - self.road;
        if vn > 0.3 {
            self.last_up = t;
        }
        match self.open.as_mut() {
            None if vn > self.launch_v => {
                self.open = Some((t, vn, height, surfs(), rc.origin(), kmh));
                self.air = 0.0;
                self.h0 = height;
            }
            Some(o) => {
                o.1 = o.1.max(vn);
                o.2 = o.2.max(height);
                if off >= 3 {
                    self.air += dt;
                }
                // over once it has stayed down for 0.3 s (one hop, one event)
                if vn < 0.3 && t - self.last_up > 0.3 {
                    let (t0, pv, ph, s, p, k0) = self.open.take().unwrap();
                    let kerb = s.contains(":2") || s.contains(":5");
                    self.bump(|c| { c.launch += 1; if kerb { c.launch_kerb += 1 } });
                    println!("LAUNCH t {t0:.2} at ({:.1}, {:.1}, {:.1}) {k0:.0} km/h: peak {pv:.2} m/s up off the road, body {:.2} m higher, 3+ wheels off for {:.2} s | thr {:.2} brk {:.2} st {:+.3} | {s}",
                             p.x, p.y, p.z, ph - self.h0, self.air, rc.thr, rc.brk, rc.steer);
                }
            }
            _ => {}
        }
        let (pr, rr) = (rc.w.dot(lat), rc.w.dot(fw));
        if (pr.abs() > 1.5 || rr.abs() > 1.5) && t - self.last_rate > 0.3 {
            self.last_rate = t;
            self.bump(|c| c.rate += 1);
            let p = rc.origin();
            println!("RATE t {t:.2} at ({:.1}, {:.1}, {:.1}) {kmh:.0} km/h: pitch {pr:+.2} roll {rr:+.2} rad/s | {}", p.x, p.y, p.z, surfs());
        }
        for k in 0..4 {
            let c = &rc.c[k];
            // the ground under the hub, from the contact
            let gy = c.hub.y - c.gap / c.gn.y.max(0.2);
            if c.on_ground && self.prev_on[k] {
                let ang = c.gn.dot(self.prev_gn[k]).clamp(-1.0, 1.0).acos().to_degrees();
                if ang > 20.0 {
                    let first = t - self.last_njump > 0.3;
                    if first {
                        self.bump(|c| c.njump += 1);
                    }
                    if first || self.verbose {
                        self.last_njump = t;
                        println!("NJUMP t {t:.3} wheel {k} at ({:.2}, {:.2}, {:.2}) {kmh:.0} km/h: normal turned {ang:.0} deg {:?} -> {:?} on {} | fz {:.0}",
                                 c.hub.x, c.hub.y, c.hub.z, self.prev_gn[k], c.gn, ground.surf_name(c.surf), c.fz);
                    }
                }
                let dy = gy - self.prev_gy[k];
                if dy.abs() > 0.06 {
                    let first = t - self.last_hstep > 0.3;
                    if first {
                        self.bump(|c| c.hstep += 1);
                    }
                    if first || self.verbose {
                        self.last_hstep = t;
                        println!("HSTEP t {t:.3} wheel {k} at ({:.2}, {:.2}, {:.2}) {kmh:.0} km/h: ground {dy:+.3} m in one step on {} | n {:?} fz {:.0}",
                                 c.hub.x, c.hub.y, c.hub.z, ground.surf_name(c.surf), c.gn, c.fz);
                    }
                }
            }
            let quarter = rc.body_mass() * 9.81 * 0.25;
            if c.on_ground && c.fz > 3.0 * quarter && !self.spiking[k] {
                self.bump(|c| c.spike += 1);
                if self.verbose {
                    println!("SPIKE t {t:.3} wheel {k} at ({:.2}, {:.2}, {:.2}) {kmh:.0} km/h: fz {:.0} N on {} n {:?}", c.hub.x, c.hub.y, c.hub.z, c.fz, ground.surf_name(c.surf), c.gn);
                }
            }
            self.spiking[k] = c.on_ground && c.fz > 2.0 * quarter;
            let defl = rc.radius(k) - c.gap;
            if c.on_ground && defl > 0.06 && t - self.last_pen > 0.3 {
                self.last_pen = t;
                self.bump(|c| c.pen += 1);
                println!("PEN t {t:.3} wheel {k} at ({:.2}, {:.2}, {:.2}) {kmh:.0} km/h: tyre {defl:.3} m deflected, fz {:.0} on {}",
                         c.hub.x, c.hub.y, c.hub.z, c.fz, ground.surf_name(c.surf));
            }
            self.prev_gn[k] = c.gn;
            self.prev_gy[k] = gy;
            self.prev_on[k] = c.on_ground;
        }
    }

    /// A wall contact from rigcar walls(): segment index, contact point,
    /// normal, closing speed.
    pub fn wall(&mut self, t: f32, ground: &Ground, wi: u32, at: Vec3, n: Vec3, vn: f32, kmh: f32) {
        if vn > -0.5 {
            return;
        }
        if self.last_wall.get(&wi).map_or(false, |&l| t - l < 1.0) {
            self.last_wall.insert(wi, t);
            return;
        }
        self.last_wall.insert(wi, t);
        let (src, w) = ground.wall(wi);
        let src = src.to_string();
        self.bump(|c| {
            c.wall += 1;
            *c.wall_by.entry(src.clone()).or_default() += 1;
        });
        println!("WALL t {t:.2} at ({:.2}, {:.2}, {:.2}) {kmh:.0} km/h, {:.1} m/s in: #{wi} ({:.2},{:.2})-({:.2},{:.2}) y {:.2}..{:.2} n ({:+.2}, {:+.2}) {src}",
                 at.x, at.y, at.z, -vn, w[0], w[1], w[2], w[3], w[4], w[5], n.x, n.z);
    }

    /// The counts since the last call, as one line.
    pub fn lap_line(&mut self) -> String {
        let n = std::mem::take(&mut self.n);
        Self::fmt(&n)
    }

    pub fn total_line(&self) -> String {
        Self::fmt(&self.total)
    }

    fn fmt(n: &Counts) -> String {
        let mut by: Vec<_> = n.wall_by.iter().collect();
        by.sort_by(|a, b| b.1.cmp(a.1));
        format!("launch {} (kerb {}) rate {} njump {} hstep {} pen {} spike {} wall {}{}", n.launch, n.launch_kerb, n.rate, n.njump, n.hstep, n.pen, n.spike, n.wall,
                if by.is_empty() { String::new() } else { format!(" [{}]", by.iter().take(6).map(|(s, c)| format!("{c}x {s}")).collect::<Vec<_>>().join("; ")) })
    }
}
