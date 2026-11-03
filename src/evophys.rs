//! AC EVO's suspension, diff and brake models, as the client exe runs them
//! (L:/evoview_work/q/SPEC_PHYS.md). Each part reads its own protobuf
//! message, the same fields the game's init copies:
//!
//! - Coilover (CoiloverData): spring + progressive rate + helper spring +
//!   collar, two bump stops, a two-way two-speed damper. Coilover::force is
//!   fn 0x1421f2d20, the damper 0x1421f1a00, a bump stop 0x1421f3280/3130.
//! - Anti-roll bar (ARBData): F = k (z_right - z_left) between the hubs,
//!   fn 0x142201450.
//! - Differential (DifferentialData): fn 0x1420e9540, by type.
//! - Brakes (BrakesData): fn 0x1420d5f00 per wheel.
//!
//! Sign convention: travel `x` and velocity `v` are positive in compression
//! (bump); forces are positive pushing the wheel away from the body.

use crate::proto::fields;

fn f32_at(d: &[u8], a: usize, b: usize, want: u32) -> Option<f32> {
    let mut hit = None;
    fields(d, a, b, |f, v| {
        if f == want {
            hit = Some(v.f32());
        }
    });
    hit
}

fn sub(d: &[u8], a: usize, b: usize, want: u32) -> Option<(usize, usize)> {
    let mut hit = None;
    fields(d, a, b, |f, v| {
        if f == want && hit.is_none() {
            hit = v.range();
        }
    });
    hit
}

/// BumpStopData {1 Range, 2 Reference, 3 Force, 4 Gamma, 5 length, 6 damping}
#[derive(Clone, Copy, Debug, Default)]
pub struct BumpStop {
    /// free travel before the stop touches, m
    pub range: f32,
    /// the travel into the stop at which it pushes `force`
    pub reference: f32,
    pub force: f32,
    pub gamma: f32,
    pub length: f32,
    /// N per m/s while engaged
    pub damping: f32,
}

impl BumpStop {
    fn read(d: &[u8], a: usize, b: usize) -> BumpStop {
        let g = |f| f32_at(d, a, b, f).unwrap_or(0.0);
        BumpStop { range: g(1), reference: g(2), force: g(3), gamma: g(4), length: g(5), damping: g(6) }
    }

    /// force at `depth` metres into the stop (fn 0x1421f3280)
    fn push(&self, depth: f32) -> f32 {
        if self.force > 0.0 && self.reference > 0.0 {
            self.force * (depth / self.reference).powf(if self.gamma > 0.0 { self.gamma } else { 1.0 })
        } else {
            0.0
        }
    }
}

/// DampersData: slow / fast bump and rebound, N per m/s, and the speeds
/// where the fast slope takes over.
#[derive(Clone, Copy, Debug, Default)]
pub struct Damper {
    pub slow_bump: f32,
    pub slow_rebound: f32,
    pub fast_bump: f32,
    pub fast_rebound: f32,
    pub fast_threshold_bump: f32,
    pub fast_threshold_rebound: f32,
    /// A LUT damper (DampersData f13 lut_list, set by the setup's clicks,
    /// carphys::damper_lut): shaft speed (m/s, + bump) -> force, in place
    /// of the two slopes. Leaked once per car load so the coilover stays Copy.
    pub lut: Option<&'static [(f32, f32)]>,
    /// DampersData f14 damper_lut_scale. GUESS: a force multiplier (where the
    /// game applies it is not decoded; 1.3 on the 296 GT3, 1.0 the Mustang)
    pub lut_scale: f32,
}

impl Damper {
    fn read(d: &[u8], a: usize, b: usize) -> Damper {
        let two = |f: u32| sub(d, a, b, f).map(|(c, e)| (f32_at(d, c, e, 1).unwrap_or(0.0), f32_at(d, c, e, 2).unwrap_or(0.0))).unwrap_or_default();
        let (fast, slow) = (two(1), two(2));
        Damper {
            slow_bump: slow.0,
            slow_rebound: slow.1,
            fast_bump: fast.0,
            fast_rebound: fast.1,
            fast_threshold_bump: f32_at(d, a, b, 3).unwrap_or(0.0),
            fast_threshold_rebound: f32_at(d, a, b, 4).unwrap_or(0.0),
            lut: None,
            lut_scale: f32_at(d, a, b, 14).filter(|s| *s > 0.0).unwrap_or(1.0),
        }
    }

    /// Damping force at shaft speed `v` (positive = bump), opposing the
    /// motion (fn 0x1421f1a00, without the LUT / active-damper paths).
    pub fn force(&self, v: f32) -> f32 {
        if let Some(l) = self.lut.filter(|l| l.len() >= 2) {
            let x = v.clamp(l[0].0, l[l.len() - 1].0);
            let i = l.partition_point(|p| p.0 < x).clamp(1, l.len() - 1);
            let (a, b) = (l[i - 1], l[i]);
            let t = (x - a.0) / (b.0 - a.0).max(1e-6);
            return (a.1 + (b.1 - a.1) * t) * self.lut_scale;
        }
        if v > 0.0 {
            let t = self.fast_threshold_bump;
            if t < v {
                -self.slow_bump * t - (v - t) * self.fast_bump
            } else {
                -self.slow_bump * v
            }
        } else {
            let t = self.fast_threshold_rebound;
            if t < -v {
                self.slow_rebound * t - (t + v) * self.fast_rebound
            } else {
                -self.slow_rebound * v
            }
        }
    }
}

/// CoiloverData {1 springRate, 2 progressiveSpringRate, 3 bumpStopUp,
/// 4 bumpStopDown, 5 collar_position, 6 damper, 7 helperK, 8 helperRange}
#[derive(Clone, Copy, Debug, Default)]
pub struct Coilover {
    pub k: f32,
    pub k_prog: f32,
    pub collar: f32,
    pub helper_k: f32,
    pub helper_range: f32,
    pub stop_up: BumpStop,
    pub stop_down: BumpStop,
    pub damper: Damper,
}

impl Coilover {
    pub fn read(d: &[u8], a: usize, b: usize) -> Coilover {
        let g = |f| f32_at(d, a, b, f).unwrap_or(0.0);
        let stop = |f| sub(d, a, b, f).map(|(c, e)| BumpStop::read(d, c, e)).unwrap_or_default();
        Coilover {
            k: g(1),
            k_prog: g(2),
            collar: g(5),
            helper_k: g(7),
            helper_range: g(8),
            stop_up: stop(3),
            stop_down: stop(4),
            damper: sub(d, a, b, 6).map(|(c, e)| Damper::read(d, c, e)).unwrap_or_default(),
        }
    }

    /// (spring, damping) force at travel `x` and speed `v` (fn 0x1421f2d20).
    /// The spring part never pulls; the bump stops add to both.
    pub fn force(&self, x: f32, v: f32) -> (f32, f32) {
        let c = x + self.collar;
        let mut spring = if c <= 0.0 {
            (self.helper_range + c) * self.helper_k
        } else {
            (c * self.k_prog + self.k) * c + self.helper_k * self.helper_range
        }
        .max(0.0);
        let mut damp = -self.damper.force(v);
        // bump stop: up (compression) past +range; down (droop) past -range
        let up = self.stop_up.range;
        if up < x {
            let f = self.stop_up.push(x - up);
            let dv = v * self.stop_up.damping;
            if 0.0 < dv + f {
                spring += f;
                damp += dv;
            }
        }
        let down = -self.stop_down.range;
        if x < down {
            let f = -self.stop_down.push(down - x);
            let dv = v * self.stop_down.damping;
            if dv + f < 0.0 {
                spring += f;
                damp += dv;
            }
        }
        (spring, damp)
    }
}

/// ARBData.stiffness: the bar's force on each hub, N (left, right).
pub fn arb_forces(k: f32, z_left: f32, z_right: f32) -> (f32, f32) {
    let dz = z_right - z_left;
    (k * dz, -k * dz)
}

/// DifferentialData.type (enum DifferentialType, as the step switches)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum DiffKind {
    /// clutch-pack LSD: preload + power / coast ramp on the input torque
    #[default]
    Lsd,
    Spool,
    /// torque-bias (Torsen): the lighter-loaded side x TBR
    Tbr,
    Tbr2,
    Elsd,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Diff {
    pub kind: DiffKind,
    pub power: f32,
    pub coast: f32,
    pub preload: f32,
    pub tbr_power: f32,
    pub tbr_coast: f32,
}

impl Diff {
    /// DifferentialData {1 power, 2 coast, 3 preload, 4 type, 11/12 tbr}
    pub fn read(d: &[u8], a: usize, b: usize) -> Diff {
        let g = |f| f32_at(d, a, b, f).unwrap_or(0.0);
        let mut ty = 0u64;
        fields(d, a, b, |f, v| {
            if f == 4 {
                if let crate::proto::Val::Varint(x) = v {
                    ty = x;
                }
            }
        });
        let (power, coast) = (g(1), g(2));
        // the init turns any diff with both ramps >= 1 into a spool
        let kind = if power >= 1.0 && coast >= 1.0 {
            DiffKind::Spool
        } else {
            match ty {
                1 => DiffKind::Spool,
                2 | 3 => DiffKind::Tbr,
                4 => DiffKind::Tbr2,
                5 => DiffKind::Elsd,
                _ => DiffKind::Lsd,
            }
        };
        Diff { kind, power, coast, preload: g(3), tbr_power: g(11), tbr_coast: g(12) }
    }

    /// Locking torque the diff can hold between the two sides, N m.
    /// `t_in` = input (driveshaft) torque, + driving; `t_left` / `t_right`
    /// = the tyre-side torques (for a torque-bias diff).
    pub fn lock(&self, t_in: f32, t_left: f32, t_right: f32) -> f32 {
        match self.kind {
            DiffKind::Spool => f32::INFINITY,
            DiffKind::Tbr | DiffKind::Tbr2 => {
                t_left.abs().min(t_right.abs()) * if t_in > 0.0 { self.tbr_power } else { self.tbr_coast }
            }
            DiffKind::Lsd | DiffKind::Elsd => {
                (if t_in > 0.0 { self.power * t_in } else { (self.coast * t_in).abs() }) + self.preload
            }
        }
    }

    /// The torque it applies against the side-to-side speed difference
    /// (`dw` = left - right, rad/s): smoothed sign, as the game does.
    pub fn torque(&self, t_in: f32, dw: f32, t_left: f32, t_right: f32) -> f32 {
        let l = self.lock(t_in, t_left, t_right);
        if l.is_infinite() {
            return 0.0;
        }
        -(dw / (dw.abs() + 0.01)) * l
    }
}

/// BrakesData {1 totalTorque, 2 frontBias (%), 3 handBrakeTorque}
#[derive(Clone, Copy, Debug)]
pub struct Brakes {
    pub total: f32,
    pub front_bias: f32,
    pub handbrake: f32,
}

impl Default for Brakes {
    // the game's own fallbacks (fn 0x1420d4c70): 2000 N m, 70 % front
    fn default() -> Self {
        Brakes { total: 2000.0, front_bias: 0.7, handbrake: 0.0 }
    }
}

impl Brakes {
    pub fn read(d: &[u8], a: usize, b: usize) -> Brakes {
        let mut out = Brakes::default();
        if let Some(t) = f32_at(d, a, b, 1).filter(|v| *v > 0.0) {
            out.total = t;
        }
        // front bias in %; the Giulia GTAm, Delta Integrale and 190E store a
        // fraction (0.75) - their stock setups carry the % the game uses
        if let Some(p) = f32_at(d, a, b, 2).filter(|v| *v > 0.0) {
            out.front_bias = if p <= 1.0 { p } else { p * 0.01 };
        }
        out.handbrake = f32_at(d, a, b, 3).unwrap_or(0.0);
        out
    }

    /// Brake torque on each wheel FL FR RL RR, N m. Each wheel gets the
    /// whole total x its axle's share (not half of it), as fn 0x1420d5f00.
    pub fn torques(&self, pedal: f32, handbrake: f32) -> [f32; 4] {
        let p = pedal.clamp(0.0, 1.0);
        let f = p * self.total * self.front_bias;
        let r = (p * self.total * (1.0 - self.front_bias)).max(0.0) + handbrake.clamp(0.0, 1.0) * self.handbrake;
        [f, f, r, r]
    }
}

/// ABS as the game runs it (ABS_V2: ctor 0x142194f00, step 0x142195300,
/// wheel test 0x142194e60). CarData f7 ElectronicsData f2 ABSData {1
/// settings[] {1 minSlipRatio, 2 maxSlipRatio, 3 refSlipAngle deg, 4
/// cutLevel, 5 maxTorqueVariation}, 2 frequency Hz, 3 channels 1/2/4, 4
/// minSpeedKmh}. Level 0 is "off" (min slip <= 0); the setup's
/// ElectronicsSettings f3 picks the level (default 1).
#[derive(Clone, Debug, Default)]
pub struct Abs {
    pub levels: Vec<[f32; 5]>,
    pub frequency: f32,
    pub channels: i32,
    pub min_speed: f32,
    pub level: usize,
    timer: f32,
    /// per wheel brake multiplier, FL FR RL RR
    pub mult: [f32; 4],
    pub active: bool,
}

impl Abs {
    pub fn read(d: &[u8], a: usize, b: usize) -> Abs {
        let mut out = Abs { frequency: 50.0, channels: 4, mult: [1.0; 4], level: 1, ..Default::default() };
        fields(d, a, b, |f, v| match f {
            1 => {
                if let Some((x, y)) = v.range() {
                    let g = |n| f32_at(d, x, y, n).unwrap_or(0.0);
                    out.levels.push([g(1), g(2), g(3).to_radians(), g(4), g(5)]);
                }
            }
            2 => out.frequency = v.f32(),
            3 => out.channels = v.f32() as i32,
            4 => out.min_speed = v.f32() / 3.6,
            _ => {}
        });
        out
    }

    /// One update. `sr` / `alpha` per wheel FL FR RL RR (braking slip < 0),
    /// `speed` m/s, `pedal` 0..1. Holds its multipliers between updates.
    pub fn step(&mut self, dt: f32, speed: f32, pedal: f32, sr: [f32; 4], alpha: [f32; 4]) {
        self.timer += dt;
        if self.frequency > 0.0 && self.timer < 1.0 / self.frequency {
            return;
        }
        self.timer = 0.0;
        let lv = self.levels.get(self.level.min(self.levels.len().saturating_sub(1))).copied();
        let on = matches!(lv, Some(l) if l[0] > 0.0) && speed > self.min_speed && pedal >= 0.01;
        self.mult = [1.0; 4];
        self.active = false;
        let Some(l) = lv.filter(|_| on) else { return };
        // the slip it allows: max straight on, min at the reference angle
        let trig = |k: usize| {
            let t = if l[2] > 0.0 { (alpha[k].abs() / l[2]).clamp(0.0, 1.0) } else { 0.0 };
            sr[k] < -(t * (l[0] - l[1]) + l[1])
        };
        let cut = l[3];
        match self.channels {
            1 => {
                let f = if trig(0) || trig(1) { cut } else { 1.0 };
                let r = if trig(2) || trig(3) { cut } else { 1.0 };
                self.mult = [f.min(r); 4];
            }
            2 => {
                let f = if trig(0) || trig(1) { cut } else { 1.0 };
                let r = if trig(2) || trig(3) { cut } else { 1.0 };
                self.mult = [f, f, r, r];
            }
            _ => {
                for k in 0..4 {
                    if trig(k) {
                        self.mult[k] = cut;
                    }
                }
                // a wheel may not drop more than maxTorqueVariation below
                // its axle partner
                for k in 0..4 {
                    let o = k ^ 1;
                    if self.mult[k] < 1.0 && (self.mult[k] - self.mult[o]) / self.mult[o] < -l[4] {
                        self.mult[k] = (1.0 - l[4]) * self.mult[o];
                    }
                }
            }
        }
        self.active = self.mult.iter().any(|m| *m < 0.95);
    }
}

/// Traction control as the game runs it (ctor 0x142197c30, step
/// 0x1421981d0). CarData f7 ElectronicsData f1 TCData {2 frequencyHz, 3
/// minSpeedKmh, 7 settings[] {1 minSlipRatio, 2 maxSlipRatio, 3
/// refSlipAngleDeg, 4 engineCutLevel, 5 angularACCgain, 6 oversteerGain,
/// 7 slipAngleActivationDeg}}. The setup's tc1 picks the level (default 1).
/// Out: a throttle multiplier, 1 - (x^2 (c + 1) - c), x = slip / allowed
/// slip (max straight on, min at the reference angle), c the cut level.
#[derive(Clone, Debug, Default)]
pub struct Tc {
    pub levels: Vec<[f32; 7]>,
    pub frequency: f32,
    pub min_speed: f32,
    pub level: usize,
    timer: f32,
    pub mult: f32,
}

impl Tc {
    pub fn read(d: &[u8], a: usize, b: usize) -> Tc {
        let mut out = Tc { frequency: 50.0, mult: 1.0, level: 1, ..Default::default() };
        fields(d, a, b, |f, v| match f {
            7 => {
                if let Some((x, y)) = v.range() {
                    let g = |n| f32_at(d, x, y, n).unwrap_or(0.0);
                    out.levels.push([g(1), g(2), g(3).to_radians(), g(4), g(5), g(6), g(7).to_radians()]);
                }
            }
            2 => out.frequency = v.f32(),
            3 => out.min_speed = v.f32() / 3.6,
            _ => {}
        });
        out
    }

    /// `sr` the driven wheels' worst slip ratio, `alpha` their slip angle
    /// (rad), `throttle` 0..1. The game also feeds an oversteer measure and
    /// the yaw acceleration into the cut once the slip angle passes the
    /// activation angle; those two are left at 0 here.
    pub fn step(&mut self, dt: f32, speed: f32, sr: f32, alpha: f32, throttle: f32) -> f32 {
        let Some(l) = self.levels.get(self.level.min(self.levels.len().saturating_sub(1))).copied() else { return 1.0 };
        // a slip ratio means nothing at a standstill; with no minSpeedKmh in
        // its data (the 296 GT3) the cut sat at 0 % and a stopped car could
        // never pull away again
        if l[0] <= 0.0 || speed < self.min_speed.max(1.5) {
            self.mult = 1.0;
            return 1.0;
        }
        self.timer += dt;
        if self.frequency > 0.0 && self.timer < 1.0 / self.frequency {
            return self.mult;
        }
        self.timer = 0.0;
        let t = if l[2] > 0.0 { (alpha.abs() / l[2]).clamp(0.0, 1.0) } else { 0.0 };
        let c = l[3];
        let mut m = 1.0;
        if throttle > 0.1 {
            let x = sr / (t * (l[0] - l[1]) + l[1]);
            m = (1.0 - (x * x * (c + 1.0) - c)).clamp(0.0, 1.0);
        }
        self.mult = m;
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coil() -> Coilover {
        Coilover {
            k: 100_000.0,
            k_prog: 0.0,
            collar: 0.0,
            helper_k: 0.0,
            helper_range: 0.0,
            stop_up: BumpStop { range: 0.05, reference: 0.01, force: 5000.0, gamma: 2.0, length: 0.02, damping: 0.0 },
            stop_down: BumpStop::default(),
            damper: Damper { slow_bump: 4000.0, slow_rebound: 8000.0, fast_bump: 2000.0, fast_rebound: 4000.0, fast_threshold_bump: 0.1, fast_threshold_rebound: 0.1, lut: None, lut_scale: 1.0 },
        }
    }

    #[test]
    fn spring_linear_then_bump_stop() {
        let c = coil();
        assert!((c.force(0.02, 0.0).0 - 2000.0).abs() < 1e-3);
        // 1 cm into the stop = its reference: + 5000 N
        assert!((c.force(0.06, 0.0).0 - (6000.0 + 5000.0)).abs() < 1e-2);
        assert_eq!(c.force(-0.01, 0.0).0, 0.0);
    }

    #[test]
    fn damper_two_speed_and_opposes() {
        let d = coil().damper;
        assert!((d.force(0.05) + 200.0).abs() < 1e-3);
        // past the knee: slow x knee + fast x the rest
        assert!((d.force(0.2) + (400.0 + 200.0)).abs() < 1e-3);
        assert!((d.force(-0.05) - 400.0).abs() < 1e-3);
        assert!((d.force(-0.2) - (800.0 + 400.0)).abs() < 1e-3);
    }

    #[test]
    fn damper_lut_interpolates_and_clamps() {
        static L: [(f32, f32); 3] = [(-6.0, 6000.0), (0.0, 0.0), (6.0, -12000.0)];
        let d = Damper { lut: Some(&L), lut_scale: 1.0, ..Default::default() };
        assert!((d.force(3.0) + 6000.0).abs() < 1e-2);
        assert!((d.force(-3.0) - 3000.0).abs() < 1e-2);
        assert!((d.force(9.0) + 12000.0).abs() < 1e-2);
    }

    #[test]
    fn lsd_ramps_and_brakes_split() {
        let df = Diff { kind: DiffKind::Lsd, power: 0.3, coast: 0.2, preload: 50.0, ..Default::default() };
        assert!((df.lock(1000.0, 0.0, 0.0) - 350.0).abs() < 1e-3);
        assert!((df.lock(-1000.0, 0.0, 0.0) - 250.0).abs() < 1e-3);
        let b = Brakes { total: 3000.0, front_bias: 0.6, handbrake: 1000.0 };
        let t = b.torques(1.0, 1.0);
        assert!((t[0] - 1800.0).abs() < 1e-3 && (t[3] - 2200.0).abs() < 1e-3);
    }
}
