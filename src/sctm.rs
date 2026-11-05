//! AC EVO's tyre model, SCTM, as the game runs it.
//!
//! Ported from the client exe (Ghidra; spec L:/evoview_work/q/SPEC_SCTM.md):
//! `SCTM::solve` 0x142297c60, its brush curve 0x142297780, the slip-speed
//! falloff 0x142297930, the compound set-up in "Using Stereo's combined
//! factor" 0x142180930 and the defaults of the SCTM constructor 0x1422974f0.
//! Field names are TyresData.proto's (TyresModelData f4, TyresCamberData
//! f7, TyresSpeedSensitivityData f8 of TyresCompoundData).
//!
//! What the solver needs from outside (`Input`) is the car's job: load,
//! slip angle / ratio, camber, speed, and the grip multiplier the runtime
//! builds from temperature, pressure and wear (~1 for a fresh tyre at
//! temperature). Inputs whose meaning is not pinned down (the pressure / p[10] terms, the +0x98 speed term) are documented on
//! `Input` and default to neutral.

use crate::proto::fields;

#[derive(Clone, Copy, Debug)]
pub struct Falloff {
    pub mu0t: f32,
    pub sens: f32,
    pub t_sens: f32,
    pub t_ref: f32,
    pub ref_speed: f32,
}

impl Falloff {
    /// Peak friction `d` with the patch sliding at `v_slip` m/s, tread at
    /// `temp` deg C: mu0T of it is kept, the rest fades with slip speed.
    pub fn apply(&self, v_slip: f32, d: f32, temp: f32) -> f32 {
        if v_slip <= 0.0 {
            return d;
        }
        let x = (v_slip / (((temp - self.t_ref) * self.t_sens + self.ref_speed).max(0.0) + 1e-4)).abs();
        let g = ((-x).exp() + x).ln();
        let keep = self.mu0t * d;
        (-(g * self.sens) * (g * self.sens)).exp() * (d - keep) + keep
    }
}

/// One compound's SCTM parameters (the solver's object).
#[derive(Clone, Copy, Debug)]
pub struct Sctm {
    pub ky: f32,
    pub ls_y: f32,
    pub kx: f32,
    pub ls_x: f32,
    pub fz0: f32,
    /// tan of the peak slip angle at Fz0 and at 2 Fz0
    pub tan_peak: f32,
    pub tan_peak_loaded: f32,
    /// peak slip ratio at Fz0 / 2 Fz0 (0: x stiffness = cf_x_mult x y)
    pub sr_peak: f32,
    pub sr_peak_loaded: f32,
    pub camber_gain: f32,
    pub dcamber0: f32,
    pub dcamber1: f32,
    pub cf_x_mult: f32,
    /// 1 + brakeDXMod
    pub brake_dx: f32,
    /// combinedFactor: p-norm exponent of the combined slip (0 / 2 euclid)
    pub combined_exp: f32,
    pub grip_slip_factor: f32,
    pub falloff_y: Falloff,
    pub falloff_x: Falloff,
    pub mz_trail_remap: f32,
    pub mz_nd_slip_reduction: f32,
    pub mz_scale: f32,
    pub brush_exp: f32,
    pub new_combined: bool,
    pub fx_shift_sa: f32,
    pub fx_scale_sa: f32,
    pub fy_shift_sr: f32,
    pub fy_scale_sr: f32,
    /// past the peak slip the force falls toward `min` of the peak:
    /// D ((1 - min) / (((s - 1/k) rate)^exp + 1) + min) (fn 0x1425081b0;
    /// f80..82 lateral, f83..85 longitudinal - slicks 0.75/5/1.5 and
    /// 0.8/2.5/1.65). [min, rate, exp]; rate or exp 0 = flat.
    pub post_y: [f32; 3],
    pub post_x: [f32; 3],
    /// TyresModelData f18 relaxationLengthY / f19 relaxationLengthX, m
    /// (rigcar: the slip lags the wheel's motion over this rolled length)
    pub relax_y: f32,
    pub relax_x: f32,
    /// TyresModelData f16 grainFactor: cornering stiffness / (1 + f grain%)
    pub grain_factor: f32,
    /// SCTM +0x3c (ctor 0.1): both slip stiffnesses x (1 + it x p[10]),
    /// p[10] = pressure / ideal - 1 (the tyre step's local_208). Read from
    /// TyresPressureData f1 pressure_limit_angle_gain - by its name (the
    /// limit angle is what the stiffness inverts); the copy itself was not
    /// found in the exe. Kunos compounds: 0.25.
    pub pressure_gain: f32,
    /// TyresData radius (m) and vertical rate (N/m, + progressive N/m^2),
    /// for the contact patch length
    pub radius: f32,
    pub rate: f32,
    pub rate_prog: f32,
    /// TyresData angularInertia (kg m^2), damping (N s/m), mass (kg)
    pub inertia: f32,
    pub damping: f32,
    pub mass: f32,
    /// TyresGrooveData (f10): grooveFactor, how much of the water the tread
    /// clears (0 slick, 0.3 intermediate, ~0.45 road, 0.68 wet), and how
    /// slip angle / slip ratio eat into it (grooveSAFactor / grooveSRFactor)
    pub groove: f32,
    pub groove_sa: f32,
    pub groove_sr: f32,
}

/// The water film's grip loss per (1 - groove) x wetness x m/s of patch
/// speed (the solver's constant DAT_14531f65c / DAT_145c9964c, 0.025)
pub const WET_K: f32 = 0.025;

#[derive(Clone, Copy, Debug)]
pub struct Input {
    /// vertical load, N
    pub fz: f32,
    /// slip angle, rad
    pub alpha: f32,
    /// slip ratio (negative braking)
    pub sr: f32,
    /// camber, rad
    pub camber: f32,
    /// speed over the ground, m/s (slip speed, and the aligning moment
    /// fades in below 1 m/s)
    pub speed: f32,
    /// grip multiplier (the runtime's thermal / pressure / wear chain)
    pub grip: f32,
    /// tread temperature, deg C (the falloff's: a blend of tread surface and
    /// core). NaN = each falloff's own TrefMuT. Slick data puts TrefMuT at
    /// 120 with muTSensitivity 0.4, so below ~114 C the slip-speed scale
    /// clamps to 0 and a sliding slick drops to mu0T (0.25). In the game a
    /// sliding tread flashes hot fast; without a thermal model, TrefMuT is
    /// the honest constant.
    pub temp: f32,
    /// p[7], the contact patch length in metres: the game passes the chord
    /// 2 sqrt(R^2 - (R - deflection)^2) (fn 0x1425c1960). NaN = from this
    /// tyre's radius and vertical rate at `fz`.
    pub trail_base: f32,
    /// wear percent 0..100 (grip / (1 + 0.2 x clamp(wear / 100)))
    pub wear: f32,
    /// p[8], graining %, which softens the cornering stiffness
    pub grain: f32,
    /// p[10], pressure / ideal - 1 (over-inflated > 0: stiffer, the peak
    /// slip angle comes in)
    pub pressure: f32,
    /// p[12], the water under the patch, 0..1: the dynamic track's wetness
    /// at the contact point (tyre step: the track map lookup's 2nd value)
    pub wet: f32,
}

impl Default for Input {
    fn default() -> Self {
        Input { fz: 0.0, alpha: 0.0, sr: 0.0, camber: 0.0, speed: 0.0, grip: 1.0, temp: f32::NAN, trail_base: f32::NAN, wear: 0.0, wet: 0.0, grain: 0.0, pressure: 0.0 }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Output {
    /// lateral force, N (sign of tan(alpha'), the model's slip convention)
    pub fy: f32,
    /// longitudinal force, N
    pub fx: f32,
    /// aligning moment, N m (= -fy x trail)
    pub mz: f32,
    /// pneumatic trail, m
    pub trail: f32,
    /// normalised combined slip (1 = at the peak)
    pub nd_slip: f32,
    /// peak friction used, lateral / longitudinal
    pub dy: f32,
    pub dx: f32,
}

fn f32_at(d: &[u8], a: usize, b: usize, want: u32) -> Option<f32> {
    let mut hit = None;
    fields(d, a, b, |f, v| {
        if f == want {
            hit = Some(v.f32());
        }
    });
    hit
}

fn sub(d: &[u8], want: u32) -> Option<(usize, usize)> {
    let mut hit = None;
    fields(d, 0, d.len(), |f, v| {
        if f == want && hit.is_none() {
            hit = v.range();
        }
    });
    hit
}

impl Sctm {
    /// From a `.tyre` (TyresCompoundData) file, with the game's defaults.
    pub fn from_compound(td: &[u8]) -> Option<Sctm> {
        let (a, b) = sub(td, 4)?;
        let g = |f: u32| f32_at(td, a, b, f).unwrap_or(0.0);
        let (dy0, mut dx0, ls_y, ls_x, fz0) = (g(1), g(2), g(3), g(4), g(5));
        if dy0 <= 0.0 || fz0 <= 0.0 {
            return None;
        }
        if dx0 == 0.0 {
            dx0 = dy0 * 1.2;
        }
        let fla = if g(6) > 0.0 { g(6) } else { 7.5 };
        let fla_gain = g(7);
        let use_sr = g(71) != 0.0;
        let (sr, sr_gain) = (g(60), g(70));
        let (cam, spd) = (sub(td, 7), sub(td, 8));
        let cg = |f: u32| cam.and_then(|(c, e)| f32_at(td, c, e, f)).unwrap_or(0.0);
        let (mut dc0, mut dc1) = (cg(4), cg(5));
        if dc0 == 0.0 || dc1 == 0.0 {
            (dc0, dc1) = (0.1, -0.8);
        }
        let sg = |f: u32, d: f32| spd.and_then(|(c, e)| f32_at(td, c, e, f)).filter(|v| *v != 0.0).unwrap_or(d);
        let fy = Falloff { mu0t: sg(11, 0.9), sens: sg(9, 0.01), t_sens: sg(12, 0.4), t_ref: sg(13, 70.0), ref_speed: sg(8, 2.0) };
        // the _x set only when given (mu0T_x > 0), else the same as y
        let fx = if sg(112, 0.0) > 0.0 {
            Falloff { mu0t: sg(112, 0.9), sens: sg(114, 0.01), t_sens: sg(115, 0.4), t_ref: sg(116, 70.0), ref_speed: sg(113, 2.0) }
        } else {
            fy
        };
        let nz = |v: f32, d: f32| if v != 0.0 { v } else { d };
        let td_f = |m: u32, f: u32| sub(td, m).and_then(|(c, e)| f32_at(td, c, e, f));
        Some(Sctm {
            ky: dy0 * fz0 / fz0.powf(ls_y),
            ls_y,
            kx: dx0 * fz0 / fz0.powf(ls_x),
            ls_x,
            fz0,
            tan_peak: fla.to_radians().tan(),
            tan_peak_loaded: (fla * fla_gain + fla).to_radians().tan(),
            sr_peak: if use_sr { sr } else { 0.0 },
            sr_peak_loaded: if use_sr { sr * sr_gain + sr } else { 0.0 },
            camber_gain: cg(1),
            dcamber0: dc0,
            dcamber1: dc1,
            cf_x_mult: nz(g(8), 1.0),
            brake_dx: if g(9) != 0.0 { g(9) + 1.0 } else { 1.0 },
            combined_exp: g(10),
            grip_slip_factor: nz(g(13), 0.75),
            falloff_y: fy,
            falloff_x: fx,
            mz_trail_remap: nz(g(22), 0.1),
            mz_nd_slip_reduction: nz(g(21), 0.8),
            mz_scale: nz(g(20), 0.12),
            brush_exp: if g(23) >= 1.0 { g(23) } else { 3.0 },
            new_combined: g(73) != 0.0,
            // as the game: an absent field is 0 (no effect)
            fx_shift_sa: g(74),
            fx_scale_sa: g(75),
            fy_shift_sr: g(76),
            fy_scale_sr: g(77),
            post_y: [g(80), g(81), g(82)],
            post_x: [g(83), g(84), g(85)],
            relax_y: g(18),
            relax_x: g(19),
            grain_factor: g(16),
            pressure_gain: sub(td, 6).and_then(|(c, e)| f32_at(td, c, e, 1)).unwrap_or(0.1),
            radius: td_f(3, 2).unwrap_or(0.33),
            rate: td_f(3, 3).filter(|v| *v > 0.0).unwrap_or(250_000.0),
            rate_prog: td_f(3, 300).unwrap_or(0.0),
            inertia: td_f(3, 5).filter(|v| *v > 0.0).unwrap_or(1.5),
            damping: td_f(3, 4).unwrap_or(500.0),
            mass: td_f(3, 10).filter(|v| *v > 0.0).unwrap_or(12.0),
            groove: td_f(10, 1).unwrap_or(0.0),
            groove_sa: td_f(10, 2).unwrap_or(0.0),
            groove_sr: td_f(10, 3).unwrap_or(0.0),
        })
    }

    /// Contact patch length at load `fz`: the chord of the tyre at its
    /// vertical deflection (rate d + prog d^2 = fz).
    pub fn patch_len(&self, fz: f32) -> f32 {
        let fz = fz.max(0.0);
        let d = if self.rate_prog > 0.0 {
            (-self.rate + (self.rate * self.rate + 4.0 * self.rate_prog * fz).sqrt()) / (2.0 * self.rate_prog)
        } else {
            fz / self.rate
        };
        let h = self.radius - d;
        if d <= 0.0 || h <= 0.0 { 0.0 } else { 2.0 * (self.radius * self.radius - h * h).sqrt() }
    }

    /// Peak friction at load (the game's D(Fz) = K Fz^ls / Fz).
    pub fn d_y(&self, fz: f32) -> f32 {
        if fz > 0.0 { fz.powf(self.ls_y) * self.ky / fz } else { 0.0 }
    }

    pub fn d_x(&self, fz: f32) -> f32 {
        if fz > 0.0 { fz.powf(self.ls_x) * self.kx / fz } else { 0.0 }
    }

    /// The brush curve: force factor at slip `s` for peak `d`, stiffness
    /// `k` (1 / peak slip) - saturates at d past the peak.
    /// The brush curve to its peak at s = 1/k, then the post-peak fall
    /// (fn 0x1425081b0).
    fn brush(&self, d: f32, k: f32, s: f32, post: [f32; 3]) -> f32 {
        if s > 1.0 / k {
            let [min, rate, exp] = post;
            if rate > 0.0 && exp > 0.0 && !sctm_dbg().0 {
                return ((1.0 - min) / (((s - 1.0 / k) * rate).powf(exp) + 1.0) + min) * d;
            }
            return d;
        }
        let x = 1.0 - k * s;
        let sign = if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 };
        (1.0 - x.abs().powf(self.brush_exp) * sign) * d
    }

    pub fn solve(&self, i: &Input) -> Output {
        let mut o = Output::default();
        let fz = i.fz;
        if fz <= 0.0 || (i.alpha == 0.0 && i.sr == 0.0 && i.camber == 0.0) {
            return o;
        }
        let alpha = i.alpha + self.camber_gain * i.camber;
        let wear = 1.0 + 0.2 * (i.wear * 0.01).clamp(0.0, 1.0);
        let dy0 = self.d_y(fz) / wear;
        let dx0 = self.d_x(fz) / wear;
        let mut dx = i.grip * dx0;
        if i.sr < 0.0 {
            dx *= self.brake_dx;
        }
        // camber: the side of the slip decides the sign
        let mut g = i.camber.abs();
        if (i.camber >= 0.0 && alpha >= 0.0) || (i.camber <= 0.0 && alpha <= 0.0) {
            g = -g;
        }
        let mut c = self.dcamber0 * g - g * g * self.dcamber1;
        if c <= -1.0 {
            c = -0.9;
        }
        let mut dy = i.grip * dy0 / (c + 1.0);
        let v_slip = ((i.speed * alpha.sin()).powi(2) + (alpha.cos() * i.speed * i.sr).powi(2)).sqrt();
        let t = |f: &Falloff| if i.temp.is_nan() { f.t_ref } else { i.temp };
        dy = self.falloff_y.apply(v_slip, dy, t(&self.falloff_y));
        dx = self.falloff_x.apply(v_slip, dx, t(&self.falloff_x));
        // standing water (fn 0x142297850, when p[12] > 0): the tread clears
        // grooveFactor of it, less with slip; what it leaves costs WET_K per
        // m/s of patch speed - a slick on a fully wet track has nothing left
        // by 40 m/s, a wet tyre keeps ~2/3
        if i.wet > 0.0 {
            let q = (self.groove - i.alpha.abs() * self.groove_sa - i.sr.abs() * self.groove_sr).clamp(0.0, 1.0);
            let f = (1.0 - (1.0 - q) * WET_K * i.wet * i.speed).clamp(0.0, 1.0);
            dy *= f;
            dx *= f;
        }
        let dl = (fz - self.fz0) / self.fz0;
        let gs = self.grip_slip_factor * (i.grip - 1.0) + 1.0;
        let pk = (1.0 + self.pressure_gain * i.pressure).max(0.1);
        let ky = 1.0 / (((self.tan_peak_loaded - self.tan_peak) * dl + self.tan_peak) * gs)
            * pk / (self.grain_factor * i.grain + 1.0);
        let kx = if self.sr_peak > 0.0 {
            1.0 / (((self.sr_peak_loaded - self.sr_peak) * dl + self.sr_peak) * gs) * pk
        } else {
            self.cf_x_mult * ky
        };
        // the game's lateral slip: sin / max(|cos|, 0.2), not tan - it stays
        // finite sideways (<= 5 at 90 deg). ⚠ With tan, a slick's
        // fx_scale_SA -0.1 in the old combined formula's 1 / (scale tan^2 + 1)
        // went to zero at 72 deg: a sliding wheel made 4-8x its load in Fx
        // and the car "caught" its slides and launched forward.
        let sy = alpha.sin() / alpha.cos().abs().max(0.2);
        let dbg = sctm_dbg();
        let sx = i.sr;
        let n = self.combined_exp;
        let comb = |a: f32, b: f32| if n <= 0.0 || n == 2.0 {
            (a * a + b * b).sqrt()
        } else {
            (a.abs().powf(n) + b.abs().powf(n)).powf(1.0 / n)
        };
        let s = comb(sx, sy);
        let (fy, fx) = if s <= 0.0 {
            (0.0, 0.0)
        } else if !self.new_combined {
            (self.brush(dy, ky, s, self.post_y) * (sy / s) * fz, self.brush(dx, kx, s, self.post_x) * (sx / s) * fz)
        } else if dbg.1 {
            // EVOVIEW_SCTM_OLDNC=1: the old new-combined reading (scales
            // floored at 0.05 so the sideways singularity cannot fire)
            let fx = self.brush(dx, kx / ((self.fx_shift_sa * sy * sy).max(-0.8) + 1.0), sx.abs(), self.post_x)
                / (self.fx_scale_sa * sy * sy + 1.0).max(0.5) * fz * if sx < 0.0 { -1.0 } else { 1.0 };
            let fy = self.brush(dy, ky / ((self.fy_shift_sr * sx * sx).max(-0.8) + 1.0), sy.abs(), self.post_y)
                / (self.fy_scale_sr * sx * sx + 1.0).max(0.5) * fz * if sy < 0.0 { -1.0 } else { 1.0 };
            (fy, fx)
        } else {
            // the game's new combined (fn 0x1425086e0, flag +0xb8): both
            // forces from the combined slip, each pointed by its own
            // shifted direction and cut by the other slip, at most 99 %
            let ys = comb(self.fy_shift_sr * sx, sy);
            let xs = comb(sx, self.fx_shift_sa * sy);
            let fy = if ys > 0.0 {
                (1.0 - (sx.abs() * self.fy_scale_sr).min(0.99)) * self.brush(dy, ky, s, self.post_y) * (sy / ys) * fz
            } else {
                0.0
            };
            let fx = if xs > 0.0 {
                (1.0 - (sy.abs() * self.fx_scale_sa).min(0.99)) * self.brush(dx, kx, s, self.post_x) * (sx / xs) * fz
            } else {
                0.0
            };
            (fy, fx)
        };
        let nd = s * ky;
        let t = (1.0 - self.mz_nd_slip_reduction * nd).clamp(0.0, 1.0);
        let trail = ((3.0 - (t + t)) * t * t * (self.mz_trail_remap + 1.0) - self.mz_trail_remap)
            * if i.trail_base.is_nan() { self.patch_len(i.fz) } else { i.trail_base }
            * self.mz_scale * i.speed.clamp(0.0, 1.0);
        o.fy = fy;
        o.fx = fx;
        o.trail = trail;
        o.mz = -fy * trail;
        o.nd_slip = nd;
        o.dy = dy;
        o.dx = dx;
        o
    }
}

/// Debug switches, read once: EVOVIEW_SCTM_NOPOST=1 (flat past the peak),
/// EVOVIEW_SCTM_OLDNC=1 (the earlier new-combined reading).
fn sctm_dbg() -> (bool, bool) {
    static D: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    *D.get_or_init(|| (crate::envc::var_os("EVOVIEW_SCTM_NOPOST").is_some(), crate::envc::var_os("EVOVIEW_SCTM_OLDNC").is_some()))
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn tyre() -> Sctm {
        let fy = Falloff { mu0t: 0.9, sens: 0.01, t_sens: 0.4, t_ref: 70.0, ref_speed: 2.0 };
        Sctm {
            ky: 1.5 * 4000f32 / 4000f32.powf(0.85), ls_y: 0.85, kx: 1.6 * 4000f32 / 4000f32.powf(0.85), ls_x: 0.85,
            fz0: 4000.0, tan_peak: 8f32.to_radians().tan(), tan_peak_loaded: 9f32.to_radians().tan(),
            sr_peak: 0.0, sr_peak_loaded: 0.0, camber_gain: 0.0, dcamber0: 0.1, dcamber1: -0.8,
            cf_x_mult: 1.0, brake_dx: 1.0, combined_exp: 0.0, grip_slip_factor: 0.75,
            falloff_y: fy, falloff_x: fy, mz_trail_remap: 0.1, mz_nd_slip_reduction: 0.8, mz_scale: 0.12,
            brush_exp: 3.0, new_combined: false, fx_shift_sa: 1.0, fx_scale_sa: 0.4, fy_shift_sr: 2.0, fy_scale_sr: 1.5,
            post_y: [0.0; 3], post_x: [0.0; 3], relax_y: 0.0, relax_x: 0.0, grain_factor: 0.0, pressure_gain: 0.1,
            radius: 0.33, rate: 250_000.0, rate_prog: 0.0, inertia: 1.5, damping: 500.0, mass: 12.0,
            groove: 0.0, groove_sa: 0.45, groove_sr: 0.45,
        }
    }

    #[test]
    fn peaks_at_d_times_fz() {
        let t = tyre();
        let i = Input { fz: 4000.0, alpha: 12f32.to_radians(), speed: 0.0, ..Default::default() };
        let o = t.solve(&i);
        // past the peak slip, at zero slip speed: exactly D(Fz0) Fz0 = Dy0 Fz0
        assert!((o.fy - 1.5 * 4000.0).abs() < 1.0, "{o:?}");
    }

    #[test]
    fn sideways_slick_stays_within_its_grip() {
        // a slick's new-combined data: fx_scale_SA -0.1 made Fx blow up at 72 deg
        let t = Sctm { new_combined: true, combined_exp: 1.705, fx_shift_sa: 0.9, fx_scale_sa: -0.1,
                       fy_shift_sr: 0.85, fy_scale_sr: 0.5, post_y: [0.75, 5.0, 1.5], post_x: [0.8, 2.5, 1.65], ..tyre() };
        for deg in [60.0f32, 70.0, 72.0, 72.4, 75.0, 85.0, 90.0] {
            for sr in [0.0f32, 0.05, 0.5, 3.0] {
                let o = t.solve(&Input { fz: 4000.0, alpha: deg.to_radians(), sr, speed: 10.0, ..Default::default() });
                let f = (o.fx * o.fx + o.fy * o.fy).sqrt();
                assert!(f < 2.0 * 4000.0, "{deg} deg sr {sr}: {o:?}");
            }
        }
        // far past the peak the lateral force has fallen toward 75 %
        let peak = t.solve(&Input { fz: 4000.0, alpha: 10f32.to_radians(), ..Default::default() }).fy;
        let deep = t.solve(&Input { fz: 4000.0, alpha: 60f32.to_radians(), ..Default::default() }).fy;
        assert!(deep < 0.9 * peak && deep > 0.6 * peak, "{peak} {deep}");
    }

    #[test]
    fn water_costs_a_slick_more_than_a_wet() {
        let slick = tyre();
        let wet = Sctm { groove: 0.68, ..tyre() };
        let i = |w: f32| Input { fz: 4000.0, alpha: 12f32.to_radians(), speed: 30.0, wet: w, ..Default::default() };
        let (s0, s1) = (slick.solve(&i(0.0)).dy, slick.solve(&i(1.0)).dy);
        let (w0, w1) = (wet.solve(&i(0.0)).dy, wet.solve(&i(1.0)).dy);
        assert!(s1 < 0.35 * s0 && w1 > 0.6 * w0, "{s0} {s1} {w0} {w1}");
    }

    #[test]
    fn linear_then_saturating_and_odd() {
        let t = tyre();
        let f = |a: f32| t.solve(&Input { fz: 4000.0, alpha: a.to_radians(), ..Default::default() }).fy;
        assert!(f(1.0) < f(4.0) && f(4.0) < f(8.0));
        assert!((f(-4.0) + f(4.0)).abs() < 1e-3);
    }

    #[test]
    fn aligning_moment_restoring_and_fading() {
        let t = tyre();
        let o = t.solve(&Input { fz: 4000.0, alpha: 2f32.to_radians(), speed: 20.0, ..Default::default() });
        assert!(o.trail > 0.0 && o.mz * o.fy < 0.0, "{o:?}");
        let deep = t.solve(&Input { fz: 4000.0, alpha: 20f32.to_radians(), speed: 20.0, ..Default::default() });
        assert!(deep.trail < o.trail);
    }
}

/// `--sctm-dump`: the slip-speed falloff and peak of every shipped compound,
/// to sanity-check field units against the whole roster.
pub fn dump(pkg: &std::path::Path) {
    let Ok(p) = crate::pkg::Package::open_filtered(pkg, &[r"content\cars\common_phsx\tyres"]) else { return };
    let mut keys: Vec<_> = p.by_path.keys().filter(|k| k.ends_with(".tyre")).cloned().collect();
    keys.sort();
    for k in keys {
        let td = p.read(p.by_path[&k]);
        let Some(m) = Sctm::from_compound(&td) else { continue };
        if crate::envc::var("EVOVIEW_SCTM_CURVES").is_ok() {
            let txt = String::from_utf8_lossy(&td).to_string();
            for w in txt.split(|c: char| !(c.is_ascii_alphanumeric() || "_./\\".contains(c))).filter(|w| w.ends_with(".curve")) {
                let ck = p.by_path.keys().find(|q| q.ends_with(&w.to_lowercase().replace('/', "\\"))).cloned();
                let pts = ck.map(|c| crate::carphys::read_curve(&p.read(p.by_path[&c]))).unwrap_or_default();
                println!("SCTMCURVE {k} {w} {pts:?}");
            }
        }
        if crate::envc::var("EVOVIEW_SCTM_SUBS").is_ok() {
            for sf in [3u32, 4, 5, 6, 8, 12] {
                let Some((a, b)) = sub(&td, sf) else { continue };
                let mut out = String::new();
                fields(&td, a, b, |f, v| out += &format!(" {f}={}", v.f32()));
                println!("SCTMSUB {k} f{sf}:{out}");
            }
        }
        let f = m.falloff_y;
        println!("SCTMDUMP {k} mu0t {} sens {} tsens {} tref {} vref {} | D(fz0) {:.3} nc {} exp {}",
            f.mu0t, f.sens, f.t_sens, f.t_ref, f.ref_speed, m.d_y(m.fz0), m.new_combined, m.combined_exp);
    }
}

#[cfg(test)]
mod patch_tests {
    #[test]
    fn gt3_patch_is_about_20cm() {
        let mut t = super::tests::tyre();
        t.radius = 0.335;
        t.rate = 300_873.0;
        t.rate_prog = 179_531.0;
        let l = t.patch_len(4170.0);
        assert!((0.17..0.23).contains(&l), "{l}");
    }
}
