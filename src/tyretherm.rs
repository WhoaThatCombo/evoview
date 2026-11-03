//! The game's tyre thermal model (TyreThermalModel, client exe 20260907:
//! init FUN_1423ef550 / geometry FUN_1423ef5c0, contact heat FUN_1423eea30,
//! step FUN_1423efa50, the tyre's call FUN_14236xxxx around 0x8021440 of the
//! full decompile). Per tyre: 12 patches round the tread x 3 across it
//! (inner, middle, outer), each linked to its neighbours (round the tread
//! wrapping, across not), plus one core.
//!
//! Heat in, every physics step, to the three patches in contact (the column
//! under the tyre, from its turned angle):
//!   Q   = heat_partition_coeff x mu(Fz) Fz x slip speed       (friction)
//!       + surface_core_mass_ratio x rolling                    (rolling)
//!   per patch ((road T - their mean) road_conduction A + Q / 3) x a weight
//!   for its row: 1 + c, 1, 1 - c with c = clamp(contactIMO camber, +-0.75)
//!   (the pressure-off-ideal term is left at ideal: evoview has no pressure).
//! The core gets the rest of the rolling heat. Then each patch cools to the
//! air at h = Re^0.571 0.318 cool_factor 4 0.02624 / L (Re = L v / 1.525e-5,
//! h 30 below 1 m/s; x (1 + cool_factor_rain wet); x 2.5 over 180 C), trades
//! heat with its neighbours (thermal_conductivity / 0.02 x w 0.01 / 3) and
//! with the core (the same conductance x the patch area).
//!
//! Out: the "practical" temperature, lerp(core, the contact column's surface
//! weighted by camber, practicalTempSource), read through the compound's
//! thermalPerformanceCurve as a grip multiplier; and the tyre model's input
//! temperature lerp(mean surface, core, practicalTempSource).
//!
//! Pressure (FUN_142370f30 / FUN_14236ede0, the tyre's step): it follows
//! the core, P = 0.16 ((core - air) 0.6 + air - 26) + an offset set at the
//! reset so the tyre starts at its idealPressure; grip / (1 + |P - ideal|
//! gainD) (FUN_14236e470), and P / ideal - 1 shifts the contact heat to the
//! middle (over-inflated) or the shoulders.
//!
//! Flat spots (FUN_142374210): while the wheel is locked (|omega| <= 0.3 or
//! slip ratio < -0.98) above 3 m/s, flat += FlatSpotK x speed x Fz x 1e-5
//! x dt (x the car's damage and a tyre factor, both 1 here). The flat spot
//! itself touches no force (only the FFB thump, rigcar); at 1 the tyre
//! blows (FUN_142374210 / FUN_142374820, also when the core passes
//! explosionTemperature): its pressure goes (+0x500 = ideal) and its
//! working radius drops to the rim's (+0x458 = 0 in the radius getter).

const COLS: usize = 12;
const ROWS: usize = 3;
const N: usize = COLS * ROWS;

/// TyresCompoundData f5 TyresThermalPatchData, and the tyre's size (f3).
#[derive(Clone, Debug)]
pub struct Data {
    pub density: f32,
    pub specific_heat: f32,
    pub surface_core_mass_ratio: f32,
    pub cool_factor_rain: f32,
    pub cool_factor: f32,
    pub heat_partition: f32,
    pub practical_temp_source: f32,
    pub contact_imo: f32,
    pub road_conduction: f32,
    pub conductivity: f32,
    pub rolling_factor: f32,
    /// TyresData f1 width, f2 radius, f6 rimRadius, f16 blanketTemperature
    pub width: f32,
    pub radius: f32,
    pub rim_radius: f32,
    pub blanket: f32,
    /// TyresPressureData (f6): f4 gainD, f5 idealPressure (psi); TyresData
    /// f17 FlatSpotK
    pub gain_d: f32,
    pub ideal_p: f32,
    pub flat_spot_k: f32,
    /// TyresData f15 explosionTemperature (C; 0 = never)
    pub explosion_t: f32,
    /// Tread wear (FUN_14236b020): TyresData f8 treadHeightMM, f9
    /// treadConsumptionK; TyresModelData f17 contactWearIMO (camber split),
    /// f14 wearCurve (tread mm -> grip %), and the reference load f5 Fz0.
    pub tread_mm: f32,
    pub tread_k: f32,
    pub wear_imo: f32,
    pub wear_curve: Vec<(f32, f32)>,
    pub fz0: f32,
    /// Graining (FUN_142374350): TyresThermalPatchData f16 grainGain, f17
    /// grainGamma, f18 grainSlipAngleGain, f19 grainSlipAngleGamma, f20
    /// grainSlipAngleThreshold (deg); TyresModelData f16 grainFactor (the
    /// tyre model's cornering stiffness / (1 + factor grain%)).
    pub grain_gain: f32,
    pub grain_gamma: f32,
    pub grain_sa_gain: f32,
    pub grain_sa_gamma: f32,
    pub grain_sa_th: f32,
    pub grain_factor: f32,
    /// thermalPerformanceCurve: practical temperature -> grip multiplier
    pub curve: Vec<(f32, f32)>,
    /// Rolling resistance (the tyre step, FUN_14236b1e0 -> tyre+0x400):
    /// RollingResistanceData (f9) rr0 + rr1 v^2 (per kN of load), and
    /// TyresPressureData f2 rollingResistanceGain (under-inflated rolls
    /// harder). None: the compound has no f9.
    pub rr: Option<(f32, f32)>,
    pub rr_p_gain: f32,
    /// The radial spring under pressure (FUN_14236e6d0): TyresPressureData
    /// f8 pressureSpringGain (N/m per psi off ideal), or f7
    /// pressureSpringCurve (psi -> N/m, replaces the rate when present)
    pub spring_gain: f32,
    pub spring_curve: Vec<(f32, f32)>,
    /// TyresPressureData f1 pressure_limit_angle_gain
    pub limit_gain: f32,
}

impl Data {
    /// From a .tyre (TyresCompoundData); `curve` loads a .curve by path.
    pub fn read(td: &[u8], curve: &mut dyn FnMut(&str) -> Vec<(f32, f32)>) -> Option<Data> {
        use crate::proto::{fields, Val};
        let sub = |d: &[u8], want: u32| {
            let mut hit = None;
            fields(d, 0, d.len(), |f, v| if f == want && hit.is_none() { hit = v.range() });
            hit
        };
        let (a, b) = sub(td, 5)?;
        let (mut g, mut path) = (std::collections::HashMap::new(), None);
        fields(td, a, b, |f, v| match v {
            Val::F32(x) => { g.insert(f, x); }
            _ => if f == 25 {
                if let Some((s, e)) = v.range() {
                    path = std::str::from_utf8(&td[s..e]).ok().map(|s| s.to_string());
                }
            }
        });
        let f = |k: u32, d: f32| g.get(&k).copied().filter(|v| *v != 0.0).unwrap_or(d);
        let (mut w, mut r, mut rim, mut blanket, mut flat, mut boom) = (0.25f32, 0.33f32, 0.23f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut tread, mut tread_k) = (0.0f32, 0.0f32);
        if let Some((s, e)) = sub(td, 3) {
            fields(td, s, e, |k, v| if let Val::F32(x) = v {
                match k { 1 => w = x, 2 => r = x, 6 => rim = x, 8 => tread = x, 9 => tread_k = x,
                          15 => boom = x, 16 => blanket = x, 17 => flat = x, _ => {} }
            });
        }
        let (mut imo, mut gfac, mut fz0, mut wpath) = (0.0f32, 0.0f32, 4000.0f32, None);
        if let Some((s, e)) = sub(td, 4) {
            fields(td, s, e, |k, v| match v {
                Val::F32(x) => match k { 5 => fz0 = x, 16 => gfac = x, 17 => imo = x, _ => {} },
                _ => if k == 14 {
                    if let Some((a, b)) = v.range() {
                        wpath = std::str::from_utf8(&td[a..b]).ok().map(|s| s.to_string());
                    }
                }
            });
        }
        let (mut gain_d, mut ideal_p, mut rr_p_gain, mut spring_gain, mut spath, mut limit_gain) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, None, 0.0f32);
        if let Some((s, e)) = sub(td, 6) {
            fields(td, s, e, |k, v| match v {
                Val::F32(x) => match k { 1 => limit_gain = x, 2 => rr_p_gain = x, 4 => gain_d = x, 5 => ideal_p = x, 8 => spring_gain = x, _ => {} },
                _ => if k == 7 {
                    if let Some((a, b)) = v.range() {
                        spath = std::str::from_utf8(&td[a..b]).ok().filter(|s| !s.is_empty()).map(|s| s.to_string());
                    }
                }
            });
        }
        let rr = sub(td, 9).map(|(s, e)| {
            let (mut a, mut b) = (0.0f32, 0.0f32);
            fields(td, s, e, |k, v| if let Val::F32(x) = v {
                match k { 1 => a = x, 2 => b = x, _ => {} }
            });
            (a, b)
        });
        if !g.contains_key(&110) {
            return None;
        }
        Some(Data {
            density: f(110, 1100.0),
            specific_heat: f(111, 1600.0),
            surface_core_mass_ratio: f(112, 0.15),
            cool_factor_rain: f(113, 0.0),
            cool_factor: f(114, 1.0),
            heat_partition: f(115, 1.0),
            practical_temp_source: g.get(&23).copied().unwrap_or(0.0),
            contact_imo: f(26, 0.0),
            road_conduction: f(116, 0.0),
            conductivity: f(117, 0.3),
            rolling_factor: f(118, 1.0),
            width: w,
            radius: r,
            rim_radius: if rim > 0.05 && rim < r { rim } else { r * 0.7 },
            blanket,
            gain_d,
            ideal_p,
            flat_spot_k: flat,
            explosion_t: boom,
            tread_mm: tread,
            tread_k,
            wear_imo: imo,
            wear_curve: wpath.map(|p| curve(&p)).unwrap_or_default(),
            fz0: if fz0 > 0.0 { fz0 } else { 4000.0 },
            grain_gain: g.get(&16).copied().unwrap_or(0.0),
            grain_gamma: g.get(&17).copied().unwrap_or(1.0),
            grain_sa_gain: g.get(&18).copied().unwrap_or(0.0),
            grain_sa_gamma: g.get(&19).copied().unwrap_or(1.0),
            grain_sa_th: g.get(&20).copied().unwrap_or(0.0),
            grain_factor: gfac,
            curve: path.map(|p| curve(&p)).unwrap_or_default(),
            rr,
            rr_p_gain,
            spring_gain,
            spring_curve: spath.map(|p| curve(&p)).unwrap_or_default(),
            limit_gain,
        })
    }
}

/// One tyre's temperatures.
#[derive(Clone, Debug)]
pub struct Tyre {
    /// patch temperatures, row-major (row 0 = the inner side for +camber)
    pub t: [f64; N],
    q: [f64; N],
    pub core: f64,
    core_q: f64,
    /// the turned fraction of a revolution: which column is on the road
    phase: f64,
    /// pressure = from_core(core, air) + p_offset (psi)
    p_offset: f32,
    /// flat spot 0..1, and the tyre blown (at a full flat spot or over its
    /// explosion temperature): no pressure, rolling on the rim
    pub flat: f32,
    pub blown: bool,
    /// tread left per row (inner, middle, outer), mm, and graining, %
    pub tread: [f32; 3],
    pub grain: f32,
    /// the air temperature of the last step (the pressure reads it)
    air: f32,
    // geometry (FUN_1423ef5c0)
    c_patch: f64,
    c_core: f64,
    area: f64,
    area_nb: f64,
    k: f64,
    len: f64,
}

impl Tyre {
    pub fn new(d: &Data, t0: f32, air: f32) -> Tyre {
        let (w, r, rr) = (d.width as f64, d.radius as f64, d.rim_radius as f64);
        let pi = std::f64::consts::PI;
        let mass = (pi * r * r * w - pi * rr * rr * w) * 0.27 * d.density as f64;
        let scmr = d.surface_core_mass_ratio as f64;
        Tyre {
            t: [t0 as f64; N],
            q: [0.0; N],
            core: t0 as f64,
            core_q: 0.0,
            phase: 0.0,
            p_offset: d.ideal_p - from_core(t0, air),
            flat: 0.0,
            blown: false,
            tread: [d.tread_mm; 3],
            grain: 0.0,
            air,
            c_patch: scmr * mass * d.specific_heat as f64 / N as f64,
            c_core: (1.0 - scmr) * mass * d.specific_heat as f64 / N as f64,
            area: 2.0 * pi * r * w / N as f64,
            area_nb: w * 0.01 / ROWS as f64,
            k: d.conductivity as f64 / 0.02,
            len: 1.0 / (1.0 / w + 0.5 / r),
        }
    }

    fn col(&self) -> usize {
        ((COLS as f64 * self.phase) as usize) % COLS
    }

    /// The step's heat: `fz` N, `mu` the tyre's peak friction at that load,
    /// `v_slip` m/s, `omega` rad/s, `camber` rad, road temperature C, `rr_k`
    /// the surface's rolling resistance multiplier (10 in sand).
    pub fn heat(&mut self, d: &Data, fz: f32, mu: f32, v_slip: f32, omega: f32, camber: f32, road: f32, rr_k: f32) {
        if fz <= 0.0 {
            return;
        }
        let scmr = d.surface_core_mass_ratio as f64;
        // rolling: |rolling resistance torque x omega| x rolling_factor
        let rolling = (self.rr_torque(d, fz, omega) * rr_k * omega).abs() as f64 * d.rolling_factor as f64;
        let q = d.heat_partition as f64 * (mu * fz * v_slip.abs()) as f64 + scmr * rolling;
        let col = self.col();
        let c = (d.contact_imo * camber).clamp(-0.75, 0.75) as f64;
        // over-inflated: the middle carries more, the shoulders less
        let p3 = if d.ideal_p > 0.0 { (self.pressure_raw() / d.ideal_p - 1.0) as f64 } else { 0.0 };
        let (mid, edge, scale) = (1.0 + p3 * 0.1, -p3 * 0.1 * 0.5, 1.0 - p3 * 0.5);
        let idx = [col, COLS + col, 2 * COLS + col];
        let mean = idx.iter().map(|&i| self.t[i]).sum::<f64>() / 3.0;
        let dq = (road as f64 - mean) * d.road_conduction as f64 * self.area + q / 3.0;
        for (row, wgt) in [(0usize, 1.0 + c + edge), (1, mid), (2, 1.0 - c + edge)] {
            self.q[idx[row]] += (wgt * scale).max(0.0) * dq;
        }
        self.core_q += (1.0 - scmr) * rolling;
    }

    /// The rolling resistance torque, N m (unsigned; it opposes the spin).
    /// The game's (tyre+0x400 in FUN_14236b1e0) = min(Fz, 10 kN)/1000 x
    /// (rr0 + rr1 v^2) x ((ideal/P - 1) rollingResistanceGain + 1) x radius.
    /// Left out: its rr_slip term (gated on an unidentified value > 20),
    /// the track's wet factor, rrWearMult, and x10 on surface kind 4. A
    /// compound without the data: 1.2 % of the load.
    pub fn rr_torque(&self, d: &Data, fz: f32, omega: f32) -> f32 {
        let v = omega * d.radius;
        let coef = match d.rr {
            Some((a, b)) => {
                let p = self.pressure_raw();
                let pk = if p > 0.0 && d.ideal_p > 0.0 { (d.ideal_p / p - 1.0) * d.rr_p_gain + 1.0 } else { 1.0 };
                (a + b * v * v) * pk.max(0.0) * 0.001
            }
            None => 0.012,
        };
        coef * fz.clamp(0.0, 10000.0) * d.radius
    }

    /// Advance `dt` s: `omega` rad/s, air temperature C, `wet` 0..1.
    pub fn step(&mut self, d: &Data, dt: f32, omega: f32, air: f32, wet: f32) {
        let dt = dt as f64;
        let om = omega as f64;
        self.phase = (dt * om / std::f64::consts::TAU + self.phase).rem_euclid(1.0);
        self.core += dt * self.core_q / (N as f64 * self.c_core);
        self.core_q = 0.0;
        let v = (om * d.radius as f64).abs();
        let mut h = if v >= 1.0 {
            (self.len * v / 1.525e-5).powf(0.571) * 0.318 * d.cool_factor as f64 * 4.0 * (0.02624 / self.len)
        } else {
            30.0
        };
        h *= 1.0 + d.cool_factor_rain as f64 * wet as f64;
        self.air = air;
        if d.explosion_t > 0.0 && self.core > d.explosion_t as f64 && std::env::var_os("EVOVIEW_NO_BLOWOUT").is_none() {
            self.blown = true;
        }
        let prev = self.t;
        for i in 0..N {
            let (row, col) = (i / COLS, i % COLS);
            let mut t = prev[i] + dt * self.q[i] / self.c_patch;
            self.q[i] = 0.0;
            let hh = if prev[i] > 180.0 { h * 2.5 } else { h };
            t += (air as f64 - prev[i]) * hh * self.area * dt / self.c_patch;
            let mut nb = |j: usize| t += (prev[j] - prev[i]) * self.k * self.area_nb * dt / self.c_patch;
            nb(row * COLS + (col + 1) % COLS);
            nb(row * COLS + (col + COLS - 1) % COLS);
            if row > 0 {
                nb(i - COLS);
            }
            if row + 1 < ROWS {
                nb(i + COLS);
            }
            self.t[i] = t;
        }
        let mut sum = 0.0;
        for i in 0..N {
            let q = (self.core - prev[i]) * self.k * self.area * dt;
            self.t[i] += q / self.c_patch;
            sum += q;
        }
        self.core -= sum / self.c_core;
    }

    fn pressure_raw(&self) -> f32 {
        self.p_offset + from_core(self.core as f32, self.air)
    }

    /// Set the pressure to `psi` at the tyre's current temperatures: the
    /// setup's cold pressure at the start (the game's reset fn 0x142370f30
    /// sets the offset so), or a pit stop's / the garage's for a fitted
    /// tyre. It then follows the core from there.
    pub fn set_pressure(&mut self, psi: f32) {
        self.p_offset = psi - from_core(self.core as f32, self.air);
    }

    /// The radial spring rate (N/m) at this pressure, from the tyre's own
    /// `rate` (FUN_14236e6d0): pressureSpringCurve(P) when the data has one,
    /// else rate + (P - ideal) x pressureSpringGain. A blown tyre keeps the
    /// plain rate (its change is the rim radius). Not ported: the game's
    /// speed stiffening (smoothstep(|v| / tyre+0x270) x tyre+0x274; fields
    /// unidentified).
    pub fn spring_rate(&self, d: &Data, rate: f32) -> f32 {
        if self.blown || d.ideal_p <= 0.0 {
            return rate;
        }
        let p = self.pressure_raw().max(0.0);
        if !d.spring_curve.is_empty() {
            return lut(&d.spring_curve, p).max(0.0);
        }
        (rate + (p - d.ideal_p) * d.spring_gain).max(rate * 0.2)
    }

    /// The tyre model's pressure input p[10]: pressure / ideal - 1 (0 when
    /// blown or without an idealPressure).
    pub fn pressure_ratio(&self, d: &Data) -> f32 {
        if self.blown || d.ideal_p <= 0.0 { 0.0 } else { self.pressure_raw().max(0.0) / d.ideal_p - 1.0 }
    }

    /// Pressure, psi (0 when the data has no idealPressure).
    pub fn pressure(&self, d: &Data) -> f32 {
        if self.blown { 0.0 } else if d.ideal_p > 0.0 { self.pressure_raw().max(0.0) } else { 0.0 }
    }

    /// The wheel locked or near it above 3 m/s grinds a flat spot.
    pub fn flat_spot(&mut self, d: &Data, omega: f32, sr: f32, speed: f32, fz: f32, dt: f32) {
        if (omega.abs() <= 0.3 || sr < -0.98) && speed > 3.0 && fz > 0.0 {
            self.flat = (self.flat + d.flat_spot_k * speed * fz * 1e-5 * dt).min(1.0);
            if self.flat >= 1.0 && std::env::var_os("EVOVIEW_NO_BLOWOUT").is_none() {
                self.blown = true;
            }
        }
    }

    pub fn mean(&self) -> f32 {
        (self.t.iter().sum::<f64>() / N as f64) as f32
    }

    /// The contact column's surface, weighted toward the loaded edge.
    pub fn surface(&self, d: &Data, camber: f32) -> f32 {
        let col = self.col();
        let c = (d.contact_imo * camber).clamp(-1.0, 1.0) as f64;
        (((1.0 - c) * self.t[2 * COLS + col] + (1.0 + c) * self.t[col] + self.t[COLS + col]) / 3.0) as f32
    }

    /// lerp(core, surface, practicalTempSource): what the grip curve reads.
    pub fn practical(&self, d: &Data, camber: f32) -> f32 {
        let core = self.core as f32;
        (self.surface(d, camber) - core) * d.practical_temp_source + core
    }

    /// lerp(mean surface, core, practicalTempSource): the tyre model's input.
    pub fn model_temp(&self, d: &Data) -> f32 {
        let m = self.mean();
        (self.core as f32 - m) * d.practical_temp_source + m
    }

    /// The thermal performance curve at the practical temperature.
    /// Tread wear and graining for one step (FUN_14236b020 / FUN_142374350):
    /// the tread goes by Fz / Fz0 x treadConsumptionK x slip speed x dt x
    /// 1e-4, a third per row split by camber, faster when the contact
    /// surface is over 60 C (the heat factor at tyre+0x2e4 is undecoded:
    /// 1 % per degree here); a row worn through blows the tyre. Graining
    /// builds over grainSlipAngleThreshold of slip angle at speed and wears
    /// off with rolling slip. (Blistering and cold graining have their
    /// temperature thresholds at defaults the data never sets - off.)
    pub fn wear(&mut self, d: &Data, fz: f32, v_slip: f32, speed: f32, alpha: f32, nd: f32, camber: f32, dt: f32) {
        if fz <= 0.0 || d.tread_mm <= 0.0 {
            return;
        }
        let amount = fz / d.fz0 * d.tread_k * v_slip.abs() * dt * 1e-4;
        let heat = 1.0 + ((self.surface(d, camber) - 60.0) * 0.01).max(0.0);
        let per = amount / 3.0 * heat;
        let c = (d.wear_imo * camber).clamp(-1.0, 1.0);
        self.tread[0] -= (1.0 + c) * per;
        self.tread[1] -= per;
        self.tread[2] -= (1.0 - c) * per;
        if self.tread.iter().any(|t| *t <= 0.0) && std::env::var_os("EVOVIEW_NO_BLOWOUT").is_none() {
            self.blown = true;
        }
        for t in self.tread.iter_mut() {
            *t = t.max(0.0);
        }
        if speed > 2.0 {
            let sa = alpha.abs().to_degrees().min(20.0);
            if d.grain_sa_th > 0.0 && sa > d.grain_sa_th {
                self.grain += speed * d.grain_sa_gain * 0.001 * (sa / d.grain_sa_th).powf(d.grain_sa_gamma) * dt;
            } else if d.grain_gain > 0.0 {
                self.grain -= nd.clamp(0.0, 2.5).powf(d.grain_gamma) * dt * speed * d.grain_gain * 5e-5;
            }
            self.grain = self.grain.clamp(0.0, 100.0);
        }
    }

    /// The camber-weighted tread under the patch, mm (FUN_14236ec10).
    pub fn tread_at(&self, d: &Data, camber: f32) -> f32 {
        let c = (d.wear_imo * camber).clamp(-1.0, 1.0);
        ((1.0 - c) * self.tread[2] + (1.0 + c) * self.tread[0] + self.tread[1]) / 3.0
    }

    pub fn grip(&self, d: &Data, camber: f32) -> f32 {
        let p = if d.ideal_p > 0.0 { 1.0 / (1.0 + (self.pressure(d) - d.ideal_p).abs() * d.gain_d) } else { 1.0 };
        // the wear curve: tread mm -> grip % (slick: 99 new, 95 at 1 mm, 80 at 0.1)
        let wear = if d.wear_curve.is_empty() || d.tread_mm <= 0.0 { 1.0 } else { lut(&d.wear_curve, self.tread_at(d, camber)) / 100.0 };
        let p = p * wear.max(0.0);
        // (blown: the game's tyre model keeps running on the rim at the
        // pressure term's grip - the radius is the change, rigcar)
        p * self.curve_at(d, camber)
    }

    fn curve_at(&self, d: &Data, camber: f32) -> f32 {
        if d.curve.is_empty() {
            return 1.0;
        }
        let x = self.practical(d, camber);
        let c = &d.curve;
        if x <= c[0].0 {
            return c[0].1;
        }
        for w in c.windows(2) {
            if x <= w[1].0 {
                let t = (x - w[0].0) / (w[1].0 - w[0].0).max(1e-6);
                return w[0].1 + (w[1].1 - w[0].1) * t;
            }
        }
        c[c.len() - 1].1
    }
}

fn lut(c: &[(f32, f32)], x: f32) -> f32 {
    if x <= c[0].0 {
        return c[0].1;
    }
    for w in c.windows(2) {
        if x <= w[1].0 {
            let t = (x - w[0].0) / (w[1].0 - w[0].0).max(1e-6);
            return w[0].1 + (w[1].1 - w[0].1) * t;
        }
    }
    c[c.len() - 1].1
}

/// The pressure's temperature part (FUN_14236ede0; 0.6 is DAT_145c95900).
fn from_core(core: f32, air: f32) -> f32 {
    ((core - air) * 0.6 + air - 26.0) * 0.16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Data {
        Data { density: 1100.0, specific_heat: 1600.0, surface_core_mass_ratio: 0.15, cool_factor_rain: 3.0,
               cool_factor: 0.9, heat_partition: 1.4, practical_temp_source: 0.8, contact_imo: 1.6,
               road_conduction: 1000.0, conductivity: 0.35, rolling_factor: 1.6, width: 0.3, radius: 0.34,
               rim_radius: 0.23, blanket: 0.0, gain_d: 0.0025, ideal_p: 27.0, flat_spot_k: 0.1, explosion_t: 450.0,
               tread_mm: 3.0, tread_k: 6.0, wear_imo: 4.0, wear_curve: vec![(0.0, 0.0), (0.1, 80.0), (1.0, 95.0), (2.9, 100.0), (3.0, 99.0)],
               fz0: 4000.0, grain_gain: 0.47, grain_gamma: 0.47, grain_sa_gain: 0.565, grain_sa_gamma: 0.845, grain_sa_th: 10.6, grain_factor: 0.001,
               curve: vec![(0.0, 0.8), (70.0, 1.0), (105.0, 1.0), (140.0, 0.9)], rr: None, rr_p_gain: 0.0, spring_gain: 0.0, spring_curve: Vec::new(), limit_gain: 0.0 }
    }

    #[test]
    fn sliding_heats_and_air_cools() {
        let d = data();
        let mut t = Tyre::new(&d, 30.0, 25.0);
        assert!((t.pressure(&d) - 27.0).abs() < 1e-3);
        let om = 40.0 / d.radius;
        for _ in 0..(20.0 / 0.003) as usize {
            t.heat(&d, 4000.0, 1.6, 1.0, om, 0.0, 30.0, 1.0);
            t.step(&d, 0.003, om, 25.0, 0.0);
        }
        let hot = t.mean();
        assert!(t.pressure(&d) > 27.0, "{}", t.pressure(&d));
        assert!(hot > 50.0 && hot < 250.0, "{hot}");
        for _ in 0..(60.0 / 0.003) as usize {
            t.step(&d, 0.003, om, 25.0, 0.0);
        }
        assert!(t.mean() < hot - 10.0, "{} vs {hot}", t.mean());
        assert!((t.grip(&d, 0.0) - 1.0).abs() < 0.25);
    }

    #[test]
    fn locked_wheel_flat_spots() {
        let d = data();
        let mut t = Tyre::new(&d, 60.0, 25.0);
        // one second locked at 30 m/s under 4 kN: 0.1 x 30 x 4000 x 1e-5
        for _ in 0..(1.0 / 0.003) as usize {
            t.flat_spot(&d, 0.0, -1.0, 30.0, 4000.0, 0.003);
        }
        assert!((t.flat - 0.12).abs() < 0.01, "{}", t.flat);
        // rolling does not
        t.flat_spot(&d, 80.0, -0.05, 30.0, 4000.0, 1.0);
        assert!((t.flat - 0.12).abs() < 0.01);
        assert!(!t.blown);
        // locked long enough: a full flat spot blows it
        for _ in 0..(10.0 / 0.003) as usize {
            t.flat_spot(&d, 0.0, -1.0, 30.0, 4000.0, 0.003);
        }
        assert!(t.blown && t.pressure(&d) == 0.0);
    }
}
