//! Time of day, done the game's way: where the sun is, and what every
//! sun-angle-driven setting reads at that moment.
//!
//! **The sun** (exe 0x142786ce0, read from the shipped binary 2026-10-27) is
//! Paul Schlyter's "How to compute planetary positions": a day number from
//! 2000 Jan 0 (`367y - 7(y + (m+9)/12)/4 + 275m/9 + D - 730530`), his rates
//! for w / e / M / obliquity, one Kepler step for the eccentric anomaly, RA
//! and declination, then local sidereal time `L/15 + 12 + UT + lon/15` and
//! the horizon frame. The engine keeps `(az', zenith)` with
//! `az' = 180 - azimuth` (azimuth from north, clockwise) and builds the light
//! vector (exe 0x142864330) as
//!
//! ```text
//! p = (az' - 90) deg
//! light = (-sin z cos p, -cos z, sin z sin p)       (pointing away from it)
//! ```
//!
//! which is +X east, -Z north - the world evoview draws in, unmirrored.
//! The moon slot (exe 0x142786a20) is the same computation 12 h later and
//! 25 deg lower: there is no lunar orbit.
//!
//! **Sun angle.** Every `*_by_sun_elevation` curve in the game is keyed by
//! the zenith angle, 0 overhead, 90 on the horizon, 180 straight down. The
//! curves that cover 0..360 tell evening from morning: 0..180 is the sun
//! going down, 180..360 coming back up (`angle()`).
//!
//! **Time zones** are the game's own table (content/data/timezones): a UTC
//! offset per region plus EU / US / Australian daylight saving.
//!
//! What is NOT the game's: the sun and sky colour and brightness. The game's
//! sky is a precomputed atmosphere that has not been decoded, so those come
//! from a small transmittance model (`light()`), normalised so that a high
//! sun reproduces the constants evoview was calibrated with. Every such
//! point is recorded as a strict-mode guess.

use glam::Vec3;
use std::sync::RwLock;

// ------------------------------------------------------------------ clock --

/// The game's TimeZone enum (DateTime.proto).
pub const TZ_NAMES: [&str; 18] = [
    "UTC", "local", "Italy", "Germany", "Belgium", "Spain (Madrid)", "Spain (Barcelona)", "France",
    "Netherlands", "United Kingdom", "Hungary", "Austria", "Australia (Bathurst)", "US (Texas)",
    "South Africa", "US (Pacific)", "Japan", "US (Eastern)",
];

#[derive(Clone, Copy, PartialEq)]
enum Dst {
    None,
    Eu,
    Us,
    Au,
}

/// (standard UTC offset in hours, daylight-saving rule) of a TimeZone value.
fn zone(tz: u32) -> (f64, Dst) {
    match tz {
        2..=8 | 10 | 11 => (1.0, Dst::Eu),
        9 => (0.0, Dst::Eu),
        12 => (10.0, Dst::Au),
        13 => (-6.0, Dst::Us),
        14 => (2.0, Dst::None),
        15 => (-8.0, Dst::Us),
        16 => (9.0, Dst::None),
        17 => (-5.0, Dst::Us),
        _ => (0.0, Dst::None),
    }
}

/// 0 = Sunday.
fn weekday(y: i32, m: u32, d: u32) -> u32 {
    let t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if m < 3 { y - 1 } else { y };
    ((y + y / 4 - y / 100 + y / 400 + t[(m - 1) as usize] + d as i32) % 7) as u32
}

fn days_in(y: i32, m: u32) -> u32 {
    match m {
        2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Day of month of the n-th (1-based; 0 = last) Sunday.
fn sunday(y: i32, m: u32, n: u32) -> u32 {
    if n == 0 {
        let last = days_in(y, m);
        return last - weekday(y, m, last);
    }
    let first = 1 + (7 - weekday(y, m, 1)) % 7;
    first + 7 * (n - 1)
}

/// Is daylight saving in force on this local date? (Day resolution: the
/// switch-over hour itself is not modelled.)
fn dst_on(rule: Dst, y: i32, m: u32, d: u32) -> bool {
    let md = m * 100 + d;
    match rule {
        Dst::None => false,
        Dst::Eu => md >= 300 + sunday(y, 3, 0) && md < 1000 + sunday(y, 10, 0),
        Dst::Us => md >= 300 + sunday(y, 3, 2) && md < 1100 + sunday(y, 11, 1),
        Dst::Au => md >= 1000 + sunday(y, 10, 1) || md < 400 + sunday(y, 4, 1),
    }
}

/// UTC offset (hours) of a TimeZone value on a local date.
pub fn utc_offset(tz: u32, y: i32, m: u32, d: u32) -> f64 {
    let (std, rule) = zone(tz);
    std + if dst_on(rule, y, m, d) { 1.0 } else { 0.0 }
}

/// Where and when: what the game's DateTimeLocation holds.
#[derive(Clone, Copy, Debug)]
pub struct Clock {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    /// local time of day, minutes
    pub minutes: f64,
    pub lat: f64,
    pub lon: f64,
    pub tz: u32,
}

impl Default for Clock {
    /// The game's default session: 2014-08-15 (the thumbnail rig's date) at
    /// 14:00, at 0 N 0 E until a track says where it is.
    fn default() -> Self {
        Clock { year: 2014, month: 8, day: 15, minutes: 14.0 * 60.0, lat: 0.0, lon: 0.0, tz: 0 }
    }
}

impl Clock {
    pub fn hhmm(&self) -> String {
        let m = self.minutes.rem_euclid(1440.0) as u32;
        format!("{:02}:{:02}", m / 60, m % 60)
    }

    /// Move by `mins`, rolling the date over.
    pub fn step(&mut self, mins: f64) {
        self.minutes += mins;
        while self.minutes >= 1440.0 {
            self.minutes -= 1440.0;
            self.day += 1;
            if self.day > days_in(self.year, self.month) {
                self.day = 1;
                self.month += 1;
                if self.month > 12 {
                    self.month = 1;
                    self.year += 1;
                }
            }
        }
        while self.minutes < 0.0 {
            self.minutes += 1440.0;
            if self.day > 1 {
                self.day -= 1;
            } else {
                if self.month > 1 {
                    self.month -= 1;
                } else {
                    self.month = 12;
                    self.year -= 1;
                }
                self.day = days_in(self.year, self.month);
            }
        }
    }
}

// ------------------------------------------------------------------- sun --

/// The sun's (az', zenith) in degrees, exactly as the engine computes them.
pub fn sun_angles(c: &Clock) -> (f64, f64) {
    let ut = c.minutes / 60.0 - utc_offset(c.tz, c.year, c.month, c.day);
    schlyter(c.year, c.month, c.day, ut, c.lat, c.lon, 0.0)
}

/// The moon slot: the sun at UT + 12 h, 25 degrees lower.
pub fn moon_angles(c: &Clock) -> (f64, f64) {
    let ut = c.minutes / 60.0 - utc_offset(c.tz, c.year, c.month, c.day);
    let (az, zen) = schlyter(c.year, c.month, c.day, ut, c.lat, c.lon, 12.0);
    (az, zen + 25.0)
}

fn rev(x: f64) -> f64 {
    x.rem_euclid(std::f64::consts::TAU)
}

fn schlyter(y: i32, m: u32, dd: u32, ut: f64, lat: f64, lon: f64, extra_h: f64) -> (f64, f64) {
    let (yi, mi, di) = (y as i64, m as i64, dd as i64);
    let d = (367 * yi - (7 * (yi + (mi + 9) / 12)) / 4 + (275 * mi) / 9 + di - 730530) as f64 + ut / 24.0;
    let rad = std::f64::consts::PI / 180.0;
    let w = 282.9404 + 4.70935e-5 * d;
    let e = 0.016709 - 1.151e-9 * d;
    let m_deg = 356.047 + 0.9856002585 * d;
    let oblecl = 23.4393 * rad - 3.563e-7 * rad * d;
    let l = rev(w * rad + m_deg * rad);
    let mr = rev(m_deg * rad);
    let ea = rev((m_deg + e / rad * mr.sin() * (1.0 + e * mr.cos())) * rad);
    let x = ea.cos() - e;
    let yv = (1.0 - e * e).max(0.0).sqrt() * ea.sin();
    let r = (x * x + yv * yv).sqrt();
    let mut lon_s = yv.atan2(x) / rad + w;
    if lon_s > 360.0 {
        lon_s -= 360.0;
    }
    let ls = lon_s * rad;
    let (xs, ys) = (r * ls.cos(), r * ls.sin());
    let (xe, ye, ze) = (xs, ys * oblecl.cos(), ys * oblecl.sin());
    let dec = ze.atan2((xe * xe + ye * ye).sqrt());
    let ra = ye.atan2(xe) / rad;
    let mut lst = l / rad / 15.0 + 12.0 + ut + extra_h + lon / 15.0;
    if lst > 24.0 {
        lst -= 24.0;
    }
    let ha = rev((lst * 15.0 - ra) * rad);
    let (x1, y1, z1) = (ha.cos() * dec.cos(), ha.sin() * dec.cos(), dec.sin());
    let la = lat * rad;
    let xh = x1 * la.sin() - z1 * la.cos();
    let zh = x1 * la.cos() + z1 * la.sin();
    let az = rev(y1.atan2(xh) + std::f64::consts::PI);
    let alt = zh.clamp(-1.0, 1.0).asin() / rad;
    let az_out = rev(std::f64::consts::PI - az) / rad;
    (az_out, 90.0 - alt)
}

/// Unit vector TOWARD the body (the negated engine light vector).
pub fn direction(az_out: f64, zenith: f64) -> Vec3 {
    let p = ((az_out - 90.0) as f32).to_radians();
    let z = (zenith as f32).to_radians();
    Vec3::new(z.sin() * p.cos(), z.cos(), -z.sin() * p.sin()).normalize()
}

// ---------------------------------------------------------------- curves --

/// Piecewise-linear curve, points (t, value), clamped at the ends.
#[derive(Clone, Copy)]
pub struct Curve(pub &'static [(f32, f32)]);

impl Curve {
    pub fn at(&self, t: f32) -> f32 {
        let p = self.0;
        if p.is_empty() {
            return 0.0;
        }
        if t <= p[0].0 {
            return p[0].1;
        }
        for w in p.windows(2) {
            let ((t0, v0), (t1, v1)) = (w[0], w[1]);
            if t <= t1 {
                let k = if t1 > t0 { (t - t0) / (t1 - t0) } else { 1.0 };
                return v0 + (v1 - v0) * k;
            }
        }
        p[p.len() - 1].1
    }
}

/// The game's own curves (content/tracks/common_assets), as decoded with the
/// game's protobuf schema. Ship-safe: a handful of numbers, not game files.
pub mod curves {
    use super::Curve;
    /// post_process/natural6/exposure_compensation_1 (EV), the default
    /// (editor/default) and natural7 preset's
    pub const EXPOSURE_COMP: Curve = Curve(&[
        (0.0, -1.2), (40.0, -1.2), (78.0, -0.8), (88.0, 0.5), (90.0, 0.7), (94.0, 4.0), (99.0, 4.5),
        (180.0, 4.5), (261.0, 4.5), (266.0, 4.0), (270.0, 0.7), (272.0, 0.5), (277.0, -0.8),
        (320.0, -1.2), (360.0, -1.2),
    ]);
    /// natural6/min_brightness, max_brightness: the auto-exposure clamp
    pub const MIN_BRIGHTNESS: Curve = Curve(&[
        (0.0, 0.05), (80.0, 0.05), (90.0, 0.45), (96.0, 0.45), (99.0, 1.0), (260.0, 1.0),
        (264.0, 0.45), (270.0, 0.45), (280.0, 0.05), (360.0, 0.05),
    ]);
    pub const MAX_BRIGHTNESS: Curve = Curve(&[
        (0.0, 10.0), (85.0, 10.0), (95.0, 1.0), (100.0, 0.5), (270.0, 0.5), (275.0, 1.0),
        (285.0, 10.0), (360.0, 10.0),
    ]);
    /// natural6/white_balance, tint (editor units, as grade::Preset)
    pub const WB_TEMP: Curve = Curve(&[
        (0.0, 0.0), (70.0, 0.0), (86.5, -10.0), (90.0, -30.0), (260.0, -30.0), (270.0, -30.0),
        (273.5, -10.0), (290.0, 0.0), (359.0, 0.0),
    ]);
    pub const WB_TINT: Curve = Curve(&[
        (-0.1, 0.0), (74.2449, -0.203), (84.7462, -9.55), (93.1956, -8.14), (98.0, -4.48),
        (104.806, 0.0), (180.0, 0.0), (262.0, 0.0), (266.0, 60.0), (269.0, 0.0), (359.0, 0.0),
    ]);
    /// natural6/master_gain_4 and master_gamma_4, channel w (UE4 scalar)
    pub const MASTER_GAIN: Curve = Curve(&[
        (0.0, 1.0), (60.0, 1.0), (90.0, 0.4), (115.0, 0.6), (180.0, 0.6), (245.0, 0.6),
        (270.0, 0.4), (300.0, 1.0), (360.0, 1.0),
    ]);
    pub const MASTER_GAMMA: Curve = Curve(&[
        (0.0, 1.0), (60.0, 1.0), (90.0, 1.3), (115.0, 1.2), (180.0, 1.2), (245.0, 1.2),
        (270.0, 1.3), (300.0, 1.0), (358.558, 1.0),
    ]);
    /// environment: night_irradiance (the night IV multiplier)
    pub const NIGHT_IRRADIANCE: Curve = Curve(&[(90.0, 10.0), (270.0, 10.0)]);
    /// environment: stars_intensity_by_sun_elevation
    pub const STARS: Curve = Curve(&[(83.0, 0.0), (84.0, 0.0), (88.0, 0.005), (95.0, 0.015), (180.0, 0.015)]);
    /// environment: reflection_probe_intensity_by_sun_elevation (channel x)
    pub const REFLECTION: Curve = Curve(&[(0.0, 1.0), (80.0, 1.0), (85.0, 0.37), (90.0, 0.23), (98.0, 0.0), (180.0, 0.0)]);
    /// environment: box_reflection_probe_intensity (channel x): never below 0.15
    pub const BOX_REFLECTION: Curve = Curve(&[(0.0, 1.0), (80.0, 1.0), (85.0, 0.37), (90.0, 0.23), (98.0, 0.15), (180.0, 0.15)]);
    /// environment: sky_inscatter_scale_main_pass_by_sun_elevation
    pub const SKY_INSCATTER: Curve = Curve(&[
        (0.0, 1.0), (75.38, 1.0), (83.0, 1.0), (96.0, 1.0), (98.0, 0.1), (105.0, 0.25),
        (120.0, 0.4), (180.0, 0.5),
    ]);
    /// environment: car_shadow_decal_intensity_by_sunelevation
    pub const SHADOW_DECAL: Curve = Curve(&[(0.0, 1.0), (80.0, 1.0), (85.0, 0.9), (95.0, 0.65), (180.0, 0.5)]);
    /// night_gsaa_vehicle_itensity_sun_elevation: how far "night" the car
    /// shading is (0 day, 1 night)
    pub const NIGHT: Curve = Curve(&[(0.0, 0.0), (85.0, 0.0), (100.0, 1.0), (250.0, 1.0), (275.0, 0.0), (360.0, 0.0)]);
}

// ----------------------------------------------------------------- state --

/// Everything the renderer reads for one moment of the day.
#[derive(Clone, Copy, Debug)]
pub struct Sky {
    pub clock: Clock,
    /// toward the sun / moon
    pub sun: Vec3,
    pub moon: Vec3,
    /// zenith angle of the sun, degrees
    pub zenith: f32,
    /// the 0..360 curve key
    pub angle: f32,
    /// direct sun radiance multiplier per channel, 1 = evoview's calibrated
    /// noon sun
    pub sun_rgb: [f32; 3],
    /// sky (ambient) multiplier, and its colour at zenith / horizon
    pub sky_k: f32,
    pub zenith_rgb: [f32; 3],
    pub horizon_rgb: [f32; 3],
    /// moonlight, multiplier on the sun's calibrated colour
    pub moon_k: f32,
    /// 0 day .. 1 night
    pub night: f32,
    /// exposure multiplier relative to the calibrated daylight exposure
    pub exposure: f32,
    pub stars: f32,
    pub reflection: f32,
    pub night_iv: f32,
    pub shadow_decal: f32,
    pub wb_temp: f32,
    pub wb_tint: f32,
    pub gain: f32,
    pub gamma: f32,
    /// the auto exposure's modelled scene average (game units, DAY_LUM
    /// scale) and the preset's clamp on it at this sun angle
    pub ae_lum: f32,
    pub ae_min: f32,
    pub ae_max: f32,
}

/// Curve key: zenith on the way down (afternoon), 360 - zenith on the way
/// up (morning).
pub fn angle(c: &Clock, zenith: f64) -> f32 {
    let (y, m, d) = (c.year, c.month, c.day);
    let off = utc_offset(c.tz, y, m, d);
    // solar noon: where the zenith angle is smallest; a few minutes either
    // side of the clock's position say which way it is moving
    let probe = |mins: f64| {
        let mut k = *c;
        k.minutes = mins;
        let ut = k.minutes / 60.0 - off;
        schlyter(y, m, d, ut, c.lat, c.lon, 0.0).1
    };
    let later = probe(c.minutes + 2.0);
    let earlier = probe(c.minutes - 2.0);
    if later >= earlier {
        zenith as f32
    } else {
        (360.0 - zenith) as f32
    }
}

/// Sun colour and strength through the atmosphere at altitude `alt_deg`.
/// ⚠ Guess (see module docs): Rayleigh + aerosol optical depth at 650 / 550
/// / 450 nm and Kasten-Young air mass, normalised to a 60-degree sun.
fn transmittance(alt_deg: f32) -> [f32; 3] {
    let tau = [0.0503 + 0.0711, 0.1004 + 0.0884, 0.2249 + 0.1146];
    let air = |h: f32| -> f32 {
        let h = h.max(-2.5);
        1.0 / (h.to_radians().sin().max(0.0) + 0.50572 * (h + 6.07995).powf(-1.6364))
    };
    let (m, m0) = (air(alt_deg), air(60.0));
    std::array::from_fn(|i| (-(tau[i] * (m - m0))).exp())
}

fn env_k(name: &str, d: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

impl Sky {
    pub fn at(clock: Clock) -> Sky {
        let (az, zen) = sun_angles(&clock);
        let (maz, mzen) = moon_angles(&clock);
        let ang = angle(&clock, zen);
        let alt = 90.0 - zen as f32;
        let malt = 90.0 - mzen as f32;
        let c = curves::EXPOSURE_COMP;

        // direct sun: the disc leaves between +2 and -1 degrees
        let t = transmittance(alt);
        let up = smoothstep(-1.0, 2.0, alt);
        let sun_rgb = [t[0] * up, t[1] * up, t[2] * up];

        // sky light: full by day, ~10x down per 3 degrees of twilight,
        // starlight + glow floor at night
        let day_sky = 0.35 + 0.65 * smoothstep(0.0, 35.0, alt);
        let sky_k = if alt >= 0.0 { day_sky } else { (0.35 * 10f32.powf(alt / 3.0)).max(env_k("EVOVIEW_NIGHT_SKY", 4.0e-5)) };
        let golden = 1.0 - smoothstep(0.0, 12.0, alt);
        let dusk = 1.0 - smoothstep(-8.0, 0.0, alt);
        let zenith_rgb = mix3(mix3([0.38, 0.56, 0.92], [0.30, 0.42, 0.78], golden), [0.10, 0.16, 0.42], dusk);
        let horizon_rgb = mix3(mix3([0.78, 0.82, 0.88], [1.0, 0.62, 0.36], golden), [0.45, 0.32, 0.36], dusk);

        // moonlight: ~1/400000 of the sun is real; the game boosts nights
        // with exposure, so this is about what stays visible after it
        let moon_k = env_k("EVOVIEW_MOON", 3.0e-5) * smoothstep(-1.0, 5.0, malt);

        // auto exposure, the game's way with a modelled average: adapt to
        // the scene, clamped by the preset's min / max brightness, then
        // 2^compensation. 10 = the daytime ceiling every preset sits on.
        // ⚠ The game's scene average is in its own units, where a clear day
        // sits far above the max-brightness clamp of 10 (exposure is pinned
        // there all day). DAY_LUM is how far above: it decides how late in
        // the dusk the exposure starts to open up. 300 = from about the horizon
        // the horizon, fitted by eye (strict SKY2).
        let lum = env_k("EVOVIEW_DAY_LUM", 300.0) * (0.6 * sun_rgb[1] + 0.4 * sky_k + 0.6 * moon_k);
        let adapted = lum.clamp(curves::MIN_BRIGHTNESS.at(ang).min(curves::MAX_BRIGHTNESS.at(ang)),
                                curves::MAX_BRIGHTNESS.at(ang).max(0.05));
        let day_adapted = 10.0f32;
        let game = (day_adapted / adapted) * (c.at(ang) - c.at(45.0)).exp2();
        // ⚠ Cap (strict SKY2): never brighter on screen than noon while the
        // scene itself is darker. Without the game's real scene average
        // the compensation curve alone overexposes the dusk ~3x.
        let scene = (0.6 * sun_rgb[1] + 0.4 * sky_k + 0.6 * moon_k).max(1e-6);
        let exposure = game.min(env_k("EVOVIEW_DUSK_CAP", 0.7) / scene).max(1.0f32.min(game));

        Sky {
            clock,
            sun: direction(az, zen),
            moon: direction(maz, mzen),
            zenith: zen as f32,
            angle: ang,
            sun_rgb,
            sky_k,
            zenith_rgb,
            horizon_rgb,
            moon_k,
            night: curves::NIGHT.at(ang),
            exposure,
            stars: curves::STARS.at(zen as f32),
            reflection: curves::REFLECTION.at(zen as f32),
            night_iv: curves::NIGHT_IRRADIANCE.at(zen as f32),
            shadow_decal: curves::SHADOW_DECAL.at(zen as f32),
            wb_temp: curves::WB_TEMP.at(ang),
            wb_tint: curves::WB_TINT.at(ang),
            ae_lum: lum,
            ae_min: curves::MIN_BRIGHTNESS.at(ang).min(curves::MAX_BRIGHTNESS.at(ang)),
            ae_max: curves::MAX_BRIGHTNESS.at(ang).max(0.05),
            gain: curves::MASTER_GAIN.at(ang),
            gamma: curves::MASTER_GAMMA.at(ang),
        }
    }

    /// The direction shadows are cast from: the sun, or the moon at night.
    pub fn key(&self) -> Vec3 {
        if self.sun.y > -0.02 { self.sun } else { self.moon }
    }

    /// Four vec4s for the shaders (render::Uniforms.sky, track frame):
    /// 0 xyz key light dir, w key strength (sun or moon, 1 = calibrated sun)
    /// 1 rgb key colour multiplier, w sky multiplier
    /// 2 rgb sky zenith colour, w night 0..1
    /// 3 rgb sky horizon colour, w night irradiance multiplier
    pub fn pack(&self) -> [[f32; 4]; 4] {
        let k = self.key();
        let (rgb, s) = if self.sun.y > -0.02 {
            (self.sun_rgb, 1.0)
        } else {
            ([0.55, 0.68, 1.0], self.moon_k)
        };
        [
            [k.x, k.y, k.z, s],
            [rgb[0], rgb[1], rgb[2], self.sky_k],
            [self.zenith_rgb[0], self.zenith_rgb[1], self.zenith_rgb[2], self.night],
            [self.horizon_rgb[0], self.horizon_rgb[1], self.horizon_rgb[2], self.night_iv],
        ]
    }

    pub fn describe(&self) -> String {
        format!(
            "{} {:04}-{:02}-{:02} ({}) sun alt {:.1} deg, curve angle {:.0}, exposure x{:.1}",
            self.clock.hhmm(), self.clock.year, self.clock.month, self.clock.day,
            TZ_NAMES.get(self.clock.tz as usize).unwrap_or(&"?"), 90.0 - self.zenith, self.angle,
            self.exposure)
    }
}

static CLOCK: RwLock<Option<Clock>> = RwLock::new(None);
/// Bumped on every change, so per-frame users can tell when to re-bake.
static GEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The clock in use. Starts from the default, the track's location (`place`)
/// and the --date / --tod flags.
pub fn clock() -> Clock {
    if let Some(c) = *CLOCK.read().unwrap() {
        return c;
    }
    let mut c = Clock::default();
    apply_flags(&mut c);
    *CLOCK.write().unwrap() = Some(c);
    c
}

pub fn set(c: Clock) {
    *CLOCK.write().unwrap() = Some(c);
    GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

pub fn generation() -> u32 {
    GEN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Put the clock where the track is (Track Info coordinates and time zone).
pub fn place(lat: f64, lon: f64, tz: u32) {
    let mut c = clock();
    c.lat = lat;
    c.lon = lon;
    c.tz = tz;
    set(c);
}

pub fn step(mins: f64) {
    let mut c = clock();
    c.step(mins);
    set(c);
}

pub fn now() -> Sky {
    static CACHE: RwLock<Option<(u32, Sky)>> = RwLock::new(None);
    let g = generation();
    if let Some((cg, s)) = *CACHE.read().unwrap() {
        if cg == g {
            return s;
        }
    }
    let s = Sky::at(clock());
    *CACHE.write().unwrap() = Some((g, s));
    s
}

/// The shader block (`Sky::pack`) in use: the time of day once one is
/// chosen, else the calibrated fixed sun the viewer always had.
pub fn block() -> [[f32; 4]; 4] {
    let mut b = if active() {
        now().pack()
    } else {
        let d = crate::render::SUN.normalize();
        [
            [d.x, d.y, d.z, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.38, 0.56, 0.92, 0.0],
            [0.78, 0.82, 0.88, 0.0],
        ]
    };
    // rain: the sun goes behind the deck, the sky to a flat grey
    let r = crate::rain::rain();
    if r > 0.0 {
        b[0][3] *= 1.0 - 0.85 * r;
        for row in [2usize, 3] {
            let l = b[row][0] * 0.3 + b[row][1] * 0.59 + b[row][2] * 0.11;
            for c in 0..3 {
                b[row][c] += (l * 0.85 - b[row][c]) * (0.8 * r);
            }
        }
        b[1][3] *= 1.0 - 0.25 * r;
    }
    b
}

/// Exposure multiplier on top of the calibrated outdoor exposure.
pub fn exposure() -> f32 {
    if active() { now().exposure } else { 1.0 }
}

/// (modelled scene average, min, max brightness) for the auto exposure's
/// clamp. Without a time of day: the calibrated noon (natural6 noon clamp).
pub fn ae_clamp() -> (f32, f32, f32) {
    if active() {
        let s = now();
        (s.ae_lum, s.ae_min, s.ae_max)
    } else {
        (env_k("EVOVIEW_DAY_LUM", 300.0), 0.05, 10.0)
    }
}

/// Direction shadows are cast from.
pub fn key_dir() -> Vec3 {
    let b = block()[0];
    Vec3::new(b[0], b[1], b[2]).normalize()
}

/// Whether a time of day has been chosen (by flag or key): until then the
/// viewer keeps its calibrated fixed sun, so old shots do not change.
pub fn active() -> bool {
    ACTIVE.load(std::sync::atomic::Ordering::Relaxed)
}

static ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn activate() {
    ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
    GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// --tod HH:MM and --date YYYY-MM-DD (also EVOVIEW_TOD / EVOVIEW_DATE).
fn apply_flags(c: &mut Clock) {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str, env: &str| {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
            .or_else(|| std::env::var(env).ok())
    };
    if let Some(t) = flag("--tod", "EVOVIEW_TOD") {
        let p: Vec<f64> = t.split(':').filter_map(|x| x.trim().parse().ok()).collect();
        if let Some(h) = p.first() {
            c.minutes = h * 60.0 + p.get(1).copied().unwrap_or(0.0);
            ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if let Some(d) = flag("--date", "EVOVIEW_DATE") {
        let p: Vec<i64> = d.split('-').filter_map(|x| x.trim().parse().ok()).collect();
        if p.len() == 3 {
            c.year = p[0] as i32;
            c.month = (p[1] as u32).clamp(1, 12);
            c.day = (p[2] as u32).clamp(1, 31);
            ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noon_is_high_and_midnight_is_low() {
        // Monza, 21 June, local noon-ish vs midnight (CEST)
        let mut c = Clock { year: 2025, month: 6, day: 21, minutes: 13.0 * 60.0 + 30.0,
                            lat: 45.62, lon: 9.28, tz: 2 };
        let (_, z) = sun_angles(&c);
        assert!((90.0 - z - 67.0).abs() < 1.5, "noon altitude {}", 90.0 - z);
        c.minutes = 60.0;
        assert!(sun_angles(&c).1 > 100.0);
    }

    #[test]
    fn morning_sun_is_in_the_east() {
        // equator, equinox, 07:00 UTC at 0 E: sun low in the east (+X)
        let c = Clock { year: 2025, month: 3, day: 20, minutes: 7.0 * 60.0, lat: 0.0, lon: 0.0, tz: 0 };
        let (az, zen) = sun_angles(&c);
        let d = direction(az, zen);
        assert!(d.x > 0.9 && d.y > 0.1, "{d:?}");
    }

    #[test]
    fn thumbnail_rig_sun() {
        // the game's thumbnail date at 0/0: ~57 deg up
        let c = Clock::default();
        let c = Clock { minutes: 600.0, ..c };
        let (_, z) = sun_angles(&c);
        assert!((90.0 - z - 57.0).abs() < 1.5, "{}", 90.0 - z);
    }

    #[test]
    fn dst_rules() {
        assert_eq!(utc_offset(2, 2025, 7, 1), 2.0);
        assert_eq!(utc_offset(2, 2025, 1, 1), 1.0);
        assert_eq!(utc_offset(12, 2025, 1, 1), 11.0);
        assert_eq!(utc_offset(13, 2025, 7, 4), -5.0);
    }

    #[test]
    fn evening_and_morning_keys() {
        let mut c = Clock { year: 2025, month: 6, day: 21, minutes: 18.0 * 60.0, lat: 45.6, lon: 9.3, tz: 2 };
        let (_, z) = sun_angles(&c);
        assert!(angle(&c, z) < 180.0);
        c.minutes = 8.0 * 60.0;
        let (_, z) = sun_angles(&c);
        assert!(angle(&c, z) > 180.0);
    }
}

