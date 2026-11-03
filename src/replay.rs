//! Reading Assetto Corsa EVO `.rpy` replays.
//!
//! Replays live in `Saved Games\ACE\Replay\Saved`. A file is a small header,
//! some protobuf session-metadata chunks, then one motion stream per car:
//!
//! ```text
//! u32   version                  16 in every file seen
//! f64   duration, seconds        at offset 2, NOT 4 - the double is packed
//!                                against the version field, MSB at byte 9
//! u32   123                      at offset 22
//! u32+  the recording car's model, length-prefixed
//! N x   [u32 len][protobuf]      track, session structure, entry list
//! u32   123
//! u16   car count
//! per car:
//!     16B guid, then the SAME guid again
//!     u32, u8                    always 0 so far
//!     u32 nMatrixFrames, then that many 104-byte frames
//!     u32 nBaseFrames,   then a variable-length telemetry block
//! ```
//!
//! The 104-byte matrix frame is what the viewer needs:
//!
//! ```text
//!   0  vec3  body position, world space, Y-up (evoview's own axes)
//!  12  8B    packed rotation, encoding unknown - and unnecessary, see below
//!  20  vec3  wheel front-RIGHT, world space
//!  32  vec3  wheel front-left
//!  44  vec3  wheel rear-RIGHT
//!  56  vec3  wheel rear-left
//!  68..103   two more vec3s and three scalars, not identified
//! ```
//!
//! Because the wheels are stored in WORLD space rather than relative to the
//! body, orientation comes straight out of them and the packed rotation can be
//! ignored entirely. Checked against direction of travel across whole laps:
//! `dot(forward, velocity)` lands between 0.98 and 1.000, and all 24 cars on a
//! Laguna Seca grid derive the same heading. It also yields real body roll and
//! pitch from suspension travel, which a stored body quaternion would not.
//!
//! `tools/rpy.py` is the same reader in Python, handy for poking at a file.

use glam::Vec3;
use std::path::Path;

/// Bytes per matrix frame.
const FRAME: usize = 104;

/// Car models are named nowhere in the header - not in the session metadata
/// and not in the entry list. They appear only in customization blocks AFTER
/// the car array, laid out as:
///
/// ```text
/// field 100 (tag a2 06, len 36)  the guid as a dashed string
/// field 101 (tag aa 06, len 21)  the same guid in binary
/// field  20 (tag a2 01)          customization, holding paths like
///     content\cars\ks_ford_mustang_gt3\skins\rimdesign_mstggt3_1.design
/// ```
///
/// ⚠ Requiring field 101 to follow field 100 is what makes this safe. A bare
/// guid string also turns up elsewhere in the tail, and anchoring on those
/// assigns cars the WRONG model. Anchored properly, 965 of 967 cars across all
/// 73 local replays resolve.
const GUID_FIELD: &[u8] = &[0xa2, 0x06, 0x24];
const PGUID_TAG: &[u8] = &[0xaa, 0x06];
const CAR_PREFIX: &[u8] = br"content\cars\";
/// How far past the guid to look for the car path before giving up.
const LOOKAHEAD: usize = 4096;

/// Bytes per base (telemetry) frame.
const BASE: usize = 48;

/// Half-precision float. The base frames pack almost everything this way; Rust
/// has no f16 in stable std, and pulling in a crate for eleven lines is silly.
fn f16(b: &[u8], at: usize) -> f32 {
    let h = u16::from_le_bytes([b[at], b[at + 1]]) as u32;
    let (sign, exp, man) = (h >> 15, (h >> 10) & 0x1f, h & 0x3ff);
    let bits = match exp {
        0 if man == 0 => sign << 31,
        // Subnormals are rare here but cheap to get right: normalise by hand.
        0 => {
            let shift = man.leading_zeros() - 21;
            (sign << 31) | ((127 - 15 - shift) << 23) | ((man << (shift + 1)) & 0x7f_ffff)
        }
        0x1f => (sign << 31) | 0x7f80_0000 | (man << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// What the car was actually doing, from the base-frame stream.
///
/// The 48-byte record, as measured against the motion in the matrix frames:
///
/// ```text
///   0  f16   engine rpm          1900 idle .. ~8000
///   2  f16   steering wheel, degrees (about +/-190)
///   4  u8    throttle 0..255     37% of a lap at 0, 49% pinned
///   5  u8    brake 0..255        74% of a lap at 0
///   6  f16 x4  wheel angular velocity, rad/s, same corner order as the
///             matrix frame (front-right, front-left, rear-right, rear-left)
///  14  f16   ?                   0..8950
///  16  f16 x3 ?                  roughly +/-48, +/-6, +/-58
///  22  f16   reference wheel speed, rad/s - the speedometer source
///  24  f16   lateral g           about +/-2.5
///  26  f16   ?                   about +/-4.3
///  28  f16   ?                   about +/-7.5
///  30  u8    flag, 0 or 1
///  31  u8    gear
///  32  ?     two bytes
///  34  zero x4
///  38  ?     two bytes
///  40  f64   timestamp, seconds from the start of the recording
/// ```
///
/// Verified rather than guessed: wheel rad/s times the tyre radius tracks the
/// speed derived from the matrix frames at a ratio of 0.97-1.00, deviating to
/// 1.08 under acceleration (wheelspin) and 0.93 under braking (lockup); and
/// rpm sawtooths against gear exactly as upshifts should.
#[derive(Debug, Clone, Copy, Default)]
pub struct Telemetry {
    pub rpm: f32,
    /// Steering wheel angle in degrees. ⚠ This is the WHEEL, not the road
    /// angle - a GT3 turns roughly 190 degrees lock to lock at the rim for
    /// something like 15 degrees at the tyre.
    pub steer_deg: f32,
    pub throttle: f32,
    pub brake: f32,
    /// Per-wheel angular velocity, rad/s.
    pub wheel: [f32; 4],
    /// Reference wheel speed, rad/s. Multiply by tyre radius for m/s.
    ///
    /// ⚠ Only the LOCAL player's car has this - it reads 0 for every remote
    /// driver, whose telemetry is reconstructed from network state. Use
    /// `Replay::speed` for a figure that works for every car.
    pub speed_rads: f32,
    pub lat_g: f32,
    pub gear: u8,
    pub time: f32,
    /// synthetic replays only (--convoy): the body scraping the ground 0..1
    /// and where, metres (along, lateral) from the middle of the contact patch
    pub scrape: f32,
    pub scrape_at: [f32; 2],
}

/// One car's pose at an instant, ready to hand to the renderer.
#[derive(Debug, Clone, Copy)]
pub struct Pose {
    pub pos: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub wheels: [Vec3; 4],
}

pub struct Car {
    pub guid: [u8; 16],
    /// The car folder under `content\cars`, when it could be resolved.
    ///
    /// ⚠ For mods this is the folder the LIVERY assets come from, which is not
    /// always the mod itself: tuning variants (`*_mod_mech_1`) resolve to their
    /// base car, and `rx8_widebody_drift` resolves to its donor
    /// `ks_toyota_gr86`. For rendering that is usually the right answer anyway,
    /// since it is the mesh the mod was built on.
    pub model: Option<String>,
    pub n_matrix: u32,
    pub n_base: u32,
    /// Offset of this car's frame 0 within the file.
    off: usize,
    /// Offset of this car's base-frame stream within the file.
    base_off: usize,
}

impl Car {
    /// The guid the way the game writes it in the customization blocks: the
    /// two halves read as little-endian u64s. That is also the tail of the
    /// entry-list id string `<steamid><a>-<b>`, so these are driver guids.
    pub fn dashed(&self) -> String {
        let a = u64::from_le_bytes(self.guid[..8].try_into().unwrap());
        let b = u64::from_le_bytes(self.guid[8..].try_into().unwrap());
        let s = format!("{a:016x}{b:016x}");
        format!("{}-{}-{}-{}-{}", &s[..8], &s[8..12], &s[12..16], &s[16..20], &s[20..])
    }
}

pub struct Replay {
    data: Vec<u8>,
    /// synthetic frames carry the body's own forward / up at bytes 68 / 80
    /// (a recorded frame's body basis comes from its wheels): the body can
    /// roll and pitch over wheels that stay on the ground
    explicit: bool,
    pub version: u32,
    /// The session's wall-clock length in seconds, from the header.
    ///
    /// ⚠ This is NOT the length of the captured motion. Long sessions record
    /// into a rolling buffer that wraps at roughly 17,800 frames, so a 60
    /// minute stint still only holds the last ~18 minutes of movement. Use
    /// `captured()` for playback length; this field only says how long the
    /// driver was in the session.
    pub duration: f32,
    /// The recording driver's car. Always present, and the fallback when a
    /// car's own model cannot be resolved.
    pub model: String,
    /// The track folder, read out of the session metadata, so a replay says
    /// which track to load rather than needing to be told.
    pub track: Option<String>,
    pub cars: Vec<Car>,
}

fn to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((b >> 13) & 0x3ff) as u16
}

/// A synthetic drift: `n` cars of one model sliding nose-in round a circle.
pub struct Donut {
    pub centre: Vec3,
    pub radius: f32,
    /// m/s along the circle
    pub speed: f32,
    /// nose-in slip angle, degrees
    pub angle: f32,
    pub seconds: f32,
    pub wheelbase: f32,
    pub track_width: f32,
    pub tyre_radius: f32,
    /// 0 = every car identical and evenly spaced; 1 = each its own line,
    /// angle and wobble (all whole multiples of the lap, so it still loops)
    pub variation: f32,
}

/// A pack of cars driving the racing line together (`--convoy`).
pub struct Convoy {
    /// the racing line, as recorded (a closed loop)
    pub line: Vec<Vec3>,
    /// metres along the line where the leading car starts
    pub start_s: f32,
    /// m/s
    pub speed: f32,
    /// metres between rows (two cars a row, side by side)
    pub gap: f32,
    pub seconds: f32,
    pub wheelbase: f32,
    pub track_width: f32,
    pub tyre_radius: f32,
}

/// Deterministic -1..1 per (car, salt).
fn jitter(k: usize, salt: u32) -> f32 {
    let mut x = (k as u32).wrapping_mul(0x9E37_79B9) ^ salt.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    (x as f32 / u32::MAX as f32) * 2.0 - 1.0
}

impl Replay {
    /// `n` cars of `model` doing one big shared donut on `track`, as a
    /// replay the normal playback path draws. `ground(x, z)` puts each
    /// wheel on the surface (None: the centre's height).
    pub fn donut(track: &str, model: &str, n: usize, d: &Donut, ground: &dyn Fn(f32, f32) -> Option<f32>) -> Replay {
        let rate = 50.0f32 / 3.0;
        let frames = (d.seconds * rate).ceil() as usize + 2;
        let omega = d.speed / d.radius;
        let mut data = Vec::new();
        let mut cars = Vec::new();
        for k in 0..n {
            let v = d.variation;
            let pi = std::f32::consts::PI;
            let phase = (k as f32 + 0.22 * v * jitter(k, 1)) / n as f32 * std::f32::consts::TAU;
            let r_k = d.radius * (1.0 + 0.09 * v * jitter(k, 2));
            let beta0 = d.angle + 8.0 * v * jitter(k, 3);
            // wobble rates: whole multiples of the lap rate
            let (m1, m2) = (1.0 + (k % 3) as f32, 2.0 + (k % 2) as f32);
            let off = data.len();
            let mut steer = Vec::with_capacity(frames);
            for i in 0..frames {
                let tt = i as f32 / rate;
                let w = omega * tt;
                // each car surges a little along the circle (+-0.5 m)
                let th = phase + w + 0.06 * v * (m1 * w + pi * jitter(k, 7)).sin();
                let radial = Vec3::new(th.cos(), 0.0, th.sin());
                let tangent = Vec3::new(-th.sin(), 0.0, th.cos());
                let r_now = r_k + 0.7 * v * (m2 * w + pi * jitter(k, 4)).sin();
                // each car breathes its angle its own way (plus a base 3 deg
                // so even variation 0 is not frozen)
                let beta = (beta0 + 3.0 * (m1 * w + k as f32 * 1.3).sin()
                    + v * (5.0 * (m1 * w + pi * jitter(k, 5)).sin() + 2.5 * (3.0 * w + pi * jitter(k, 6)).sin()))
                    .to_radians();
                let fwd = (tangent * beta.cos() - radial * beta.sin()).normalize();
                let right = Vec3::Y.cross(fwd).normalize();
                let c = d.centre + radial * r_now;
                let corner = |a: f32, s: f32| {
                    let g = c + fwd * (a * d.wheelbase * 0.5) + right * (s * d.track_width * 0.5);
                    let y = ground(g.x, g.z).unwrap_or(d.centre.y);
                    Vec3::new(g.x, y + d.tyre_radius, g.z)
                };
                // front-right, front-left, rear-right, rear-left
                let wh = [corner(1.0, 1.0), corner(1.0, -1.0), corner(-1.0, 1.0), corner(-1.0, -1.0)];
                let body = (wh[0] + wh[1] + wh[2] + wh[3]) * 0.25 + Vec3::Y * 0.1;
                let mut f = [0u8; FRAME];
                let put = |f: &mut [u8; FRAME], at: usize, v: Vec3| {
                    for (j, x) in [v.x, v.y, v.z].iter().enumerate() {
                        f[at + j * 4..at + j * 4 + 4].copy_from_slice(&x.to_le_bytes());
                    }
                };
                put(&mut f, 0, body);
                // the body over the wheels: rolled outward by the cornering
                // load, squatting on the throttle, and bouncing a touch
                // (1.5 / 2 Hz = 9 and 12 x the lap rate, so it loops)
                let lat_g = r_now * omega * omega / 9.81;
                let roll = (3.4 * lat_g + 0.35 * (9.0 * w + pi * jitter(k, 8)).sin()).to_radians();
                let pitch = (1.1 + 0.3 * (12.0 * w + pi * jitter(k, 9)).sin()).to_radians();
                let out = radial - fwd * radial.dot(fwd);
                let up = (Vec3::Y + out.normalize_or_zero() * roll.tan() - fwd * pitch.tan()).normalize();
                let bf = (fwd - up * fwd.dot(up)).normalize();
                put(&mut f, 68, bf);
                put(&mut f, 80, up);
                for (j, wp) in wh.iter().enumerate() {
                    put(&mut f, 20 + j * 12, *wp);
                }
                data.extend_from_slice(&f);
                // front wheels along the travel: full counter-steer
                let s = fwd.cross(tangent).y.atan2(fwd.dot(tangent));
                steer.push(-s.to_degrees());
            }
            let base_off = data.len();
            for (i, st) in steer.iter().enumerate() {
                let tt = i as f32 / rate;
                let mut b = [0u8; BASE];
                let h = |b: &mut [u8; BASE], at: usize, x: f32| b[at..at + 2].copy_from_slice(&to_f16(x).to_le_bytes());
                h(&mut b, 0, 6400.0 + 300.0 * (tt * 3.1 + k as f32).sin());
                h(&mut b, 2, (st * 13.0).clamp(-480.0, 480.0));
                b[4] = 255;
                b[5] = 0;
                let roll = d.speed / d.tyre_radius;
                h(&mut b, 6, roll);
                h(&mut b, 8, roll);
                // the rears spin up: that is the smoke
                h(&mut b, 10, roll * 2.2);
                h(&mut b, 12, roll * 2.2);
                h(&mut b, 22, roll);
                h(&mut b, 24, 1.1);
                b[31] = 2;
                b[40..48].copy_from_slice(&(tt as f64).to_le_bytes());
                data.extend_from_slice(&b);
            }
            let mut guid = [0u8; 16];
            guid[0] = 0xd0;
            guid[1] = k as u8;
            cars.push(Car { guid, model: Some(model.to_string()), n_matrix: frames as u32, n_base: frames as u32, off, base_off });
        }
        Replay { data, explicit: true, version: 16, duration: d.seconds, model: model.to_string(), track: Some(track.to_string()), cars }
    }

    /// Cars of `models` (one each, cycling) driving `c.line` two abreast,
    /// as a replay. `ground(x, z, top)` is the highest surface under `top`
    /// (the line's own height + a little, so bridges overhead are skipped).
    pub fn convoy(track: &str, models: &[String], n: usize, c: &Convoy,
                  ground: &dyn Fn(f32, f32, f32) -> Option<f32>) -> Replay {
        let rate = 50.0f32 / 3.0;
        let frames = (c.seconds * rate).ceil() as usize + 2;
        // the recorded line is a coarse polyline: resampled every metre and
        // smoothed, or its kinks come through as twitches in the curvature
        // (body roll, the lane bias) and the cars jiggle
        let pts_v = {
            let src = &c.line;
            let n0 = src.len().max(2);
            let mut acc0 = vec![0.0f32; n0 + 1];
            for i in 0..n0 {
                acc0[i + 1] = acc0[i] + (src[(i + 1) % n0] - src[i]).length();
            }
            let tot = acc0[n0].max(1.0);
            let m = (tot.ceil() as usize).max(8);
            let mut v: Vec<Vec3> = (0..m).map(|j| {
                let s = j as f32 * tot / m as f32;
                let i = acc0.partition_point(|&a| a <= s).saturating_sub(1).min(n0 - 1);
                let seg = (acc0[i + 1] - acc0[i]).max(1e-4);
                src[i].lerp(src[(i + 1) % n0], (s - acc0[i]) / seg)
            }).collect();
            for _ in 0..4 {
                let prev = v.clone();
                for j in 0..m {
                    let mut sum = Vec3::ZERO;
                    for d in -3i32..=3 {
                        sum += prev[(j as i32 + d).rem_euclid(m as i32) as usize];
                    }
                    v[j] = sum / 7.0;
                }
            }
            v
        };
        // arc length along the closed line
        let pts = &pts_v;
        let np = pts.len().max(2);
        let mut acc = vec![0.0f32; np + 1];
        for i in 0..np {
            acc[i + 1] = acc[i] + (pts[(i + 1) % np] - pts[i]).length();
        }
        let total = acc[np].max(1.0);
        let at = |s: f32| -> Vec3 {
            let s = s.rem_euclid(total);
            let i = acc.partition_point(|&a| a <= s).saturating_sub(1).min(np - 1);
            let seg = (acc[i + 1] - acc[i]).max(1e-4);
            pts[i].lerp(pts[(i + 1) % np], (s - acc[i]) / seg)
        };
        let dir = |s: f32| -> Vec3 {
            let d = at(s + 2.0) - at(s - 2.0);
            Vec3::new(d.x, 0.0, d.z).normalize_or_zero()
        };
        let mut data = Vec::new();
        let mut cars = Vec::new();
        for k in 0..n {
            let model = &models[k % models.len().max(1)];
            let row = (k / 2) as f32;
            let side = if k % 2 == 0 { -1.0 } else { 1.0 };
            // two abreast, the outside car a little back, each a bit off line
            let s0 = c.start_s - row * c.gap - if k % 2 == 1 { c.gap * 0.4 } else { 0.0 } + 1.5 * jitter(k, 11);
            let lane = side * (1.25 + 0.35 * jitter(k, 12));
            let pace = c.speed * (1.0 + 0.012 * jitter(k, 13));
            let off = data.len();
            let mut steer = Vec::with_capacity(frames);
            let mut spin = Vec::with_capacity(frames);
            // per-wheel lift from body contact (kerbs), carried frame to frame
            let mut lift = [0.0f32; 4];
            let mut susp = [0.0f32; 4];
            // per frame: (scrape strength, along, lateral), (throttle, brake)
            let mut scrape: Vec<(f32, f32, f32)> = Vec::with_capacity(frames);
            let mut pedals: Vec<(f32, f32)> = Vec::with_capacity(frames);
            let mut thr = 1.0f32;
            for i in 0..frames {
                let tt = i as f32 / rate;
                // a little surge back and forth (+-0.8 m) on a common pace
                let s = s0 + pace * tt + 0.8 * (tt * 0.7 + 3.0 * jitter(k, 14)).sin();
                let axle = |s: f32| {
                    let d = dir(s);
                    let r = Vec3::Y.cross(d).normalize_or_zero();
                    // in a turn the pack drifts a little to the outside: the
                    // inside car rides the kerb rather than living on it
                    let tr = dir(s - 8.0).cross(dir(s + 8.0)).y / 16.0;
                    let out = -0.45 * (tr * 60.0).clamp(-1.0, 1.0);
                    at(s) + r * (lane + out + 0.25 * (tt * 0.45 + 2.0 * jitter(k, 15)).sin())
                };
                let fc = axle(s + c.wheelbase * 0.5);
                let rc = axle(s - c.wheelbase * 0.5);
                let fh = Vec3::new(fc.x - rc.x, 0.0, fc.z - rc.z);
                let fwd = fh.normalize_or_zero();
                let right = Vec3::Y.cross(fwd).normalize_or_zero();
                let top = at(s).y + 4.0;
                let wheel = |p: Vec3, sd: f32| {
                    let g = p + right * (sd * c.track_width * 0.5);
                    let y = ground(g.x, g.z, top).unwrap_or(p.y);
                    Vec3::new(g.x, y + c.tyre_radius, g.z)
                };
                // front-right, front-left, rear-right, rear-left
                let mut wh = [wheel(fc, 1.0), wheel(fc, -1.0), wheel(rc, 1.0), wheel(rc, -1.0)];
                // a little suspension: wheel heights follow the ground through
                // a spring, so a kerb step is a bump, not a one-frame jump
                for j in 0..4 {
                    if i > 0 {
                        susp[j] += (wh[j].y - susp[j]) * 0.55;
                        wh[j].y = susp[j];
                    } else {
                        susp[j] = wh[j].y;
                    }
                }
                // Body contact: the splitter corners, sills and diffuser
                // against the ground under them. Whatever would dig in lifts
                // its corner of the car (it rides up the kerb: pitch and
                // roll), and the lift bleeds off over ~0.2 s with a little
                // scrape chatter - instead of the nose passing through.
                let contact: [Vec3; 4] = wh.map(|w| w - Vec3::Y * c.tyre_radius);
                let o = (contact[0] + contact[1] + contact[2] + contact[3]) * 0.25;
                let fa = (contact[0] + contact[1]) * 0.5 - (contact[2] + contact[3]) * 0.5;
                let ra = (contact[0] + contact[2]) * 0.5 - (contact[1] + contact[3]) * 0.5;
                let slope_a = fa.y / Vec3::new(fa.x, 0.0, fa.z).length().max(0.1);
                let slope_l = ra.y / Vec3::new(ra.x, 0.0, ra.z).length().max(0.1);
                let half = c.wheelbase * 0.5;
                // (along, lateral, clearance, wheel weights FR FL RR RL)
                let pts: [(f32, f32, f32, [f32; 4]); 9] = [
                    (half + 0.95, 0.85, 0.07, [1.0, 0.0, 0.0, 0.0]),
                    (half + 1.0, 0.0, 0.07, [0.6, 0.6, 0.0, 0.0]),
                    (half + 0.95, -0.85, 0.07, [0.0, 1.0, 0.0, 0.0]),
                    (half * 0.5, 1.0, 0.10, [0.7, 0.0, 0.3, 0.0]),
                    (half * 0.5, -1.0, 0.10, [0.0, 0.7, 0.0, 0.3]),
                    (-half * 0.5, 1.0, 0.10, [0.3, 0.0, 0.7, 0.0]),
                    (-half * 0.5, -1.0, 0.10, [0.0, 0.3, 0.0, 0.7]),
                    (-half - 0.85, 0.8, 0.12, [0.0, 0.0, 1.0, 0.0]),
                    (-half - 0.85, -0.8, 0.12, [0.0, 0.0, 0.0, 1.0]),
                ];
                let mut want = [0.0f32; 4];
                let mut deepest = (0.0f32, 0.0f32, 0.0f32);
                for &(a, l, clr, wts) in &pts {
                    let p = o + fwd * a + right * l;
                    let Some(g) = ground(p.x, p.z, top) else { continue };
                    let floor = o.y + a * slope_a + l * slope_l + clr;
                    // the lift at a wheel raises this point by about that much
                    // times its lever (overhangs: a little more than 1)
                    let pen = g - floor;
                    if pen > deepest.0 {
                        deepest = (pen, a, l);
                    }
                    if pen > 0.0 {
                        let lever = (a.abs() / half.max(0.5)).max(1.0);
                        for j in 0..4 {
                            want[j] = want[j].max(pen / lever * wts[j] * 1.15);
                        }
                    }
                }
                for j in 0..4 {
                    let w = want[j].min(0.3);
                    // eases up quickly and settles slowly (⚠ was: snap up,
                    // fall 7 cm a frame, plus a scrape wobble - a sawtooth the
                    // eye reads as the cars jiggling)
                    lift[j] += (w - lift[j]) * if w > lift[j] { 0.7 } else { 0.25 };
                    wh[j].y += lift[j];
                }
                scrape.push((smoothstep(0.012, 0.05, deepest.0), deepest.1, deepest.2));
                // throttle: off into a tightening turn, part way through it,
                // full on the way out (the backfires come on the lift)
                let k_now = (dir(s - 8.0).cross(dir(s + 8.0)).y / 16.0).abs();
                let k_ahead = (dir(s + 12.0).cross(dir(s + 28.0)).y / 16.0).abs();
                let want_thr = if k_ahead > k_now + 0.004 { 0.0 } else if k_now > 0.012 { 0.45 } else { 1.0 };
                thr += (want_thr - thr) * 0.35;
                pedals.push((thr, if want_thr == 0.0 { 0.35 } else { 0.0 }));
                let body = (wh[0] + wh[1] + wh[2] + wh[3]) * 0.25 + Vec3::Y * 0.05;
                // the body on its wheels (the Karussell's banking), rolled
                // a little out of the turn
                let along = (wh[0] + wh[1]) * 0.5 - (wh[2] + wh[3]) * 0.5;
                let across = (wh[0] + wh[2]) * 0.5 - (wh[1] + wh[3]) * 0.5;
                let mut up = across.cross(along).normalize_or_zero();
                if up.y < 0.0 {
                    up = -up;
                }
                let d0 = dir(s - 8.0);
                let d1 = dir(s + 8.0);
                let turn = d0.cross(d1).y / 16.0; // signed curvature, 1/m
                let lat_g = pace * pace * turn.abs() / 9.81;
                let roll = (1.6 * lat_g.min(2.5)).to_radians() * turn.signum();
                let up = (up + right * roll.tan()).normalize();
                let bf = (along.normalize_or_zero() - up * along.normalize_or_zero().dot(up)).normalize_or_zero();
                let mut f = [0u8; FRAME];
                let put = |f: &mut [u8; FRAME], at: usize, v: Vec3| {
                    for (j, x) in [v.x, v.y, v.z].iter().enumerate() {
                        f[at + j * 4..at + j * 4 + 4].copy_from_slice(&x.to_le_bytes());
                    }
                };
                put(&mut f, 0, body);
                put(&mut f, 68, bf);
                put(&mut f, 80, up);
                for (j, wp) in wh.iter().enumerate() {
                    put(&mut f, 20 + j * 12, *wp);
                }
                data.extend_from_slice(&f);
                // front wheels along the line ahead
                let tf = dir(s + c.wheelbase * 0.5 + 1.0);
                let st = fwd.cross(tf).y.atan2(fwd.dot(tf));
                steer.push(-st.to_degrees());
                spin.push(pace / c.tyre_radius);
            }
            let base_off = data.len();
            for (i, st) in steer.iter().enumerate() {
                let tt = i as f32 / rate;
                let mut b = [0u8; BASE];
                let h = |b: &mut [u8; BASE], at: usize, x: f32| b[at..at + 2].copy_from_slice(&to_f16(x).to_le_bytes());
                let (th, br) = pedals[i];
                h(&mut b, 0, 5200.0 + 2300.0 * th + 250.0 * (tt * 1.3 + k as f32).sin());
                h(&mut b, 2, (st * 13.0).clamp(-480.0, 480.0));
                b[4] = (th * 255.0) as u8;
                b[5] = (br * 255.0) as u8;
                let (sc, sa, sl) = scrape[i];
                h(&mut b, 32, sc);
                h(&mut b, 34, sa);
                h(&mut b, 36, sl);
                for at in [6, 8, 10, 12, 22] {
                    h(&mut b, at, spin[i]);
                }
                h(&mut b, 24, 1.0);
                b[31] = 3;
                b[40..48].copy_from_slice(&(tt as f64).to_le_bytes());
                data.extend_from_slice(&b);
            }
            let mut guid = [0u8; 16];
            guid[0] = 0xc0;
            guid[1] = k as u8;
            cars.push(Car { guid, model: Some(model.clone()), n_matrix: frames as u32, n_base: frames as u32, off, base_off });
        }
        let model = models.first().cloned().unwrap_or_default();
        Replay { data, explicit: true, version: 16, duration: c.seconds, model, track: Some(track.to_string()), cars }
    }

    pub fn load(path: &Path) -> Result<Replay, String> {
        let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if data.len() < 64 {
            return Err("too short to be a replay".into());
        }
        let version = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let duration = f64::from_le_bytes(data[2..10].try_into().unwrap()) as f32;
        let n = u32::from_le_bytes(data[26..30].try_into().unwrap()) as usize;
        if 30 + n > data.len() {
            return Err("bad model-name length".into());
        }
        let model = String::from_utf8_lossy(&data[30..30 + n]).into_owned();

        let mut r = Replay { data, explicit: false, version, duration, model, track: None, cars: Vec::new() };
        let o = r.skip_metadata(30 + n);
        r.track = r.find_track(o);
        r.cars = r.read_cars(o)?;
        r.resolve_models();
        Ok(r)
    }

    /// The track folder from the session metadata, which holds paths like
    /// `content\tracks\laguna_seca\laguna_seca.scene`. Only the region before
    /// the car array is searched, so car-skin paths cannot be mistaken for it.
    fn find_track(&self, meta_end: usize) -> Option<String> {
        const PREFIX: &[u8] = br"content\tracks\";
        let head = &self.data[..meta_end.min(self.data.len())];
        let i = find(head, PREFIX, 0)? + PREFIX.len();
        // ⚠ Stop at the first byte that cannot be part of a folder name, not
        // at a separator: the first such path in the metadata is a bare
        // `content\tracks\laguna_seca` with no trailing slash, so scanning for
        // one runs off into the following protobuf bytes.
        let end = head[i..]
            .iter()
            .position(|b| !(b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'_'))
            .map_or(head.len(), |n| i + n);
        let name = &head[i..end];
        (!name.is_empty()).then(|| String::from_utf8_lossy(name).into_owned())
    }

    /// Walk the `[u32 len][protobuf]` metadata chunks and stop at the car
    /// array. The car array also begins with 123, so the two are told apart by
    /// whether a plausible protobuf tag follows the length.
    fn skip_metadata(&self, mut o: usize) -> usize {
        const TAGS: [u8; 6] = [0x08, 0x12, 0x1a, 0x22, 0x2a, 0x32];
        let d = &self.data;
        while o + 8 <= d.len() {
            let v = u32::from_le_bytes(d[o..o + 4].try_into().unwrap()) as usize;
            let (len, body) = if v == 123 {
                let l = u32::from_le_bytes(d[o + 4..o + 8].try_into().unwrap()) as usize;
                if o + 8 + l > d.len() || !TAGS.contains(&d[o + 8]) {
                    return o;
                }
                (l, o + 8)
            } else {
                if o + 4 + v > d.len() {
                    return o;
                }
                (v, o + 4)
            };
            o = body + len;
        }
        o
    }

    fn read_cars(&self, o: usize) -> Result<Vec<Car>, String> {
        let d = &self.data;
        if o + 6 > d.len() {
            return Err("no car array".into());
        }
        let count = u16::from_le_bytes(d[o + 4..o + 6].try_into().unwrap()) as usize;
        let mut q = o + 6;
        let mut cars = Vec::with_capacity(count);
        for _ in 0..count {
            if q + 37 > d.len() || d[q..q + 16] != d[q + 16..q + 32] {
                break; // lost sync - stop rather than emit garbage cars
            }
            let mut guid = [0u8; 16];
            guid.copy_from_slice(&d[q..q + 16]);
            q += 32 + 5;
            let n_matrix = u32::from_le_bytes(d[q..q + 4].try_into().unwrap());
            q += 4;
            let off = q;
            q += n_matrix as usize * FRAME;
            if q + 4 > d.len() {
                break;
            }
            let n_base = u32::from_le_bytes(d[q..q + 4].try_into().unwrap());
            q += 4;
            // ⚠ The base stream is nBase fixed 48-byte records followed by a
            // ~23 KB trailing section of something else. The record count is
            // exact: the timestamp at +40 stays monotonic for precisely nBase
            // records and breaks at nBase+1, in every car checked.
            let base_off = q;
            cars.push(Car { guid, model: None, n_matrix, n_base, off, base_off });
            match self.next_car(q) {
                Some(next) => q = next,
                None => break,
            }
        }
        if cars.is_empty() {
            return Err("no cars parsed".into());
        }
        Ok(cars)
    }

    /// Find the next car header by its doubled guid.
    ///
    /// ⚠ Base frames are variable-length (~66.7 bytes each on average), so
    /// there is no arithmetic that lands on the next car - it has to be a
    /// scan. A doubled 16-byte run occurs by chance inside base-frame data
    /// often enough to matter (it derailed 3 of 73 local replays), so a
    /// candidate only counts once its frame count and stream look sane.
    fn next_car(&self, mut x: usize) -> Option<usize> {
        let d = &self.data;
        let end = d.len().checked_sub(32)?;
        while x < end {
            // Cheap reject first; the full 16-byte compare is rare.
            if d[x] == d[x + 16] && d[x..x + 16] == d[x + 16..x + 32] && d[x..x + 16] != [0u8; 16] {
                if self.plausible(x) {
                    return Some(x);
                }
            }
            x += 1;
        }
        None
    }

    fn plausible(&self, x: usize) -> bool {
        let d = &self.data;
        let q = x + 32 + 5;
        if q + 4 > d.len() {
            return false;
        }
        let n = u32::from_le_bytes(d[q..q + 4].try_into().unwrap()) as usize;
        if n == 0 || n > 5_000_000 || q + 4 + n * FRAME > d.len() {
            return false;
        }
        for k in [0, n / 2, n - 1] {
            let o = q + 4 + k * FRAME;
            for c in 0..3 {
                let v = f32::from_le_bytes(d[o + c * 4..o + c * 4 + 4].try_into().unwrap());
                if !v.is_finite() || v.abs() > 1.0e6 {
                    return false;
                }
            }
        }
        true
    }

    /// Attach a car model to each car from the trailing customization data.
    ///
    /// Leaves `model` as None rather than guessing when a block does not
    /// validate - a confidently wrong car is worse than a missing one.
    fn resolve_models(&mut self) {
        let last = match self.cars.last() {
            Some(c) => c,
            None => return,
        };
        let tail_start = last.off + last.n_matrix as usize * FRAME;
        if tail_start >= self.data.len() {
            return;
        }
        let mut found = Vec::with_capacity(self.cars.len());
        for car in &self.cars {
            let mut needle = GUID_FIELD.to_vec();
            needle.extend_from_slice(car.dashed().as_bytes());
            found.push(match find(&self.data[tail_start..], &needle, 0) {
                Some(i) => {
                    let j = i + needle.len();
                    let tail = &self.data[tail_start..];
                    if tail.len() >= j + 2 && &tail[j..j + 2] == PGUID_TAG {
                        car_path(tail, j, (j + LOOKAHEAD).min(tail.len()))
                    } else {
                        None // a chance string hit, not a real record
                    }
                }
                None => None,
            });
        }
        for (car, m) in self.cars.iter_mut().zip(found) {
            car.model = m;
        }
    }

    /// Raw pose at a matrix frame index, clamped to the car's range.
    pub fn pose(&self, car: usize, frame: usize) -> Pose {
        let c = &self.cars[car];
        let f = frame.min(c.n_matrix.saturating_sub(1) as usize);
        let o = c.off + f * FRAME;
        let v3 = |at: usize| {
            let g = |k: usize| {
                f32::from_le_bytes(self.data[at + k * 4..at + k * 4 + 4].try_into().unwrap())
            };
            Vec3::new(g(0), g(1), g(2))
        };
        let pos = v3(o);
        let wheels = [v3(o + 20), v3(o + 32), v3(o + 44), v3(o + 56)];
        let front = (wheels[0] + wheels[1]) * 0.5;
        let rear = (wheels[2] + wheels[3]) * 0.5;
        let forward = (front - rear).normalize_or_zero();
        // ⚠ Wheel 0 is the RIGHT of its axle, not the left, so this is
        // `w0 - w1`. Getting it backwards inverts `up` and rolls every car
        // 180 degrees - they render upside down. It cannot be caught by
        // checking forward against direction of travel, because swapping left
        // for right leaves front-mid minus rear-mid untouched. The check that
        // does catch it: cars on a track must have up.y > 0, and with the
        // wheels the wrong way round it measures -0.999.
        let across = (wheels[0] - wheels[1]).normalize_or_zero();
        let up = forward.cross(across).normalize_or_zero();
        // `across` is not exactly perpendicular to forward once the car rolls,
        // so rebuild it from the other two.
        let right = up.cross(forward);
        if self.explicit {
            let (f, u) = (v3(o + 68), v3(o + 80));
            if f.length_squared() > 0.5 && u.length_squared() > 0.5 {
                return Pose { pos, forward: f, right: u.cross(f), up: u, wheels };
            }
        }
        Pose { pos, forward, right, up, wheels }
    }

    /// Pose at a time in seconds, interpolated between frames.
    ///
    /// ⚠ Cars have different frame counts because drivers join and leave, and
    /// nothing on disk says when a car's stream STARTS. Every car is assumed to
    /// begin at t = 0 and simply end early. The evidence for that is a race
    /// replay where all 24 cars sit correctly on the starting grid at frame 0;
    /// it has not been confirmed for drop-in lobbies, where a late joiner may
    /// well be offset. Returns None once a car's stream has run out.
    pub fn sample(&self, car: usize, t: f32) -> Option<Pose> {
        let c = &self.cars[car];
        if c.n_matrix == 0 {
            return None;
        }
        let rate = self.rate();
        let x = (t * rate).max(0.0);
        let i = x.floor() as usize;
        if i + 1 >= c.n_matrix as usize {
            return if i < c.n_matrix as usize { Some(self.pose(car, i)) } else { None };
        }
        let (a, b) = (self.pose(car, i), self.pose(car, i + 1));
        let f = x - i as f32;
        Some(Pose {
            pos: a.pos.lerp(b.pos, f),
            forward: a.forward.lerp(b.forward, f).normalize_or_zero(),
            right: a.right.lerp(b.right, f).normalize_or_zero(),
            up: a.up.lerp(b.up, f).normalize_or_zero(),
            wheels: [
                a.wheels[0].lerp(b.wheels[0], f),
                a.wheels[1].lerp(b.wheels[1], f),
                a.wheels[2].lerp(b.wheels[2], f),
                a.wheels[3].lerp(b.wheels[3], f),
            ],
        })
    }

    /// Ground speed in m/s, from the motion itself.
    ///
    /// Preferred over the telemetry's own speed field, which only the local
    /// player's car carries.
    pub fn speed(&self, car: usize, t: f32) -> f32 {
        let c = &self.cars[car];
        let rate = self.rate();
        let i = (t * rate) as usize;
        if i < 1 || i + 2 >= c.n_matrix as usize {
            return 0.0;
        }
        self.pose(car, i - 1).pos.distance(self.pose(car, i + 1).pos) * rate * 0.5
    }

    /// Telemetry for a car at a time in seconds.
    ///
    /// Base frames carry their own timestamps and arrive at an irregular rate
    /// (~12-16 Hz), so this searches by time rather than indexing. No
    /// interpolation: gear and the pedals are not meaningful blended, and at
    /// this rate nothing else visibly steps.
    pub fn telemetry(&self, car: usize, t: f32) -> Option<Telemetry> {
        let c = &self.cars[car];
        let n = c.n_base as usize;
        if n == 0 || c.base_off + n * BASE > self.data.len() {
            return None;
        }
        let stamp = |i: usize| -> f32 {
            let o = c.base_off + i * BASE + 40;
            f64::from_le_bytes(self.data[o..o + 8].try_into().unwrap()) as f32
        };
        let (mut lo, mut hi) = (0usize, n - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if stamp(mid) < t { lo = mid + 1; } else { hi = mid; }
        }
        let b = &self.data[c.base_off + lo * BASE..c.base_off + (lo + 1) * BASE];
        Some(Telemetry {
            rpm: f16(b, 0),
            steer_deg: f16(b, 2),
            throttle: b[4] as f32 / 255.0,
            brake: b[5] as f32 / 255.0,
            wheel: [f16(b, 6), f16(b, 8), f16(b, 10), f16(b, 12)],
            speed_rads: f16(b, 22),
            lat_g: f16(b, 24),
            gear: b[31],
            time: stamp(lo),
            scrape: if self.explicit { f16(b, 32) } else { 0.0 },
            scrape_at: if self.explicit { [f16(b, 34), f16(b, 36)] } else { [0.0; 2] },
        })
    }

    /// Frames per second. A constant 60 ms tick.
    ///
    /// Deriving this per file as `frames / duration` looks reasonable and is
    /// wrong: it agrees (16.67 Hz) only on short sessions, and on long ones
    /// where the buffer has wrapped it reports nonsense as low as 4.9 Hz.
    /// Checked across 833 cars in all 73 local replays by turning each car's
    /// travelled distance into an average speed: the constant gives a median
    /// 168 km/h with 70 cars outside a plausible 40-300 km/h band, while
    /// frames/duration gives a median 109 km/h and 163 outside it. Racing cars
    /// average 168 km/h; they do not average 109.
    pub fn rate(&self) -> f32 {
        50.0 / 3.0
    }

    /// Seconds of motion actually captured, which is what playback should span.
    pub fn captured(&self) -> f32 {
        let longest = self.cars.iter().map(|c| c.n_matrix).max().unwrap_or(0);
        longest as f32 / self.rate()
    }

    /// Cars that never leave the origin are unused session slots, not drivers.
    pub fn is_placeholder(&self, car: usize) -> bool {
        let c = &self.cars[car];
        if c.n_matrix == 0 {
            return true;
        }
        [0, c.n_matrix as usize / 2, c.n_matrix as usize - 1]
            .iter()
            .all(|&f| self.pose(car, f).pos.length_squared() < 1.0)
    }
}

/// `haystack.find(needle)` starting at `from`.
fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    let first = needle[0];
    (from..=haystack.len() - needle.len())
        .find(|&i| haystack[i] == first && &haystack[i..i + needle.len()] == needle)
}

/// The car folder from the first `content\cars\<model>\` path in a range.
fn car_path(buf: &[u8], from: usize, to: usize) -> Option<String> {
    let i = find(&buf[..to], CAR_PREFIX, from)? + CAR_PREFIX.len();
    let end = buf[i..to].iter().position(|&b| b == b'\\')? + i;
    let name = &buf[i..end];
    if name.is_empty() || !name.iter().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'_') {
        return None;
    }
    Some(String::from_utf8_lossy(name).into_owned())
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
