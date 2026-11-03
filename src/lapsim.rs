//! `--lapsim <track> --car <id> [--layout <l>]`: an ideal lap time from the
//! car's own physics data, quasi-steady-state.
//!
//! Every metre of the racing line gets a cornering limit (lateral grip of all
//! four tyres, with the aero load, against v^2 x curvature), then a forward
//! pass (engine force through the best gear, capped by rear traction on the
//! friction circle, minus drag, rolling resistance and the slope) and a
//! backward pass (braking on all four tyres, plus drag and the slope). Two
//! laps are run so the line's start speed is the flying one.
//!
//! From the game's data: mass, CG split, wheelbase, the torque curve (+
//! turbos, the engine map's power multiplier), gears, final drive, the tyre
//! radius, each axle's tyre model (Dy0 / Dx0 at Fz0 with load sensitivity),
//! every wing (.wing: chord x span, AOA lift / drag curves at the setup's
//! wing angles, ground-height multipliers at the setup's ride heights, gains)
//! and the layout's racing line with its elevation.
//!
//! Verified against the game's code (Ghidra, ksTyreModel / the Car aero):
//! * tyre peak friction mu(Fz) = K Fz^ls / Fz with K = D0 Fz0 / Fz0^ls
//!   (carphys::TyreModel), Dx0 = 0 -> 1.2 Dy0;
//! * wing force 0.5 rho v^2 x chord span x CL(angle) x cl_gain x GH(height);
//! * body maps: every DownForceData point gives 0.5 map(front h, rear h)
//!   (+ 0.5 x the angle terms, unset on the GT3s), front pair on the front
//!   map, rear pair on the rear one - per axle the map value itself, in m^2.
//!
//! ⚠ Not the game's simulation: no tyre temperature / pressure, no transient
//! slip, no lateral load transfer beyond a flat 5 %, no shift time, banking
//! is not added to the grip, the driver is perfect. The number is what this
//! data allows, i.e. a lower bound.

use crate::carphys::{read_curve, Physics};
use crate::proto::{fields, text};
use glam::Vec3;

struct Wing {
    name: String,
    /// lift (downforce) and drag per (m/s)^2, at the setup's angle and height
    k_l: f32,
    k_d: f32,
    /// share of this wing's load on the front axle
    front: f32,
}

/// Math.proto Surface3dData: 1 sizeX, 2 sizeY, 3/4 min/max X, 5/6 min/max
/// Y, 7 values (X fastest). Bilinear, clamped at the edges.
struct Surface {
    nx: usize,
    ny: usize,
    x: [f32; 2],
    y: [f32; 2],
    v: Vec<f32>,
}

impl Surface {
    fn read(d: &[u8]) -> Option<Surface> {
        let mut s = Surface { nx: 0, ny: 0, x: [0.0; 2], y: [0.0; 2], v: Vec::new() };
        fields(d, 0, d.len(), |f, v| match f {
            1 => s.nx = v.f32().max(0.0) as usize,
            2 => s.ny = v.f32().max(0.0) as usize,
            3 => s.x[0] = v.f32(),
            4 => s.x[1] = v.f32(),
            5 => s.y[0] = v.f32(),
            6 => s.y[1] = v.f32(),
            7 => if let Some((a, b)) = v.range() {
                s.v = d[a..b].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            },
            _ => {}
        });
        (s.nx >= 2 && s.ny >= 2 && s.v.len() >= s.nx * s.ny).then_some(s)
    }

    fn at(&self, x: f32, y: f32) -> f32 {
        let fx = ((x - self.x[0]) / (self.x[1] - self.x[0]).max(1e-6)).clamp(0.0, 1.0) * (self.nx - 1) as f32;
        let fy = ((y - self.y[0]) / (self.y[1] - self.y[0]).max(1e-6)).clamp(0.0, 1.0) * (self.ny - 1) as f32;
        let (i, j) = ((fx as usize).min(self.nx - 2), (fy as usize).min(self.ny - 2));
        let (tx, ty) = (fx - i as f32, fy - j as f32);
        let g = |a: usize, b: usize| self.v[b * self.nx + a];
        let lo = g(i, j) + (g(i + 1, j) - g(i, j)) * tx;
        let hi = g(i, j + 1) + (g(i + 1, j + 1) - g(i, j + 1)) * tx;
        lo + (hi - lo) * ty
    }
}

fn lut(c: &[(f32, f32)], x: f32) -> f32 {
    if c.is_empty() {
        return 1.0;
    }
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

fn sub_range(d: &[u8], a: usize, b: usize, want: u32) -> Option<(usize, usize)> {
    let mut hit = None;
    fields(d, a, b, |f, v| {
        if f == want && hit.is_none() {
            hit = v.range();
        }
    });
    hit
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

/// (front wing angle, rear wing angle, front ride height m, rear ride height m)
/// from the setup the car loaded with (stock, or the player's: carphys)
fn setup(vfs: &mut crate::load::Vfs, p: &Physics) -> (Option<f32>, Option<f32>, f32, f32) {
    if !p.setup_source.is_empty() {
        let out = (p.wing[0], p.wing[1], p.ride_h[0].unwrap_or(0.05), p.ride_h[1].unwrap_or(0.05));
        let src = if p.user_setup.is_empty() { &p.setup_source } else { &p.user_setup };
        println!("lapsim: setup {src}: wings {:?} / {:?} deg, ride {:.0} / {:.0} mm", out.0, out.1, out.2 * 1e3, out.3 * 1e3);
        return out;
    }
    // CarData f19 stock_setup, else any dry setup in data/
    let cd = vfs.read(&p.source);
    let mut stock = String::new();
    fields(&cd, 0, cd.len(), |f, v| {
        if let (19, Some(r)) = (f, v.range()) {
            stock = text(&cd, r);
        }
    });
    let mut keys = vfs.list_rec("data", "carsetup");
    keys.retain(|k| !k.contains("wet"));
    let found = (!stock.is_empty()).then(|| vfs.find(&stock, None)).flatten();
    let Some(k) = found.or_else(|| keys.first().cloned()) else { return (None, None, 0.05, 0.05) };
    let d = vfs.read(&k);
    let mut out = (None, None, 0.05, 0.05);
    fields(&d, 0, d.len(), |f, v| {
        let (6, Some((a, b))) = (f, v.range()) else { return };
        out.0 = f32_at(&d, a, b, 4);
        out.1 = f32_at(&d, a, b, 5);
        if let Some(h) = f32_at(&d, a, b, 2) {
            out.2 = h / 1000.0;
        }
        if let Some(h) = f32_at(&d, a, b, 3) {
            out.3 = h / 1000.0;
        }
    });
    println!("lapsim: setup {k}: wings {:?} / {:?} deg, ride {:.0} / {:.0} mm", out.0, out.1, out.2 * 1e3, out.3 * 1e3);
    out
}

/// The car's aero as (front lift, rear lift, drag) per (m/s)^2, N, at the
/// stock setup: every wing plus the body maps (what `--lapsim` uses).
pub fn aero(vfs: &mut crate::load::Vfs, p: &Physics) -> [f32; 3] {
    let mut k = [0.0f32; 3];
    for w in wings(vfs, p) {
        k[0] += w.k_l * w.front;
        k[1] += w.k_l * (1.0 - w.front);
        k[2] += w.k_d;
    }
    k
}

fn wings(vfs: &mut crate::load::Vfs, p: &Physics) -> Vec<Wing> {
    const RHO: f32 = 1.22;
    let (fa, ra, fh, rh) = setup(vfs, p);
    let mut out = Vec::new();
    let mut curve = |vfs: &mut crate::load::Vfs, path: &str| -> Vec<(f32, f32)> {
        if path.is_empty() {
            return Vec::new();
        }
        vfs.find(path, None).map(|k| read_curve(&vfs.read(&k))).unwrap_or_default()
    };
    // CG -> rear axle distance: the front carries cg_front of the weight
    let to_rear = p.wheelbase * p.cg_front;
    // the wings: data/*.wing (F2004), or CarData.aero's inline wings (f3,
    // GT cars - their wings_path f10 lists the same ones again)
    let mut blobs: Vec<Vec<u8>> = vfs.list("data", "wing").iter().map(|k| vfs.read(k)).collect();
    let had_files = !blobs.is_empty();
    let cd_bytes = vfs.read(&p.source);
    let mut maps: [String; 3] = Default::default();
    // DownForceData f1 position (z) and f2 cl_gain, in order: the first two
    // are the front map's points, the rest the rear's
    let mut points: Vec<(f32, f32)> = Vec::new();
    fields(&cd_bytes, 0, cd_bytes.len(), |f, v| {
        let (12, Some((a, b))) = (f, v.range()) else { return };
        fields(&cd_bytes, a, b, |g, x| match (g, x.range()) {
            (3, Some((c, e))) if !had_files => blobs.push(cd_bytes[c..e].to_vec()),
            (5, Some((c, e))) => {
                let z = sub_range(&cd_bytes, c, e, 1).and_then(|(i, j)| f32_at(&cd_bytes, i, j, 3)).unwrap_or(f32::NAN);
                let gain = f32_at(&cd_bytes, c, e, 2).unwrap_or(1.0);
                points.push((z, gain));
            }
            (7, Some(r)) => maps[0] = text(&cd_bytes, r),
            (8, Some(r)) => maps[1] = text(&cd_bytes, r),
            (9, Some(r)) => maps[2] = text(&cd_bytes, r),
            _ => {}
        });
    });
    // Body aero (GT cars): front lift, rear lift and drag maps over (front,
    // rear) ride height in mm, in m^2 per axle (F = 1/2 rho v^2 map): the
    // game sums 0.5 x map over the axle's two DownForceData points.
    let mut body = [0.0f32; 3];
    for (j, path) in maps.iter().enumerate() {
        if path.is_empty() {
            continue;
        }
        let Some(k) = vfs.find(path, None) else { continue };
        if let Some(s) = Surface::read(&vfs.read(&k)) {
            body[j] = s.at(fh * 1000.0, rh * 1000.0);
        }
    }
    if body.iter().any(|&b| b != 0.0) {
        println!("lapsim: body maps at {:.0} / {:.0} mm: front lift {:.3}, rear lift {:.3}, drag {:.3} m^2",
                 fh * 1e3, rh * 1e3, body[0], body[1], body[2]);
        let q = 0.5 * RHO;
        // where each map acts: its points' mean position (CG-relative z),
        // as a share of the front axle - the axle line when there are none
        let share = |pts: &[(f32, f32)], axle: f32| -> f32 {
            let pts: Vec<_> = pts.iter().filter(|p| p.0.is_finite()).collect();
            if pts.is_empty() {
                return axle;
            }
            let z = pts.iter().map(|p| p.0).sum::<f32>() / pts.len() as f32;
            ((z + to_rear) / p.wheelbase.max(0.5)).clamp(-0.5, 1.5)
        };
        let (pf, pr) = points.split_at(points.len().min(2));
        let (sf, sr) = (share(pf, 1.0), share(pr, 0.0));
        println!("lapsim: downforce points {points:?}: front map at {:.0}% front, rear map at {:.0}% front", sf * 100.0, sr * 100.0);
        out.push(Wing { name: "BODY_MAP_F".into(), k_l: q * body[0], k_d: q * body[2], front: sf });
        out.push(Wing { name: "BODY_MAP_R".into(), k_l: q * body[1], k_d: 0.0, front: sr });
    }
    for d in blobs {
        let (mut vertical, mut name) = (false, String::new());
        let (mut chord, mut span, mut cd_gain, mut cl_gain, mut angle) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut z, mut cl, mut cd, mut ghl, mut ghd) = (0.0f32, String::new(), String::new(), String::new(), String::new());
        fields(&d, 0, d.len(), |f, v| match f {
            1 => vertical = v.f32() != 0.0 || matches!(v, crate::proto::Val::Varint(1)),
            2 => name = text(&d, v.range().unwrap_or((0, 0))),
            3 => chord = v.f32(),
            4 => span = v.f32(),
            6 => if let Some((a, b)) = v.range() { z = f32_at(&d, a, b, 3).unwrap_or(0.0) },
            7 => cl = text(&d, v.range().unwrap_or((0, 0))),
            8 => cd = text(&d, v.range().unwrap_or((0, 0))),
            9 => ghl = text(&d, v.range().unwrap_or((0, 0))),
            10 => ghd = text(&d, v.range().unwrap_or((0, 0))),
            11 => cd_gain = v.f32(),
            12 => cl_gain = v.f32(),
            13 => angle = v.f32(),
            _ => {}
        });
        if vertical {
            continue;
        }
        let up = name.to_uppercase();
        let a = if up == "FRONT" { fa.unwrap_or(angle) } else if up == "REAR" { ra.unwrap_or(angle) } else { angle };
        let h = if up.starts_with("FRONT") || up.ends_with("_F") { fh } else { rh };
        let (c_l, c_d) = (curve(vfs, &cl), curve(vfs, &cd));
        let (g_l, g_d) = (curve(vfs, &ghl), curve(vfs, &ghd));
        let area = chord * span;
        let lift = 0.5 * RHO * area * lut(&c_l, a) * cl_gain * if g_l.is_empty() { 1.0 } else { lut(&g_l, h) };
        let drag = 0.5 * RHO * area * lut(&c_d, a) * cd_gain * if g_d.is_empty() { 1.0 } else { lut(&g_d, h) };
        let front = ((z + to_rear) / p.wheelbase.max(0.5)).clamp(-0.5, 1.5);
        println!("lapsim: wing {name:12} {chord:.2} x {span:.2} m at {a:5.1} deg, h {:.0} mm: downforce {:6.1} N, drag {:5.1} N at 100 km/h, {:.0}% front",
                 h * 1e3, lift * 771.6, drag * 771.6, front * 100.0);
        out.push(Wing { name, k_l: lift, k_d: drag, front });
    }
    out
}

pub fn run(pkg: &std::path::Path, base: Option<&std::path::Path>, track: &str, layout: Option<&str>, car: &str) {
    let mut vfs = match crate::load::package_vfs(pkg, car, base) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("lapsim: {e}");
            return;
        }
    };
    let Some(p) = Physics::read(&mut vfs) else {
        eprintln!("lapsim: no physics data for {car}");
        return;
    };
    let w = wings(&mut vfs, &p);
    if w.is_empty() {
        eprintln!("lapsim: ⚠ {car}: no aero data found - lap WITHOUT aero (no downforce, no drag)");
    }
    let (k_l, k_d): (f32, f32) = (w.iter().map(|x| x.k_l).sum(), w.iter().map(|x| x.k_d).sum());
    let k_lf: f32 = w.iter().map(|x| x.k_l * x.front).sum();
    let m = p.mass;
    let g = 9.81;
    let r = p.tyre_radius.unwrap_or(0.33);
    let (Some(tf), Some(tr)) = (p.tyres[0], p.tyres[1]) else {
        eprintln!("lapsim: {car} has no tyre model");
        return;
    };
    let cgf = p.cg_front;
    // ---- the line, every metre, smoothed ----
    let src = crate::trackdrive::racing_line_layout(pkg, track, layout);
    if src.len() < 16 {
        eprintln!("lapsim: no racing line for {track} {layout:?}");
        return;
    }
    let n0 = src.len();
    let mut acc0 = vec![0.0f32; n0 + 1];
    for i in 0..n0 {
        acc0[i + 1] = acc0[i] + (src[(i + 1) % n0] - src[i]).length();
    }
    let total = acc0[n0];
    let n = total.round() as usize;
    let ds = total / n as f32;
    let mut pts: Vec<Vec3> = (0..n).map(|j| {
        let s = j as f32 * ds;
        let i = acc0.partition_point(|&a| a <= s).saturating_sub(1).min(n0 - 1);
        let seg = (acc0[i + 1] - acc0[i]).max(1e-4);
        src[i].lerp(src[(i + 1) % n0], (s - acc0[i]) / seg)
    }).collect();
    for _ in 0..4 {
        let prev = pts.clone();
        for j in 0..n {
            let mut sum = Vec3::ZERO;
            for d in -3i32..=3 {
                sum += prev[(j as i32 + d).rem_euclid(n as i32) as usize];
            }
            pts[j] = sum / 7.0;
        }
    }
    let at = |j: i32| pts[j.rem_euclid(n as i32) as usize];
    let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z).normalize_or_zero();
    let curv: Vec<f32> = (0..n as i32).map(|j| {
        let (d0, d1) = (flat(at(j) - at(j - 6)), flat(at(j + 6) - at(j)));
        (d0.cross(d1).y.abs() / (6.0 * ds)).min(0.5)
    }).collect();
    let slope: Vec<f32> = (0..n as i32).map(|j| ((at(j + 5).y - at(j - 5).y) / (10.0 * ds)).clamp(-0.3, 0.3)).collect();

    // ---- the car ----
    let fz = |v: f32| {
        let df = k_l * v * v;
        let df_f = k_lf * v * v;
        ((m * g * cgf + df_f) * 0.5, (m * g * (1.0 - cgf) + df - df_f) * 0.5)
    };
    // lateral load transfer and the slip a real lap runs at: a flat 5 %
    let lat_max = |v: f32| {
        let (f, rr) = fz(v);
        0.95 * 2.0 * (tf.mu_y(f) * f + tr.mu_y(rr) * rr)
    };
    let eta = 0.92;
    let drive = |v: f32| -> f32 {
        let wheel_rpm = v / r * 60.0 / std::f32::consts::TAU;
        let mut best = 0.0f32;
        for &gr in &p.gears {
            let rpm = wheel_rpm * gr * p.final_drive;
            if rpm > p.limiter_rpm {
                continue;
            }
            let t = p.engine_torque(rpm.max(p.idle_rpm.max(1000.0)), 1.0).unwrap_or(0.0);
            best = best.max(t * gr * p.final_drive * eta / r);
        }
        best
    };
    let driven_all = matches!(p.traction, 2 | 3 | 4);
    let crr = 0.012;
    // cornering limit
    let vmax: Vec<f32> = curv.iter().map(|&k| {
        if k < 1e-5 {
            return 120.0;
        }
        let mut v = 60.0f32;
        for _ in 0..40 {
            v = (lat_max(v) / (m * k)).sqrt().min(120.0);
        }
        v
    }).collect();
    let mut v = vmax.clone();
    // forward: two laps round, so the start is the flying speed
    let mut cur = 20.0f32;
    for pass in 0..2 * n {
        let i = pass % n;
        cur = cur.min(vmax[i]);
        let ay = cur * cur * curv[i];
        let room = (1.0 - (ay / (lat_max(cur) / m)).powi(2)).max(0.0).sqrt();
        let (f, rr) = fz(cur);
        let trac = if driven_all { 2.0 * (tf.mu_x(f) * f + tr.mu_x(rr) * rr) } else { 2.0 * tr.mu_x(rr) * rr } * room;
        let a = (drive(cur).min(trac) - k_d * cur * cur - m * g * (slope[i] + crr)) / m;
        let next = (cur * cur + 2.0 * a * ds).max(1.0).sqrt();
        if pass >= n {
            v[i] = cur;
        }
        cur = next;
    }
    // backward: braking into each limit
    let mut cur = v[0];
    for pass in 0..2 * n {
        let i = (2 * n - 1 - pass) % n;
        let ay = cur * cur * curv[i];
        let room = (1.0 - (ay / (lat_max(cur) / m)).powi(2)).max(0.0).sqrt();
        let (f, rr) = fz(cur);
        let brake = 2.0 * (tf.mu_x(f) * f + tr.mu_x(rr) * rr) * room;
        let a = (brake + k_d * cur * cur + m * g * (slope[i] + crr)) / m;
        let prev = (cur * cur + 2.0 * a.max(0.0) * ds).sqrt();
        v[i] = v[i].min(prev);
        cur = v[i];
    }
    let mut t = 0.0f32;
    for i in 0..n {
        t += ds / ((v[i] + v[(i + 1) % n]) * 0.5).max(0.5);
    }
    let top = v.iter().cloned().fold(0.0, f32::max);
    let low = v.iter().cloned().fold(f32::MAX, f32::min);
    let pk = p.peak_torque().unwrap_or((0.0, 0.0));
    println!("lapsim: {car}: {m:.0} kg, {:.0}% front, {} gears x {:.2}, tyre r {r:.3} m, peak {:.0} N m @ {:.0}, limiter {:.0}",
             cgf * 100.0, p.gears.len(), p.final_drive, pk.1, pk.0, p.limiter_rpm);
    println!("lapsim: tyres F dy0 {:.2} dx0 {:.2} fz0 {:.0} / R dy0 {:.2} dx0 {:.2} fz0 {:.0}", tf.dy0, tf.dx0, tf.fz0, tr.dy0, tr.dx0, tr.fz0);
    println!("lapsim: aero at 200 km/h: downforce {:.0} N ({:.0}% front), drag {:.0} N", k_l * 3086.4, k_lf / k_l.max(1e-6) * 100.0, k_d * 3086.4);
    println!("lapsim: {track} {}: {:.0} m, ideal lap {}:{:06.3} ({:.1} km/h average), top {:.0} km/h, slowest {:.0} km/h",
             layout.unwrap_or("gp"), total, (t / 60.0).floor() as i32, t % 60.0, total / t * 3.6, top * 3.6, low * 3.6);
    if let Ok(dump) = std::env::var("EVOVIEW_LAPSIM_DUMP") {
        let mut s = String::from("s_m,x,z,y,curv,v_kmh\n");
        for i in 0..n {
            s += &format!("{:.0},{:.1},{:.1},{:.1},{:.5},{:.1}\n", i as f32 * ds, pts[i].x, pts[i].z, pts[i].y, curv[i], v[i] * 3.6);
        }
        let _ = std::fs::write(&dump, s);
    }
}
