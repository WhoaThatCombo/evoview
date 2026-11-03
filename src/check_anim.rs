//! The animation / suspension half of `--check` (check.rs runs it with the
//! effects half, check_fx.rs, into one report): exercise a car without the
//! game and say what is broken.
//!
//! Everything here is CPU-side - the rig is posed exactly as drive mode poses
//! it, and the skin matrices are applied to the real vertex data - so a check
//! needs no GPU unless `--sheet` asks for key-pose pictures.
//!
//! Sweeps: steer lock to lock, every corner from full droop to full bump (and
//! bump + lock together), pedals, every gear, every bound clip start to end,
//! every needle across its range. Findings are `{severity, code, msg}`;
//! severity `error` means "visibly wrong in game", `warn` "probably wrong",
//! `info` "worth knowing". Thresholds were set so the stock fleet (the known
//! good baseline) comes out clean.

use crate::asset::Car;
use crate::carrig::{Gauge, Inputs, Kind, Rig};
use crate::load::Skeleton;
use glam::{Mat4, Vec3};
use serde_json::{json, Value};



struct Report {
    issues: Vec<Value>,
}

impl Report {
    fn add(&mut self, sev: &str, code: &str, msg: String) {
        self.issues.push(json!({"severity": sev, "code": code, "msg": msg}));
    }
}

fn finite(m: &[[f32; 16]]) -> bool {
    m.iter().all(|x| x.iter().all(|v| v.is_finite()))
}

/// Vertex data a check needs: car-space position, normal and the one bone
/// that dominates each vertex.
struct MeshPts {
    name: String,
    pos: Vec<Vec3>,
    nrm: Vec<Vec3>,
    bone: Vec<usize>,
    idx4: Vec<[usize; 4]>,
    wgt4: Vec<[f32; 4]>,
}

fn mesh_points(car: &Car) -> Vec<MeshPts> {
    let mut out = Vec::new();
    for m in &car.meshes {
        let p = car.floats(m.positions);
        let n = car.floats(m.normals);
        let bi = car.floats(m.bone_idx);
        let bw = car.floats(m.bone_wgt);
        let nv = m.vertices as usize;
        if p.len() < nv * 3 {
            continue;
        }
        let mut mp = MeshPts { name: m.name.clone(), pos: Vec::with_capacity(nv), nrm: Vec::new(),
                               bone: Vec::with_capacity(nv), idx4: Vec::new(), wgt4: Vec::new() };
        for i in 0..nv {
            mp.pos.push(Vec3::new(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]));
            if n.len() >= nv * 3 {
                mp.nrm.push(Vec3::new(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]));
            }
            if bi.len() >= nv * 4 && bw.len() >= nv * 4 {
                let mut best = (0.0f32, usize::MAX);
                let mut i4 = [0usize; 4];
                let mut w4 = [0.0f32; 4];
                for k in 0..4 {
                    i4[k] = bi[i * 4 + k].max(0.0) as usize;
                    w4[k] = bw[i * 4 + k];
                    if bw[i * 4 + k] > best.0 {
                        best = (bw[i * 4 + k], bi[i * 4 + k] as usize);
                    }
                }
                mp.bone.push(best.1);
                mp.idx4.push(i4);
                mp.wgt4.push(w4);
            } else {
                mp.bone.push(usize::MAX);
                mp.idx4.push([0; 4]);
                mp.wgt4.push([0.0; 4]);
            }
        }
        out.push(mp);
    }
    out
}


fn descends(rig: &Rig, skel: &Skeleton, b: usize, anc: usize) -> bool {
    let mut p = Some(b);
    let mut g = 0;
    while let Some(x) = p {
        if x == anc {
            return true;
        }
        p = skel.bones.get(x).and_then(|b| b.parent);
        g += 1;
        if g > 256 {
            break;
        }
    }
    let _ = rig;
    false
}

fn to_m(skin: &[[f32; 16]]) -> Vec<Mat4> {
    skin.iter().map(Mat4::from_cols_array).collect()
}

/// What the animation check found: records in the report shape
/// `{car, check, severity, detail}`, the measured tables (physics, slots,
/// corners, ...) and key poses (label, skin matrices) for the sheet.
pub struct AnimCheck {
    pub records: Vec<Value>,
    pub extra: Value,
    pub poses: Vec<(String, Vec<[f32; 16]>)>,
}

pub fn analyse(car: &Car, skeleton: Option<&Skeleton>, name: &str) -> AnimCheck {
    let t0 = std::time::Instant::now();
    let mut rep = Report { issues: Vec::new() };
    let to_records = |issues: &[Value]| -> Vec<Value> {
        issues.iter().map(|i| json!({
            "car": name,
            "check": format!("anim_{}", i["code"].as_str().unwrap_or("?")),
            "severity": i["severity"],
            "detail": i["msg"],
        })).collect()
    };
    let Some(skel) = skeleton else {
        rep.add("error", "no_skeleton", "no skeleton: the animation checks cannot run".into());
        return AnimCheck { records: to_records(&rep.issues), extra: Value::Null, poses: Vec::new() };
    };
    let Some(rig) = skel.rig.as_deref() else {
        rep.add("error", "no_rig", "no rig built: the animation checks cannot run".into());
        return AnimCheck { records: to_records(&rep.issues), extra: Value::Null, poses: Vec::new() };
    };
    for n in &rig.notes {
        rep.add("warn", "rig", n.clone());
    }
    // ---- physics data ----
    let mut phys_json = Value::Null;
    match rig.phys.as_ref() {
        None => rep.add("error", "physics_missing", "no CarData found under data/".into()),
        Some(p) => {
            for n in &p.notes {
                rep.add("warn", "physics_range", n.clone());
            }
            // does the physics agree with where the wheel bones are?
            let pos = |c: &str| skel.index.get(&format!("WHEEL_{c}")).map(|&i| rig.bind_of(i).w_axis.truncate());
            if let (Some(lf), Some(rf), Some(lr), Some(rr)) = (pos("LF"), pos("RF"), pos("LR"), pos("RR")) {
                let tf = (lf.x - rf.x).abs();
                let tr = (lr.x - rr.x).abs();
                let wb = ((lf.z + rf.z) - (lr.z + rr.z)).abs() * 0.5;
                for (what, mesh, ph) in [("front track", tf, p.track[0]), ("rear track", tr, p.track[1]),
                                         ("wheelbase", wb, p.wheelbase)] {
                    if ph > 0.0 && (mesh - ph).abs() > 0.15 {
                        rep.add("warn", "physics_vs_mesh",
                                format!("{what}: physics {ph:.3} m, wheel bones {mesh:.3} m"));
                    }
                }
            }
            phys_json = json!({
                "source": p.source, "mass": p.mass, "wheelbase": p.wheelbase, "track": p.track,
                "cg_front": p.cg_front, "steer_lock_deg": p.steer_lock_deg, "steer_ratio": p.steer_ratio,
                "wheel_lock_deg": p.wheel_lock_deg(),
                "spring": [p.coil[0].spring, p.coil[1].spring],
                "bump": [p.coil[0].bump, p.coil[1].bump], "droop": [p.coil[0].droop, p.coil[1].droop],
                "arb": [p.coil[0].arb, p.coil[1].arb],
                "geometry": [format!("{:?}", p.geo[0].kind), format!("{:?}", p.geo[1].kind)],
                "idle_rpm": p.idle_rpm, "limiter_rpm": p.limiter_rpm, "gears": p.gears,
                "final_drive": p.final_drive,
                "limiter_raw": p.limiter_raw,
                "engine_source": p.engine_source,
                "torque_curve_points": p.torque.len(),
                "turbos": p.turbos.iter().map(|t| json!({"max_boost": t.max_boost, "wastegate": t.wastegate,
                    "rpm_ref": t.rpm_ref, "ctrl_stages": t.ctrl.len(), "boost_at_limiter": t.boost(p.limiter_rpm, 1.0)})).collect::<Vec<_>>(),
                "power_mult": p.power_mult,
                "peak_torque": p.peak_torque().map(|(r, t)| json!({"rpm": r, "nm": t})),
                "tyre_radius_phys": p.tyre_radius,
            });
        }
    }
    let pts = mesh_points(car);
    // bones that carry geometry
    let mut verts_on = vec![0usize; skel.bones.len()];
    for m in &pts {
        for &b in &m.bone {
            if b < verts_on.len() {
                verts_on[b] += 1;
            }
        }
    }
    let n_slots = rig.slots.len();
    let rest_inp = Inputs::rest(n_slots);
    let rest = rig.pose(&rest_inp);
    if !finite(&rest) {
        rep.add("error", "nan_pose", "rest pose has NaN/inf matrices".into());
    }
    let rest_w = rig.world_of(&rest, Mat4::IDENTITY);

    // ---- bindings: every slot, start to end ----
    let mut slots_json = Vec::new();
    let has_car_clips = rig.slots.iter().any(|s| !s.kind.is_driver() && s.anim.is_some());
    if !has_car_clips {
        rep.add("error", "no_car_animations",
                "the actor binds no car animation clips (pedals, wipers, doors...)".into());
    }
    let has_driver = skel.bones.iter().any(|b| b.name.starts_with(crate::driver::BONE_PREFIX));
    for want in [Kind::Accel, Kind::Brake] {
        if rig.slot_of(&want).is_none() {
            rep.add("info", "slot_unbound", format!("no {} clip bound", want.label()));
        }
    }
    if has_driver {
        for want in [Kind::DrvSteer, Kind::DrvAccel, Kind::DrvBrake] {
            if rig.slot_of(&want).is_none() {
                rep.add("warn", "driver_slot_unbound", format!("driver has no {} clip", want.label()));
            }
        }
    }
    for (si, s) in rig.slots.iter().enumerate() {
        let label = s.kind.label();
        for m in &s.missing {
            rep.add("error", "clip_missing", format!("{label}: {m} does not resolve"));
        }
        let ntr = s.anim.as_ref().map_or(0, |a| a.tracks.len());
        if !s.orphan_tracks.is_empty() {
            // a clip naming some absent bones is normal (hardtop / windscreen
            // variants); naming nothing the car has is not
            let sev = if s.orphan_tracks.len() == ntr { "warn" } else { "info" };
            if !(s.kind.is_driver() && sev == "info") {
                rep.add(sev, "clip_unbound_bones", format!("{label}: {} of {ntr} tracks name bones the car does not have ({})",
                    s.orphan_tracks.len(), s.orphan_tracks.iter().take(4).cloned().collect::<Vec<_>>().join(", ")));
            }
        }
        let n = s.frames();
        let mut max_move = 0.0f32;
        let mut max_rot = 0.0f32;
        let mut nan = false;
        let mut fly = 0.0f32;
        if n > 0 {
            for f in [0usize, n / 4, n / 2, 3 * n / 4, n.saturating_sub(1)] {
                let mut local = rest_inp.clone();
                // drive the slot through its own input so the real mapping is used
                let frac = f as f32 / (n.max(2) - 1) as f32;
                local.open = vec![0.0; n_slots];
                local.open[si] = frac;
                local.cs.throttle = if s.kind == Kind::Accel || s.kind == Kind::DrvAccel { frac } else { 0.0 };
                local.cs.brake = if s.kind == Kind::Brake || s.kind == Kind::DrvBrake { frac } else { 0.0 };
                local.cs.clutch = if s.kind == Kind::Clutch || s.kind == Kind::DrvClutch { frac } else { 0.0 };
                local.cs.handbrake = if s.kind == Kind::Handbrake || s.kind == Kind::DrvHandbrake { frac } else { 0.0 };
                let len = s.len_s();
                match s.kind {
                    Kind::Wiper => local.wiper = Some(frac * len * 0.999),
                    Kind::ShiftUp => local.shift = Some((true, frac * len * 0.999)),
                    Kind::ShiftDown => local.shift = Some((false, frac * len * 0.999)),
                    Kind::DrvShiftUp | Kind::DrvApproach => {
                        let b = rig.blend;
                        local.shift = Some((true, frac * (b.h_in + b.h_play + b.h_out) * 0.999));
                    }
                    Kind::DrvShiftDown => {
                        let b = rig.blend;
                        local.shift = Some((false, frac * (b.h_in + b.h_play + b.h_out) * 0.999));
                    }
                    Kind::DrvPaddleUp | Kind::DrvPaddleDown => {
                        let b = rig.blend;
                        local.shift = Some((s.kind == Kind::DrvPaddleUp, frac * (b.p_in + b.p_play + b.p_out) * 0.999));
                        local.paddles = true;
                    }
                    Kind::PaddleUp => { local.shift = Some((true, frac * len * 0.999)); local.paddles = true; }
                    Kind::PaddleDown => { local.shift = Some((false, frac * len * 0.999)); local.paddles = true; }
                    Kind::Susp(c) => {
                        let (d, b) = rig.corners.iter().find(|k| k.idx == c).map_or((0.05, 0.05), |k| k.travel);
                        local.travel[c] = -d + (d + b) * frac;
                    }
                    Kind::DrvSteer | Kind::DrvSteerPaddles => {
                        let half = rig.drv_steer_half_deg.max(90.0);
                        local.wheel_deg = (frac * 2.0 - 1.0) * half;
                    }
                    Kind::Optional(_) => {
                        local.drs = frac > 0.5;
                        local.cs.speed_mps = frac * 90.0;
                        local.cs.throttle = frac;
                        local.cs.rpm = frac * 9000.0;
                        local.g = Vec3::splat(frac * 3.0);
                    }
                    _ => {}
                }
                let sk = rig.pose(&local);
                if !finite(&sk) {
                    nan = true;
                    continue;
                }
                let w = rig.world_of(&sk, Mat4::IDENTITY);
                for (bi, (a, b)) in w.iter().zip(&rest_w).enumerate() {
                    let d = (a.w_axis.truncate() - b.w_axis.truncate()).length();
                    let r = {
                        let q0 = glam::Quat::from_mat4(b).normalize();
                        let q1 = glam::Quat::from_mat4(a).normalize();
                        q0.angle_between(q1).to_degrees()
                    };
                    if verts_on.get(bi).copied().unwrap_or(0) > 0 || s.kind.is_driver() {
                        max_move = max_move.max(d);
                        max_rot = max_rot.max(r);
                    }
                    fly = fly.max(d);
                }
            }
        }
        if nan {
            rep.add("error", "nan_pose", format!("{label}: NaN/inf pose"));
        }
        if n > 0 && max_move < 0.001 && max_rot < 0.3 && s.kind != Kind::DrvBase && s.kind != Kind::DrvIdle {
            rep.add("warn", "clip_static", format!("{label}: playing it moves no visible geometry"));
        }
        let limit = match &s.kind {
            Kind::Door(_) | Kind::Hood | Kind::Tailgate | Kind::Optional(_) | Kind::Wiper => 3.0,
            k if k.is_driver() => 1.5,
            _ => 0.6,
        };
        if fly > limit {
            rep.add("error", "bone_flies", format!("{label}: a bone moves {fly:.2} m (limit {limit})"));
        }
        slots_json.push(json!({
            "slot": label, "play": format!("{:?}", s.play), "frames": n,
            "paths": s.paths.iter().map(|p| crate::load::base_of(p)).collect::<Vec<_>>(),
            "missing": s.missing, "orphan_tracks": s.orphan_tracks.len(),
            "max_move_m": (max_move * 1000.0).round() / 1000.0, "max_rot_deg": max_rot.round(),
        }));
    }

    // ---- steering, suspension, wheels ----
    let mut corners_json = Vec::new();
    if rig.corners.len() < 4 {
        rep.add("error", "wheels_missing", format!("{} of 4 wheel corners found", rig.corners.len()));
    }
    let lock = rig.phys.as_ref().map_or(360.0, |p| p.steer_lock_deg);
    let mut drive = crate::drive::Drive::new(rig);
    let sweep_pose = |d: &mut crate::drive::Drive, steer: f32, travel: [f32; 4]| -> Inputs {
        let c = crate::drive::Controls { steer_v: Some(steer), ..crate::drive::Controls::none() };
        for _ in 0..3 {
            d.step(rig, &c, 1.0 / 60.0);
        }
        let mut inp = d.inputs(rig);
        inp.travel = travel;
        inp.body = Mat4::IDENTITY;
        inp
    };
    // Tyre-in-body. Each tyre is modelled as the annulus its own mesh fills
    // (outer radius, rim radius, width, in the tyre bone's frame); a BODY
    // vertex inside that annulus at some pose is bodywork the tyre would cut
    // through. Testing body points against the tyre volume needs no normals
    // and no closed meshes, and calipers / hubs inside the rim never count.
    struct TyreVol { bone: usize, axle: Vec3, r_out: f32, r_in: f32, half_w: f32, centre: Vec3, corner: usize }
    let mut vols: Vec<TyreVol> = Vec::new();
    for c in &rig.corners {
        let cn = crate::carrig::CORNERS[c.idx];
        let Some(m) = pts.iter().find(|m| m.name == format!("tyre_{cn}")) else { continue };
        let Some(&(bone, axle)) = c.spin.iter().find(|s| skel.bones[s.0].name.starts_with("TYRE_"))
            .or_else(|| c.spin.first()) else { continue };
        let inv = rig.bind_of(bone).inverse();
        let (mut ro, mut ri, mut hw) = (0.0f32, f32::MAX, 0.0f32);
        for p in &m.pos {
            let l = inv.transform_point3(*p);
            let x = l.dot(axle);
            let r = (l - axle * x).length();
            ro = ro.max(r);
            ri = ri.min(r);
            hw = hw.max(x.abs());
        }
        if ro > 0.1 {
            vols.push(TyreVol { bone, axle, r_out: ro, r_in: ri, half_w: hw,
                                centre: rig.bind_of(bone).w_axis.truncate(), corner: c.idx });
        }
    }
    let hubs: Vec<usize> = rig.corners.iter().filter_map(|c| c.hub).collect();
    let mut body_pts: Vec<Vec3> = Vec::new();
    for m in &pts {
        if m.name.starts_with("tyre_") || m.name.starts_with("rim_")
            || m.name.starts_with(crate::driver::PREFIX) || m.name.starts_with("contact_") {
            continue;
        }
        for i in 0..m.pos.len() {
            let b = m.bone[i];
            if b != usize::MAX && hubs.iter().any(|&h| descends(rig, &skel, b, h)) {
                continue;
            }
            // bones that move with the wheels or are look-ats at them
            let bn = skel.bones.get(b).map(|x| x.name.as_str()).unwrap_or("");
            if bn.contains("SUSP") || bn.contains("DIR_") || bn.contains("WHEEL") || bn.contains("TYRE")
                || bn.contains("DISC") || bn.contains("HUB") || bn.starts_with("drv:") {
                continue;
            }
            let p = m.pos[i];
            if vols.iter().any(|v| (p - v.centre).length() < v.r_out + 0.25) {
                body_pts.push(p);
            }
        }
    }
    let mut clearance: Vec<Value> = Vec::new();
    let mut tyre_test = |label: &str, inp: &Inputs, rep: &mut Report| -> usize {
        let sk = to_m(&rig.pose(inp));
        let mut worst = (0usize, 0.0f32, 0usize);
        for v in &vols {
            let w = sk[v.bone] * rig.bind_of(v.bone);
            let inv = w.inverse();
            let mut n = 0usize;
            let mut deep = 0.0f32;
            for p in &body_pts {
                let l = inv.transform_point3(*p);
                let x = l.dot(v.axle);
                if x.abs() > v.half_w - 0.004 {
                    continue;
                }
                let r = (l - v.axle * x).length();
                if r > v.r_in + 0.015 && r < v.r_out - 0.006 {
                    n += 1;
                    deep = deep.max(v.r_out - r);
                }
            }
            if n > worst.0 {
                worst = (n, deep, v.corner);
            }
        }
        clearance.push(json!({"pose": label, "verts": worst.0, "depth_mm": (worst.1 * 1000.0).round(),
                              "corner": crate::carrig::CORNERS[worst.2]}));
        // Thresholds from the stock fleet (L:/evoview_work/anim/check/stock.json):
        // at rest and at lock it is ~0 for all but 2-3 cars; at full free bump
        // half the stock cars already put 20-340 arch-liner vertices inside the
        // tyre, so a bump pose only counts when it is well past that.
        let bump = label.contains("bump");
        let (warn, err) = if bump { (400, 1500) } else { (25, 150) };
        if worst.0 >= warn {
            rep.add(if worst.0 >= err { "error" } else { "warn" }, "tyre_in_body",
                    format!("{label}: {} body vertices inside the {} tyre (up to {:.0} mm deep)",
                            worst.0, crate::carrig::CORNERS[worst.2], worst.1 * 1000.0));
        }
        worst.0
    };
    if !vols.is_empty() && !body_pts.is_empty() {
        let zero = [0.0; 4];
        let inp = sweep_pose(&mut drive, 0.0, zero);
        tyre_test("rest", &inp, &mut rep);
        // bump to where the bump stop starts: past it the stop's own squash
        // is a physics cushion the visual model is not built to clear
        let free: Vec<f32> = rig.phys.as_ref().map_or(vec![0.05, 0.05], |p| vec![p.coil[0].bump_free, p.coil[1].bump_free]);
        let full: [f32; 4] = {
            let mut t = [0.0; 4];
            for c in &rig.corners { t[c.idx] = free[if c.front { 0 } else { 1 }].min(c.travel.1); }
            t
        };
        let droop: [f32; 4] = {
            let mut t = [0.0; 4];
            for c in &rig.corners { t[c.idx] = -c.travel.0; }
            t
        };
        for (label, steer, tr) in [("full bump", 0.0, full), ("full droop", 0.0, droop),
                                   ("full bump + left lock", -1.0, full), ("full bump + right lock", 1.0, full),
                                   ("left lock", -1.0, zero), ("right lock", 1.0, zero)] {
            let inp = sweep_pose(&mut drive, steer, tr);
            tyre_test(label, &inp, &mut rep);
        }
    } else if vols.is_empty() {
        rep.add("warn", "tyre_volume", "no tyre meshes to test clearance with".into());
    }
    for c in &rig.corners {
        let cn = crate::carrig::CORNERS[c.idx];
        // does the wheel actually move with travel and steer?
        let probe = |d: &mut crate::drive::Drive, steer: f32, tr: f32| -> (Vec3, Vec3) {
            let mut t = [0.0; 4];
            t[c.idx] = tr;
            let inp = sweep_pose(d, steer, t);
            let w = rig.world_of(&rig.pose(&inp), Mat4::IDENTITY);
            let b = c.spin.first().map(|s| s.0).or(c.hub).unwrap_or(0);
            (w[b].w_axis.truncate(), w[b].transform_vector3(Vec3::Z))
        };
        let (p0, f0) = probe(&mut drive, 0.0, 0.0);
        let (pb, _) = probe(&mut drive, 0.0, c.travel.1);
        let (pd, _) = probe(&mut drive, 0.0, -c.travel.0);
        let (_, fl) = probe(&mut drive, 1.0, 0.0);
        let up = pb.y - p0.y;
        let down = p0.y - pd.y;
        if (up - c.travel.1).abs() > 0.01 + c.travel.1 * 0.25 {
            rep.add(if c.clip.is_some() { "info" } else { "warn" }, "travel_mismatch", format!("{cn}: wheel rises {:.0} mm at full bump, data says {:.0} mm",
                    up * 1000.0, c.travel.1 * 1000.0));
        }
        let lateral = (pb.x - p0.x).abs().max((pd.x - p0.x).abs());
        if lateral > 0.08 {
            rep.add("warn", "suspension_scrub", format!("{cn}: wheel moves {:.0} mm sideways over its travel",
                    lateral * 1000.0));
        }
        let steer_deg = f0.angle_between(fl).to_degrees();
        if c.front {
            let want = rig.phys.as_ref().map_or(25.0, |p| p.wheel_lock_deg());
            if (steer_deg - want).abs() > want * 0.5 + 3.0 {
                rep.add("warn", "steer_angle", format!("{cn}: road wheel turns {steer_deg:.1} deg at lock, expected ~{want:.1}"));
            }
        }
        // a spinning wheel must turn about its own axle: the rim and tyre
        // are placed with the bone's bind, so the bone's X (the axle) has to
        // stay put over a turn. (F2004 / SF-25: 3 deg camber in the bind,
        // spun about the car's X = a 6 deg wobble once a turn.)
        let mut wobble = 0.0f32;
        {
            let mut inp = sweep_pose(&mut drive, 0.0, [0.0; 4]);
            let w0 = rig.world_of(&rig.pose(&inp), Mat4::IDENTITY);
            for k in 1..12 {
                inp.spin[c.idx] = k as f32 * std::f32::consts::TAU / 12.0;
                let w = rig.world_of(&rig.pose(&inp), Mat4::IDENTITY);
                for &(b, _) in &c.spin {
                    let a0 = w0[b].transform_vector3(Vec3::X).normalize_or_zero();
                    let a = w[b].transform_vector3(Vec3::X).normalize_or_zero();
                    wobble = wobble.max(a0.angle_between(a).to_degrees());
                }
            }
        }
        if wobble > 0.3 {
            rep.add("warn", "wheel_wobble", format!("{cn}: the wheel's axle swings {wobble:.1} deg over a turn"));
        }
        // geometry on the spinning bones
        let spin_verts: usize = c.spin.iter().map(|s| verts_on.get(s.0).copied().unwrap_or(0)).sum();
        if spin_verts == 0 {
            rep.add("error", "wheel_not_bound", format!("{cn}: nothing is skinned to the spinning wheel bones"));
        }
        if !skel.index.contains_key(&format!("DISC_{cn}")) {
            rep.add("info", "no_disc_bone", format!("{cn}: no DISC_{cn} bone - discs cannot spin"));
        }
        corners_json.push(json!({
            "corner": cn, "kind": rig.phys.as_ref().map(|p| format!("{:?}", p.geo[if c.front {0} else {1}].kind)),
            "travel_mm": [c.travel.0 * 1000.0, c.travel.1 * 1000.0], "rise_mm": up * 1000.0, "drop_mm": down * 1000.0,
            "lateral_mm": lateral * 1000.0, "steer_deg_at_lock": steer_deg, "clip": c.clip.is_some(),
            "spin_bones": c.spin.len(), "spin_verts": spin_verts, "wobble_deg": wobble,
        }));
    }
    // calipers must not spin, discs must stay at the wheel
    let spin_bones: Vec<usize> = rig.corners.iter().flat_map(|c| c.spin.iter().map(|s| s.0)).collect();
    for m in &pts {
        let low = m.name.to_lowercase();
        if !(low.contains("caliper") || low.contains("disc")) || m.bone.is_empty() {
            continue;
        }
        let b = m.bone[0];
        if low.contains("caliper") && spin_bones.contains(&b) {
            rep.add("error", "caliper_spins", format!("{}: caliper is skinned to a spinning bone", m.name));
        }
        let c = m.pos.iter().copied().sum::<Vec3>() / m.pos.len().max(1) as f32;
        let near = rig.corners.iter().map(|k| (k.centre - c).length()).fold(f32::MAX, f32::min);
        if near > 0.45 {
            rep.add("error", "brake_detached", format!("{}: {:.2} m from the nearest wheel centre", m.name, near));
        }
    }
    // steering wheel and hands
    let mut steer_json = json!({});
    match rig.steer_bone {
        None => {}
        Some(sb) => {
            let on: usize = (0..skel.bones.len()).filter(|&b| descends(rig, &skel, b, sb))
                .map(|b| verts_on.get(b).copied().unwrap_or(0)).sum();
            if on == 0 {
                rep.add("error", "steering_wheel_not_bound", "no geometry follows STEER_BASE".into());
            }
            let half = rig.drv_steer_half_deg;
            if has_driver && rig.slot_of(&Kind::DrvSteer).is_some() {
                if lock > rig.drv_steer_range_deg + 1.0 {
                    rep.add("info", "driver_steer_short",
                            format!("steer lock +-{lock:.0} deg is past the driver clip's +-{:.0}: hands stop there",
                                    rig.drv_steer_range_deg));
                }
                // the clip must come back round to its middle pose at either end
                if let Some(si) = rig.slot_of(&Kind::DrvSteer) {
                    let lh = skel.index.get(&format!("{}LeftHand", crate::driver::BONE_PREFIX)).copied();
                    if let Some(lh) = lh {
                        let at = |deg: f32| {
                            let mut inp = rest_inp.clone();
                            inp.wheel_deg = deg;
                            rig.world_of(&rig.pose(&inp), Mat4::IDENTITY)[lh].w_axis.truncate()
                        };
                        let r = rig.drv_steer_range_deg;
                        let d = (at(r) - at(0.0)).length().max((at(-r) - at(0.0)).length());
                        if d > 0.05 && rig.slots[si].frames() > 2 {
                            rep.add("warn", "driver_steer_range",
                                    format!("steer clip is not a full turn (ends {:.0} mm from its centre pose); mapped to +-{r:.0} deg from the hands' grip", d * 1000.0));
                        }
                    }
                }
                // Hands stay on the rim. The rim is the circle the hands sit on
                // at rest, turning with the wheel; hand-over-hand lets ONE hand
                // leave it, so only both hands off at once is a fault.
                let lh = skel.index.get(&format!("{}LeftHand", crate::driver::BONE_PREFIX)).copied();
                let rh = skel.index.get(&format!("{}RightHand", crate::driver::BONE_PREFIX)).copied();
                if let (Some(lh), Some(rh)) = (lh, rh) {
                    let col = rig.bind_of(sb).w_axis.truncate();
                    let axis = rig.bind_of(sb).transform_vector3(rig.steer_axis).normalize();
                    let ring = |p: Vec3| -> (f32, f32) {
                        let r = p - col;
                        let a = r.dot(axis);
                        ((r - axis * a).length(), a)
                    };
                    let inp0 = sweep_pose(&mut drive, 0.0, [0.0; 4]);
                    let w0 = rig.world_of(&rig.pose(&inp0), Mat4::IDENTITY);
                    let (rad, depth) = {
                        let (a, b) = (ring(w0[lh].w_axis.truncate()), ring(w0[rh].w_axis.truncate()));
                        ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5)
                    };
                    let off = |p: Vec3| {
                        let (r, a) = ring(p);
                        (((r - rad).powi(2) + (a - depth).powi(2)).sqrt()) > 0.09
                    };
                    let mut both_off = 0;
                    let mut samples = Vec::new();
                    for k in 0..=12 {
                        let st = -1.0 + k as f32 / 6.0;
                        let inp = sweep_pose(&mut drive, st, [0.0; 4]);
                        let w = rig.world_of(&rig.pose(&inp), Mat4::IDENTITY);
                        let (l, r) = (off(w[lh].w_axis.truncate()), off(w[rh].w_axis.truncate()));
                        if l && r {
                            both_off += 1;
                        }
                        samples.push((st, l, r));
                    }
                    if rad > 0.30 || rad < 0.08 {
                        rep.add("warn", "hands_off_wheel", format!("hands sit {rad:.2} m from the steering column at rest"));
                    }
                    if both_off >= 4 {
                        rep.add("warn", "hands_leave_wheel",
                                format!("both hands off the rim at {both_off} of 13 steer positions"));
                    }
                    steer_json = json!({"clip_grip_deg": half, "lock_deg": lock, "rim_radius_m": rad,
                                        "samples": samples.iter().map(|s| json!([s.0, s.1, s.2])).collect::<Vec<_>>()});
                }
            }
        }
    }
    // ---- needles ----
    let mut needles_json = Vec::new();
    for n in &rig.needles {
        let mut ok = true;
        match n.bone {
            None => {
                // dash variants (TRD cluster, clock option) name bones the
                // fitted interior lacks; only a car with NO live needle is wrong
                let any = rig.needles.iter().any(|m| m.bone.is_some());
                rep.add(if any { "info" } else { "error" }, "needle_unbound",
                        format!("{:?} needle: bone {} not in the car", n.gauge, n.bone_name));
                ok = false;
            }
            Some(b) => {
                if verts_on.get(b).copied().unwrap_or(0) == 0 {
                    rep.add("warn", "needle_no_geometry", format!("{:?} needle {}: no vertices on it", n.gauge, n.bone_name));
                }
            }
        }
        if n.lut.len() < 2 {
            rep.add("warn", "needle_lut", format!("{:?} needle {}: LUT has {} points", n.gauge, n.bone_name, n.lut.len()));
            ok = false;
        }
        let mut swept = Vec::new();
        if ok {
            let (lo, hi) = (n.lut[0].0, n.lut[n.lut.len() - 1].0);
            let (dlo, dhi) = n.lut.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p.1), a.1.max(p.1)));
            let b = n.bone.unwrap();
            let mut prev: Option<f32> = None;
            for k in 0..=8 {
                let x = lo - (hi - lo) * 0.1 + (hi - lo) * 1.2 * k as f32 / 8.0;
                let mut inp = rest_inp.clone();
                match n.gauge {
                    Gauge::Rpm => { inp.cs.rpm = x; inp.cs.rpm_limit = hi; }
                    Gauge::Speed => { inp.cs.speed_mps = x / 3.6; inp.cs.rpm = 1000.0; }
                    Gauge::Fuel => inp.fuel_l = x,
                    Gauge::WaterTemp => { inp.water_c = x; inp.cs.rpm = 1000.0; }
                    Gauge::OilTemp => { inp.oil_c = x; inp.cs.rpm = 1000.0; }
                    _ => {}
                }
                let sk = rig.pose(&inp);
                if !finite(&sk) {
                    rep.add("error", "nan_pose", format!("{:?} needle: NaN pose", n.gauge));
                    break;
                }
                let w = rig.world_of(&sk, Mat4::IDENTITY);
                let ang = glam::Quat::from_mat4(&rest_w[b]).normalize()
                    .angle_between(glam::Quat::from_mat4(&w[b]).normalize()).to_degrees();
                swept.push(ang);
                if let Some(p) = prev {
                    let _ = p;
                }
                prev = Some(ang);
            }
            let max = swept.iter().cloned().fold(0.0f32, f32::max);
            let span = (dhi - dlo).abs().min(360.0);
            if matches!(n.gauge, Gauge::Rpm | Gauge::Speed) && max < 1.0 {
                rep.add("error", "needle_static", format!("{:?} needle {} does not move over its range", n.gauge, n.bone_name));
            }
            if max > span.max(dhi.abs()).max(dlo.abs()) + 2.0 && span < 359.0 {
                rep.add("warn", "needle_range", format!("{:?} needle {} turns {max:.0} deg, LUT spans {span:.0}", n.gauge, n.bone_name));
            }
        }
        let at = n.bone.map(|b| { let p = rig.bind_of(b).w_axis; [p.x, p.y, p.z] });
        needles_json.push(json!({"gauge": format!("{:?}", n.gauge), "bone": n.bone_name, "bound": n.bone.is_some(), "at": at,
                                 "lut": n.lut, "swept_deg": swept}));
    }
    for m in &rig.movables {
        if m.bone.is_none() {
            rep.add("warn", "movable_unbound", format!("movable part bone {} not in the car", m.bone_name));
        }
    }
    // ---- engine, gears and rpm ----
    // Full throttle from 1st, upshifting at the limiter. In every gear but
    // the top one the car must reach the limiter, and the speed it does so
    // at must match the gearing maths: v = rpm * 2 pi / 60 * r / (gear x
    // final). Then the scripted demo must reach the limiter too (the shift
    // lights and lift-off backfires depend on it).
    let mut gears_json = Vec::new();
    let mut gearing_js = Value::Null;
    if let Some(p) = rig.phys.as_ref() {
        if p.limiter_raw > 0.0 && p.limiter_raw < 100.0 {
            rep.add("error", "rev_limiter_units",
                    format!("rev limiter reads {:.1} rpm - thousands of rpm, a unit slip", p.limiter_raw));
        }
        if p.torque.len() < 2 {
            rep.add("warn", "no_torque_curve", format!("engine {} has no power curve; a flat one stands in",
                                                      if p.engine_source.is_empty() { "?" } else { &p.engine_source }));
        }
    }
    {
        let mut d = crate::drive::Drive::new(rig);
        d.cs.gear = 1;
        let lim = d.limiter();
        let r = d.radius();
        let n = d.n_gears();
        let mut hit: Vec<Option<(f32, f32, f32)>> = vec![None; n.max(0) as usize];
        let mut max_speed = 0.0f32;
        let mut max_rpm_g1 = 0.0f32;
        let c = crate::drive::Controls { throttle_v: 1.0, ..crate::drive::Controls::none() };
        let mut t = 0.0f32;
        for i in 0..(60 * 90) {
            d.step(rig, &c, 1.0 / 60.0);
            t += 1.0 / 60.0;
            if !d.cs.rpm.is_finite() || !d.cs.speed_mps.is_finite() {
                rep.add("error", "drive_nan", format!("drive model went non-finite at step {i}"));
                break;
            }
            max_speed = max_speed.max(d.cs.speed_mps);
            if d.cs.gear == 1 && d.cs.clutch < 0.05 {
                max_rpm_g1 = max_rpm_g1.max(d.cs.rpm);
            }
            let g = d.cs.gear;
            if g >= 1 && d.cs.rpm >= lim * 0.97 && d.cs.clutch < 0.05 {
                let k = (g - 1) as usize;
                if k < hit.len() && hit[k].is_none() {
                    hit[k] = Some((d.cs.speed_mps, d.cs.rpm, t));
                }
                if g < n {
                    d.shift_up();
                } else {
                    break;
                }
            }
            if i % 120 == 0 {
                let inp = d.inputs(rig);
                if !finite(&rig.pose(&inp)) {
                    rep.add("error", "nan_pose", format!("NaN pose while driving at {:.0} km/h", d.cs.speed_mps * 3.6));
                    break;
                }
            }
        }
        for g in 1..=n {
            let ratio = d.gear_ratio(g).unwrap_or(0.0);
            let want = if ratio > 0.0 { lim * 0.97 * std::f32::consts::TAU / 60.0 * r / ratio } else { 0.0 };
            match hit[(g - 1) as usize] {
                Some((v, rpm, at)) => {
                    let want = rpm * std::f32::consts::TAU / 60.0 * r / ratio.max(1e-3);
                    let err = (v - want) / want.max(0.1);
                    if err.abs() > 0.05 {
                        rep.add("error", "gearing_mismatch", format!("gear {g}: {:.0} km/h at {:.0} rpm, gearing says {:.0} km/h ({:+.0}%)",
                                v * 3.6, rpm, want * 3.6, err * 100.0));
                    }
                    gears_json.push(json!({"gear": g, "ratio": ratio, "kmh_at_limiter": v * 3.6,
                                           "kmh_gearing": want * 3.6, "rpm": rpm, "t": at}));
                }
                None => {
                    // a gear whose limiter speed is past the car's terminal
                    // speed (drag) is fine; any other gear must get there
                    if want > max_speed * 0.97 {
                        rep.add("info", "drag_limited", format!("gear {g}: limiter at {:.0} km/h is past the car's terminal {:.0} km/h", want * 3.6, max_speed * 3.6));
                    } else if g < n {
                        rep.add("error", "limiter_unreached", format!("gear {g} never reaches the limiter at full throttle \
                                 (gearing: {:.0} km/h)", want * 3.6));
                    }
                    gears_json.push(json!({"gear": g, "ratio": ratio, "kmh_at_limiter": null, "kmh_gearing": want * 3.6}));
                }
            }
        }
        if max_speed * 3.6 < 60.0 {
            rep.add("warn", "drive_slow", format!("full throttle for 90 s reaches only {:.0} km/h", max_speed * 3.6));
        }
        let top = d.gear_ratio(n).map(|ra| lim * std::f32::consts::TAU / 60.0 * r / ra * 3.6).unwrap_or(0.0);
        if !(80.0..=600.0).contains(&top) {
            rep.add("warn", "gearing_implausible", format!("top gear at the limiter is {top:.0} km/h"));
        }
        // the scripted demo
        let mut dd = crate::drive::Drive::new(rig);
        let mut demo_max = 0.0f32;
        let mut tt = 0.0;
        while tt < crate::drive::DEMO_LEN {
            crate::drive::demo_run(&mut dd, rig, tt, tt + 0.05);
            tt += 0.05;
            demo_max = demo_max.max(dd.cs.rpm);
        }
        // a single-speed EV is drag-limited short of its limiter: then the
        // most it can reach in 1st is the bar
        let reach = lim.min(max_rpm_g1.max(lim * 0.3));
        let electric = rig.phys.as_ref().map_or(false, |p| p.electric) || n <= 1;
        if demo_max < reach * 0.95 {
            rep.add(if electric { "info" } else { "error" }, "demo_limiter", format!("the demo peaks at {demo_max:.0} rpm; the car reaches {reach:.0} (limiter {lim:.0})"));
        }
        gearing_js = json!({"radius_m": r, "limiter": lim, "demo_max_rpm": demo_max,
                            "max_speed_kmh": max_speed * 3.6, "top_gear_kmh": top});
    }
    // ---- bone sanity across a full demo loop ----
    {
        let mut d = crate::drive::Drive::new(rig);
        let mut worst = (0.0f32, String::new());
        let mut t = 0.0;
        while t < crate::drive::DEMO_LEN {
            crate::drive::demo_run(&mut d, rig, t, t + 0.25);
            t += 0.25;
            let mut inp = d.inputs(rig);
            inp.body = Mat4::IDENTITY;
            let sk = rig.pose(&inp);
            if !finite(&sk) {
                rep.add("error", "nan_pose", format!("NaN pose at demo t={t:.2}"));
                break;
            }
            let w = rig.world_of(&sk, Mat4::IDENTITY);
            for (bi, (a, b)) in w.iter().zip(&rest_w).enumerate() {
                if verts_on.get(bi).copied().unwrap_or(0) == 0 {
                    continue;
                }
                let nm = &skel.bones[bi].name;
                // doors, lids and the driver legitimately travel far
                let lid = ["DOOR", "HOOD", "TAILGATE", "BONNET", "ROOF", "TRUNK", "BOOT", "WING", "SPOILER", "WIPER",
                           "WINDOW", "CANOPY", "HARDTOP", "SUNROOF"]
                    .iter().any(|k| nm.to_uppercase().contains(k)) || nm.starts_with("drv:");
                let d = (a.w_axis.truncate() - b.w_axis.truncate()).length();
                if !lid && d > worst.0 {
                    worst = (d, nm.clone());
                }
            }
        }
        if worst.0 > 0.35 {
            rep.add("error", "bone_flies", format!("{} moves {:.2} m during the demo", worst.1, worst.0));
        }
    }

    // cost of one posed frame (what drive mode pays per frame on the CPU)
    let pose_us = {
        let mut d = crate::drive::Drive::new(rig);
        crate::drive::demo_run(&mut d, rig, 0.0, 4.0);
        let inp = d.inputs(rig);
        let t = std::time::Instant::now();
        let mut n = 0usize;
        for _ in 0..500 {
            n += rig.pose(&inp).len();
        }
        let _ = n;
        t.elapsed().as_secs_f64() * 1e6 / 500.0
    };
    let extra = json!({
        "physics": phys_json,
        "gearing": gearing_js,
        "gears": gears_json,
        "slots": slots_json,
        "corners": corners_json,
        "steering": steer_json,
        "clearance": clearance,
        "needles": needles_json,
        "bones": skel.bones.len(),
        "aims": rig.bone_names.iter().filter(|n| n.starts_with("DIR_")).count(),
        "movables": rig.movables.len(),
        "check_ms": t0.elapsed().as_secs_f64() * 1000.0,
        "pose_us": pose_us,
    });
    // ---- key poses for the contact sheet ----
    let mut poses: Vec<(String, Vec<[f32; 16]>)> = Vec::new();
    {
        let mut d = crate::drive::Drive::new(rig);
        poses.push(("anim rest".into(), rig.pose(&rest_inp)));
        let mut full = [0.0; 4];
        for c in &rig.corners { full[c.idx] = c.travel.1; }
        let inp = sweep_pose(&mut d, -1.0, full);
        poses.push(("anim left lock + bump".into(), rig.pose(&inp)));
        let mut droop = [0.0; 4];
        for c in &rig.corners { droop[c.idx] = -c.travel.0; }
        let inp = sweep_pose(&mut d, 1.0, droop);
        poses.push(("anim right lock + droop".into(), rig.pose(&inp)));
        for t in crate::drive::demo::SHOTS {
            let mut dd = crate::drive::Drive::new(rig);
            crate::drive::demo_run(&mut dd, rig, 0.0, t);
            let inp = dd.inputs(rig);
            poses.push((format!("anim demo {t:.0}s"), rig.pose(&inp)));
        }
    }
    AnimCheck { records: to_records(&rep.issues), extra, poses }
}
