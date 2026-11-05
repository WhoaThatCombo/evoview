//! Brake bias, ABS and TC changed from the cockpit while driving, the way the
//! game's wheel buttons do it.
//!
//! The game's handler is fn 0x1417901a0 (20260907 build). InputAction 148
//! Brake_Bias_Cycle, 140 TractionControl_Cycle and 144 ABS_Cycle each take a
//! payload: 1 steps down, 2 steps up.
//! - TC / ABS: level -/+ 1. Nothing happens unless the car's setup limits
//!   mark the setting modifiable. With min == max the game shows LOCKED_*;
//!   otherwise it clamps to [min, max] and shows CHANGED_*.
//! - Bias: front share -/+ 0.002 (0.2 %, hard-coded; the data's biasStep is
//!   not read there). Not modifiable gives LOCKED_BRAKE_BIAS. Otherwise it
//!   clamps to the limits' min / max.
//!
//! Level 0 is "off" in the cars' own tables (ABS settings[0] has min slip
//! -1, TC settings[0] is empty). evophys skips a level whose min slip is <= 0.
//!
//! The limits come from the car's .carsetuplimits (CarData f16 setupLimits).
//! For the M4 GT3 they are: bias 50-80 %, TC 0-12, ABS 0-10.

use crate::proto::{fields, Val};

#[derive(Clone, Copy, Debug)]
pub struct Limit {
    pub min: f32,
    pub max: f32,
    pub modifiable: bool,
}

/// CarSetupDataLimits {1 mechanical_balance {3 brakes {1 front_bias}},
/// 5 electronics {1 tc1, 3 abs}}. Each is a SetupSettingLimits {1 step,
/// 2 min, 3 max, 4 is_modifiable}. Bias is in %.
#[derive(Clone, Debug, Default)]
pub struct Limits {
    pub bias: Option<Limit>,
    pub tc: Option<Limit>,
    pub abs: Option<Limit>,
}

fn sub(d: &[u8], r: (usize, usize), want: u32) -> Option<(usize, usize)> {
    let mut hit = None;
    fields(d, r.0, r.1, |f, v| {
        if f == want && hit.is_none() {
            hit = v.range();
        }
    });
    hit
}

fn limit(d: &[u8], r: (usize, usize)) -> Limit {
    let mut l = Limit { min: 0.0, max: 0.0, modifiable: false };
    fields(d, r.0, r.1, |f, v| match (f, v) {
        (2, Val::F32(x)) => l.min = x,
        (3, Val::F32(x)) => l.max = x,
        (4, Val::Varint(x)) => l.modifiable = x != 0,
        _ => {}
    });
    l
}

impl Limits {
    pub fn read(d: &[u8]) -> Limits {
        let all = (0, d.len());
        let el = sub(d, all, 5);
        Limits {
            bias: sub(d, all, 1).and_then(|m| sub(d, m, 3)).and_then(|b| sub(d, b, 1)).map(|r| limit(d, r)),
            tc: el.and_then(|e| sub(d, e, 1)).map(|r| limit(d, r)),
            abs: el.and_then(|e| sub(d, e, 3)).map(|r| limit(d, r)),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Knob {
    Bias,
    Tc,
    Abs,
}

impl Knob {
    pub const ALL: [Knob; 3] = [Knob::Bias, Knob::Tc, Knob::Abs];
    pub fn name(self) -> &'static str {
        match self {
            Knob::Bias => "brake bias",
            Knob::Tc => "TC",
            Knob::Abs => "ABS",
        }
    }
}

/// What one press did: the HUD pops the value up, red when locked.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Outcome {
    Changed,
    Locked,
    /// the car has no such system
    None,
}

/// EVOVIEW_AIDS_UNLOCK=1: change what the car's limits lock (and past their
/// range, up to the car's own tables)
fn unlocked() -> bool {
    crate::envc::var_os("EVOVIEW_AIDS_UNLOCK").is_some()
}

/// The ABS / TC level the car runs, 0 when the system is switched off.
pub fn level(rc: &crate::rigcar::RigCar, k: Knob) -> Option<usize> {
    match k {
        Knob::Tc => rc.tc.as_ref().map(|t| if rc.tc_on { t.level } else { 0 }),
        Knob::Abs => rc.abs.as_ref().map(|a| if rc.abs_on { a.level } else { 0 }),
        Knob::Bias => None,
    }
}

/// One cockpit press: `dir` -1 down, +1 up.
pub fn step(rc: &mut crate::rigcar::RigCar, k: Knob, dir: i32) -> Outcome {
    let lim = match k {
        Knob::Bias => rc.limits().bias,
        Knob::Tc => rc.limits().tc,
        Knob::Abs => rc.limits().abs,
    };
    if k == Knob::Bias {
        let l = lim.unwrap_or(Limit { min: 0.0, max: 100.0, modifiable: false });
        if !unlocked() && (!l.modifiable || l.min == l.max) {
            return Outcome::Locked;
        }
        // thousandths, as the game steps it
        let mut b = (rc.brake_bias() * 1000.0).round() as i32 + 2 * dir.signum();
        if !unlocked() && l.max >= 0.0 {
            b = b.clamp((l.min * 10.0).round() as i32, (l.max * 10.0).round() as i32);
        }
        rc.set_brake_bias(b.clamp(0, 1000) as f32 / 1000.0);
        return Outcome::Changed;
    }
    match level_range(rc, k) {
        Ok((lo, hi)) => {
            let cur = level(rc, k).unwrap_or(0) as i32;
            set_level(rc, k, (cur + dir.signum()).clamp(lo, hi));
            Outcome::Changed
        }
        Err(o) => o,
    }
}

/// The levels a press may reach, as the game clamps them (and never past
/// the car's own table).
fn level_range(rc: &crate::rigcar::RigCar, k: Knob) -> Result<(i32, i32), Outcome> {
    let (n, lim) = match k {
        Knob::Tc => (rc.tc.as_ref().map(|t| t.levels.len()), rc.limits().tc),
        _ => (rc.abs.as_ref().map(|a| a.levels.len()), rc.limits().abs),
    };
    let Some(n) = n.filter(|n| *n > 0).map(|n| n as i32) else { return Err(Outcome::None) };
    if unlocked() {
        return Ok((0, n - 1));
    }
    let l = lim.unwrap_or(Limit { min: 0.0, max: 0.0, modifiable: false });
    if !l.modifiable || l.min == l.max {
        return Err(Outcome::Locked);
    }
    let lo = (l.min as i32).clamp(0, n - 1);
    Ok((lo, (l.max as i32).clamp(lo, n - 1)))
}

fn set_level(rc: &mut crate::rigcar::RigCar, k: Knob, v: i32) {
    let v = v.max(0) as usize;
    match k {
        Knob::Tc => {
            if let Some(t) = rc.tc.as_mut() {
                t.level = v.min(t.levels.len().saturating_sub(1));
            }
            rc.tc_on = true;
        }
        _ => {
            if let Some(a) = rc.abs.as_mut() {
                a.level = v.min(a.levels.len().saturating_sub(1));
            }
            rc.abs_on = true;
        }
    }
}

/// The car's id from its CarData path (content\cars\<id>\...).
pub fn car_id(rc: &crate::rigcar::RigCar) -> String {
    rc.source().split(['\\', '/']).nth(2).unwrap_or("car").to_string()
}

fn cfg_path() -> std::path::PathBuf {
    let base = crate::envc::var_os("APPDATA").map(std::path::PathBuf::from).unwrap_or_else(|| ".".into());
    base.join("evoview").join("aids.cfg")
}

/// The last values per car, `<car> <bias %> <tc> <abs>` in
/// %APPDATA%/evoview/aids.cfg (next to wheel.cfg / ffb.cfg).
fn saved(car: &str) -> Option<(f32, i32, i32)> {
    let text = std::fs::read_to_string(cfg_path()).ok()?;
    text.lines().find_map(|l| {
        let mut it = l.split_whitespace();
        (it.next()? == car).then_some(())?;
        Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?, it.next()?.parse().ok()?))
    })
}

/// Remember this car's values (after a press).
pub fn save(rc: &crate::rigcar::RigCar) {
    let car = car_id(rc);
    let p = cfg_path();
    let mut lines: Vec<String> = std::fs::read_to_string(&p).unwrap_or_default().lines()
        .filter(|l| !l.is_empty() && l.split_whitespace().next() != Some(car.as_str()))
        .map(str::to_string).collect();
    let lv = |k| level(rc, k).map_or(-1, |l| l as i32);
    lines.push(format!("{car} {:.1} {} {}", rc.brake_bias() * 100.0, lv(Knob::Tc), lv(Knob::Abs)));
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let _ = std::fs::write(&p, lines.join("\n") + "\n");
}

/// At load: the saved values for this car (clamped to its limits), then the
/// test overrides EVOVIEW_BIAS (front %), EVOVIEW_TC, EVOVIEW_ABS (levels;
/// these skip the limits). EVOVIEW_AIDS_FRESH=1 ignores the saved file, and
/// so does a player setup (`use_saved` false: its own bias / TC / ABS win).
pub fn startup(rc: &mut crate::rigcar::RigCar, use_saved: bool) {
    let car = car_id(rc);
    if let Some((b, tc, abs)) = saved(&car).filter(|_| use_saved && crate::envc::var_os("EVOVIEW_AIDS_FRESH").is_none()) {
        let l = rc.limits().bias;
        if l.map_or(false, |l| l.modifiable && l.max > l.min) {
            let l = l.unwrap();
            rc.set_brake_bias(b.clamp(l.min, l.max) * 0.01);
        }
        for (k, v) in [(Knob::Tc, tc), (Knob::Abs, abs)] {
            if let Ok((lo, hi)) = level_range(rc, k) {
                set_level(rc, k, v.clamp(lo, hi));
            }
        }
    }
    let env = |k: &str| crate::envc::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    if let Some(b) = env("EVOVIEW_BIAS") {
        rc.set_brake_bias((b * 0.01).clamp(0.0, 1.0));
    }
    for (k, name) in [(Knob::Tc, "EVOVIEW_TC"), (Knob::Abs, "EVOVIEW_ABS")] {
        if let Some(v) = env(name) {
            set_level(rc, k, v as i32);
        }
    }
    let l = rc.limits().clone();
    let r = |l: Option<Limit>| l.map_or("no limits".to_string(), |l| format!("{}-{}{}", l.min, l.max, if l.modifiable { "" } else { " locked" }));
    println!("aids: brake bias {:.1} % front ({}), TC {} of {} ({}), ABS {} of {} ({})",
        rc.brake_bias() * 100.0, r(l.bias),
        level(rc, Knob::Tc).map_or("-".into(), |v| v.to_string()), rc.tc.as_ref().map_or(0, |t| t.levels.len().saturating_sub(1)), r(l.tc),
        level(rc, Knob::Abs).map_or("-".into(), |v| v.to_string()), rc.abs.as_ref().map_or(0, |a| a.levels.len().saturating_sub(1)), r(l.abs));
}

/// The HUD's pop-up after a press, like the game's CHANGED_ / LOCKED_
/// notification: which value, seconds since, red when locked.
#[derive(Clone, Copy, Debug)]
pub struct Pop {
    pub knob: Knob,
    pub age: f32,
    pub locked: bool,
}

/// How long the pop-up and the panel highlight stay.
pub const POP_SECS: f32 = 1.5;

/// A press from a key or wheel button: step, say it, remember it.
/// `keep` = save to aids.cfg (not for scripted presses).
pub fn press(rc: &mut crate::rigcar::RigCar, k: Knob, dir: i32, keep: bool) -> Option<Pop> {
    let o = step(rc, k, dir);
    match o {
        Outcome::None => {
            println!("aids: this car has no {}", k.name());
            return None;
        }
        Outcome::Locked => println!("aids: {} is locked on this car", k.name()),
        Outcome::Changed => {
            println!("aids: {} {}", k.name(), value(rc, k));
            if keep {
                save(rc);
            }
        }
    }
    Some(Pop { knob: k, age: 0.0, locked: o == Outcome::Locked })
}

/// What the HUD shows: bias as front %, a level, "oFF" for level 0.
pub fn value(rc: &crate::rigcar::RigCar, k: Knob) -> String {
    match k {
        Knob::Bias => format!("{:.1}", rc.brake_bias() * 100.0),
        _ => match level(rc, k) {
            None => "-".into(),
            Some(0) => "oFF".into(),
            Some(l) => l.to_string(),
        },
    }
}

/// The panel and the pop-up (upper middle), in ACECM's look (hud::theme).
/// The panel matches the driving cluster: the same see-through square
/// style, its height, just left of it (`next_to` = the cluster's box).
/// `aspect` = width / height.
pub fn draw(hud: &mut crate::hud::Hud, rc: &crate::rigcar::RigCar, pop: Option<Pop>, aspect: f32,
            next_to: (f32, f32, f32, f32)) {
    use crate::hud::{theme as t, Weight};
    let live = pop.filter(|p| p.age < POP_SECS);
    let shown = |k: Knob| match value(rc, k).as_str() {
        "oFF" => "Off".to_string(),
        v if k == Knob::Bias => format!("{v}%"),
        v => v.to_string(),
    };
    let colour = |k: Knob, base: [f32; 4]| match live {
        Some(p) if p.knob == k && p.locked => t::red(1.0),
        Some(p) if p.knob == k => t::gold(1.0),
        _ => base,
    };
    let labels = [(Knob::Bias, "Brake bias"), (Knob::Tc, "TC"), (Knob::Abs, "ABS")];
    let u = |v: f32| v / aspect;
    let (cx, cy, _, ch) = next_to;
    let h = ch;
    let w = u(0.13);
    let (x, y0) = ((cx - w - u(0.008)).max(0.0), cy);
    hud.panel(x, y0, w, h, 10.0, t::panel(0.42), t::line(0.45));
    let pad = 0.013;
    let (ix, iw) = (x + u(pad), w - u(pad) * 2.0);
    let row = (h - pad * 2.0) / 3.0;
    let (ls, vs) = (0.0105, 0.028);
    for (i, (k, label)) in labels.iter().enumerate() {
        let y = y0 + pad + row * i as f32;
        if live.map_or(false, |p| p.knob == *k) {
            hud.round_rect(x + u(0.005), y - 0.003, w - u(0.01), row, 6.0, t::panel2(0.85));
        }
        hud.label(label, ix, y, ls, Weight::Regular, t::dim(1.0));
        hud.label(&shown(*k), ix, y + hud.line_h(ls) - 0.004, vs, Weight::Semibold, colour(*k, t::fg(1.0)));
    }
    // the split under the bias: front share filled
    let by = y0 + pad + hud.line_h(ls) - 0.004 + hud.line_h(vs) - 0.002;
    let bh = hud.px_y(4.0);
    hud.round_rect(ix, by, iw, bh, 2.0, t::well(0.75));
    hud.round_rect(ix, by, iw * rc.brake_bias(), bh, 2.0, colour(Knob::Bias, t::accent(0.95)));
    // the pop-up: the changed value, big, fading out over its last 0.4 s
    if let Some(p) = live {
        let a = ((POP_SECS - p.age) / 0.4).clamp(0.0, 1.0);
        let big = 0.03;
        let label = labels.iter().find(|l| l.0 == p.knob).map_or("", |l| l.1);
        let v = if p.locked { format!("{} (locked)", shown(p.knob)) } else { shown(p.knob) };
        let lw = hud.label_w(label, big * 0.6, Weight::Regular);
        let vw = hud.label_w(&v, big, Weight::Semibold);
        let gap = 0.012 / aspect;
        let (padx, pady) = (0.02 / aspect, 0.012);
        let pw = lw + gap + vw + padx * 2.0;
        let ph = hud.line_h(big) + pady * 2.0;
        let (px, py) = (0.5 - pw * 0.5, 0.2);
        hud.panel(px, py, pw, ph, 8.0, t::panel(0.92 * a), t::line(a));
        hud.label(label, px + padx, py + pady + (hud.line_h(big) - hud.line_h(big * 0.6)) * 0.62, big * 0.6, Weight::Regular, t::dim(a));
        let mut c = colour(p.knob, t::fg(1.0));
        c[3] *= a;
        hud.label(&v, px + padx + lw + gap, py + pady, big, Weight::Semibold, c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_limits() {
        // {1 {3 {1 {1 0.2, 2 50, 3 80, 4 1}}}, 5 {1 {1 1, 3 12, 4 1}}}
        let fb: Vec<u8> = [vec![0x0d], 0.2f32.to_le_bytes().to_vec(), vec![0x15], 50f32.to_le_bytes().to_vec(),
                           vec![0x1d], 80f32.to_le_bytes().to_vec(), vec![0x20, 1]].concat();
        let brakes = [vec![0x0a, fb.len() as u8], fb].concat();
        let mb = [vec![0x1a, brakes.len() as u8], brakes].concat();
        let tc: Vec<u8> = [vec![0x0d], 1f32.to_le_bytes().to_vec(), vec![0x1d], 12f32.to_le_bytes().to_vec(), vec![0x20, 1]].concat();
        let el = [vec![0x0a, tc.len() as u8], tc].concat();
        let d = [vec![0x0a, mb.len() as u8], mb, vec![0x2a, el.len() as u8], el].concat();
        let l = Limits::read(&d);
        let b = l.bias.unwrap();
        assert_eq!((b.min, b.max, b.modifiable), (50.0, 80.0, true));
        let t = l.tc.unwrap();
        assert_eq!((t.min, t.max, t.modifiable), (0.0, 12.0, true));
        assert!(l.abs.is_none());
    }
}
