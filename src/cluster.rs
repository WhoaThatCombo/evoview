//! The driving cluster: gear, speed, shift lights, pedals and the
//! handbrake in one small square see-through panel (bottom right), in ACECM's look (hud::theme). Drag it with
//! the mouse; ` shows / hides it, Shift+` puts it back. Where it sits is
//! kept in %APPDATA%/evoview/hud.cfg.

use crate::hud::{theme as t, Hud, Weight};

/// One frame's numbers.
pub struct Reading {
    pub kmh: f32,
    pub rpm: f32,
    pub limiter: f32,
    /// -1 reverse, 0 neutral
    pub gear: i32,
    pub throttle: f32,
    pub brake: f32,
    /// the clutch pedal 0..1 (the car's own clutch: 1 - its engagement)
    pub clutch: f32,
    pub handbrake: f32,
    pub stalled: bool,
    pub on_limiter: bool,
    /// rpm and gear are known (a replay can lack them for other cars)
    pub engine: bool,
    /// clutch and handbrake are known (not in replays)
    pub aux: bool,
}

pub struct Cluster {
    /// top-left, 0..1 screen
    pub x: f32,
    pub y: f32,
    pub on: bool,
    /// the last drawn box, for the mouse
    rect: (f32, f32, f32, f32),
    /// grab offset while dragging
    drag: Option<(f32, f32)>,
}

fn cfg() -> Option<std::path::PathBuf> {
    std::env::var_os("APPDATA").map(|a| std::path::Path::new(&a).join("evoview").join("hud.cfg"))
}

impl Cluster {
    /// From hud.cfg ("cluster2 x y on"), else bottom right.
    pub fn load() -> Cluster {
        let mut c = Cluster { x: f32::NAN, y: 0.0, on: true, rect: (0.0, 0.0, 0.0, 0.0), drag: None };
        if let Some(s) = cfg().and_then(|p| std::fs::read_to_string(p).ok()) {
            for l in s.lines() {
                let w: Vec<&str> = l.split_whitespace().collect();
                if let ["cluster2", x, y, on] = w[..] {
                    if let (Ok(x), Ok(y)) = (x.parse::<f32>(), y.parse::<f32>()) {
                        c.x = x.clamp(0.0, 0.98);
                        c.y = y.clamp(0.0, 0.98);
                    }
                    c.on = on != "0";
                }
            }
        }
        c
    }

    /// Shown, placed by its owner every frame (the replay overlay), not
    /// read from or saved to hud.cfg.
    pub fn fixed() -> Cluster {
        Cluster { x: f32::NAN, y: 0.0, on: true, rect: (0.0, 0.0, 0.0, 0.0), drag: None }
    }

    fn save(&self) {
        let Some(p) = cfg() else { return };
        let _ = std::fs::create_dir_all(p.parent().unwrap());
        let _ = std::fs::write(p, format!("cluster {:.4} {:.4} {}\n", self.x, self.y, self.on as u8));
    }

    /// Where the cluster sits (the default corner before its first frame),
    /// for the panels placed beside it.
    pub fn rect(&self, aspect: f32) -> (f32, f32, f32, f32) {
        if self.rect.2 > 0.0 {
            return self.rect;
        }
        let (w, h) = (0.2 / aspect, 0.2);
        (1.0 - w - 0.02 / aspect, 1.0 - h - 0.025, w, h)
    }

    pub fn toggle(&mut self) {
        self.on = !self.on;
        self.save();
    }

    /// Back to the default place.
    pub fn reset(&mut self) {
        self.x = f32::NAN;
        self.on = true;
        self.save();
    }

    /// A left press at (x, y): true if it grabbed the cluster.
    pub fn press(&mut self, x: f32, y: f32) -> bool {
        let (rx, ry, rw, rh) = self.rect;
        if self.on && x >= rx && x <= rx + rw && y >= ry && y <= ry + rh {
            self.drag = Some((x - rx, y - ry));
            return true;
        }
        false
    }

    /// The cursor moved: true while dragging (the camera must not orbit).
    pub fn moved(&mut self, x: f32, y: f32) -> bool {
        let Some((ox, oy)) = self.drag else { return false };
        let (_, _, rw, rh) = self.rect;
        self.x = (x - ox).clamp(0.0, 1.0 - rw);
        self.y = (y - oy).clamp(0.0, 1.0 - rh);
        true
    }

    pub fn release(&mut self) -> bool {
        if self.drag.take().is_some() {
            self.save();
            return true;
        }
        false
    }

    pub fn draw(&mut self, hud: &mut Hud, r: &Reading, aspect: f32) {
        if !self.on {
            return;
        }
        // a square, see-through panel; sizes in screen-height units, widths
        // through the aspect
        let u = |v: f32| v / aspect;
        let side = 0.2;
        let (w, h) = (u(side), side);
        if !self.x.is_finite() {
            self.x = 1.0 - w - u(0.02);
            self.y = 1.0 - h - 0.025;
        }
        let (x, y) = (self.x.clamp(0.0, 1.0 - w), self.y.clamp(0.0, 1.0 - h));
        self.rect = (x, y, w, h);
        let dragging = self.drag.is_some();
        hud.panel(x, y, w, h, 10.0, t::panel(if dragging { 0.7 } else { 0.42 }), t::line(if dragging { 1.0 } else { 0.45 }));
        let pad = 0.013;
        let (ix, iy, iw, ih) = (x + u(pad), y + pad, w - u(pad) * 2.0, h - pad * 2.0);

        // shift lights: 10 segments from 75 % of the limiter, green, gold,
        // red; all red and flashing on the limiter
        let n = 10;
        let f = ((r.rpm / r.limiter.max(1.0) - 0.75) / 0.23).clamp(0.0, 1.0);
        let lit = if r.engine { (f * n as f32).round() as usize } else { 0 };
        let flash = r.on_limiter && (hud_time() * 8.0).fract() < 0.5;
        let gap = u(0.003);
        let sw = (iw - gap * (n - 1) as f32) / n as f32;
        let sh = 0.009;
        for i in 0..n {
            let c = if flash {
                t::red(1.0)
            } else if i < lit {
                if i < 4 { t::ok(1.0) } else if i < 7 { t::gold(1.0) } else { t::red(1.0) }
            } else {
                t::well(0.7)
            };
            hud.round_rect(ix + i as f32 * (sw + gap), iy, sw, sh, 2.0, c);
        }
        let top = iy + sh + 0.01;
        let bottom = iy + ih;

        // pedals down the right: clutch, brake, throttle, filling upwards
        let ls = 0.0105;
        let pw = u(0.012);
        let pg = u(0.007);
        let pedals: Vec<(f32, [f32; 4], &str)> = if r.aux {
            vec![(r.clutch, t::hex(0x6ea8e0, 1.0), "C"), (r.brake, t::red(1.0), "B"), (r.throttle, t::ok(1.0), "T")]
        } else {
            vec![(r.brake, t::red(1.0), "B"), (r.throttle, t::ok(1.0), "T")]
        };
        let np = pedals.len() as f32;
        let pcol = pw * np + pg * (np - 1.0);
        let px0 = ix + iw - pcol;
        let ph = bottom - top - hud.line_h(ls);
        for (i, (v, c, l)) in pedals.into_iter().enumerate() {
            let bx = px0 + i as f32 * (pw + pg);
            hud.round_rect(bx, top, pw, ph, 2.0, t::well(0.75));
            let v = v.clamp(0.0, 1.0);
            if v > 0.005 {
                hud.round_rect(bx, top + ph * (1.0 - v), pw, ph * v, 2.0, c);
            }
            let lw = hud.label_w(l, ls, Weight::Regular);
            hud.label(l, bx + (pw - lw) * 0.5, top + ph + 0.002, ls, Weight::Regular, t::dim(1.0));
        }

        // the left column, centred: gear, speed, rpm, the handbrake lamp
        let cw = iw - pcol - u(0.012);
        let centre = |hud: &mut Hud, s: &str, yy: f32, size: f32, wt: Weight, c: [f32; 4]| {
            let tw = hud.label_w(s, size, wt);
            hud.label(s, ix + (cw - tw) * 0.5, yy, size, wt, c);
        };
        let gs = 0.072;
        let gear = match r.gear {
            _ if !r.engine => "–".to_string(),
            g if g < 0 => "R".to_string(),
            0 => "N".to_string(),
            g => g.to_string(),
        };
        let gc = if r.stalled || !r.engine { t::dim2(1.0) } else if flash { t::red(1.0) } else { t::fg(1.0) };
        // digits sit ~0.22 em below the line's top: pull the gear up to the lights
        centre(hud, &gear, top - gs * 0.22, gs, Weight::Semibold, gc);
        let mph = mph();
        let v = if mph { r.kmh.abs() / 1.609_344 } else { r.kmh.abs() };
        let ss = 0.034;
        let sy = top + gs * 0.86;
        centre(hud, &format!("{v:.0}"), sy, ss, Weight::Semibold, t::fg(1.0));
        let ry = sy + hud.line_h(ss) - 0.006;
        let unit = if mph { "mph" } else { "km/h" };
        let rs = if r.stalled {
            "Stalled".to_string()
        } else if r.engine {
            format!("{unit}   {:.0} rpm", r.rpm)
        } else {
            unit.to_string()
        };
        centre(hud, &rs, ry, ls, Weight::Regular, if r.stalled { t::red(1.0) } else { t::dim(1.0) });
        if !r.aux {
            return;
        }
        let hb = r.handbrake > 0.05;
        let hh = hud.line_h(ls) + 0.003;
        let hy = bottom - hh;
        let hw = hud.label_w("Handbrake", ls, Weight::Semibold) + u(0.014);
        hud.round_rect(ix + (cw - hw) * 0.5, hy, hw, hh, 3.0, if hb { t::gold(1.0) } else { t::well(0.6) });
        centre(hud, "Handbrake", hy + 0.0015, ls, Weight::Semibold, if hb { t::hex(0x1a1e25, 1.0) } else { t::dim2(1.0) });
    }
}

/// EVOVIEW_SPEED_UNITS=mph: speeds in mph (the cluster, the pit HUD).
pub fn mph() -> bool {
    use std::sync::OnceLock;
    static M: OnceLock<bool> = OnceLock::new();
    *M.get_or_init(|| std::env::var("EVOVIEW_SPEED_UNITS").map_or(false, |v| v.eq_ignore_ascii_case("mph")))
}

/// A speed in the chosen units, e.g. "58 km/h".
pub fn speed(kmh: f32) -> String {
    if mph() { format!("{:.0} mph", kmh / 1.609_344) } else { format!("{kmh:.0} km/h") }
}

/// Seconds since start, for the limiter flash.
fn hud_time() -> f32 {
    use std::sync::OnceLock;
    static T0: OnceLock<std::time::Instant> = OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_secs_f32()
}
