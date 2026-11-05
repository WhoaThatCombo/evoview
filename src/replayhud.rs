//! The replay overlay: a control bar across the bottom (timeline you can
//! click or drag, transport, playback speed, the watched car, the camera),
//! a car list down the left, and the driving cluster for the watched car.
//! Everything but the cluster fades out after a few seconds without the
//! mouse moving, so a replay left playing is a clean picture.
//!
//! Keys: Space play / pause, Left / Right 5 s back / on (Shift 30 s),
//! Up / Down speed, [ ] car, C camera, Tab car list, ` hide the overlay.

use crate::hud::{theme as t, Action, Hud, Weight};
use crate::replay::Replay;

/// Seconds without the mouse moving before the controls fade.
const IDLE: f32 = 2.5;

pub struct ReplayUi {
    /// ` : the whole overlay on / off
    pub show: bool,
    /// Tab : the car list
    pub list: bool,
    /// seconds since the mouse last moved
    idle: f32,
    /// 0..1, the controls' fade
    alpha: f32,
    /// the playhead is being dragged
    pub scrub: bool,
    /// the timeline track (x, width), for dragging
    track: (f32, f32),
    /// the car list's box, for the wheel
    list_rect: (f32, f32, f32, f32),
    /// first row shown, and how many fit
    scroll: usize,
    rows: usize,
    /// last watched car, to keep it in view when it changes
    last_watched: usize,
    /// per replay car: display name, the rpm its lights count up to
    names: Vec<String>,
    limits: Vec<f32>,
    /// whether its rpm / gear were recorded (they move)
    engine: Vec<bool>,
    /// the replay car that recorded it (the only one with wheel speed)
    you: Option<usize>,
}

/// "ks_audi_r8_lms_gt3_evo_2" -> "Audi R8 LMS GT3 Evo 2".
pub fn pretty(model: &str) -> String {
    let m = model.strip_prefix("ks_").unwrap_or(model);
    m.split('_')
        .filter(|w| !w.is_empty())
        .map(|w| {
            if w.chars().any(|c| c.is_ascii_digit()) || (w.len() <= 3 && !matches!(w, "evo" | "the" | "pro")) {
                w.to_uppercase()
            } else {
                let mut c = w.chars();
                c.next().map_or(String::new(), |f| f.to_uppercase().collect::<String>() + c.as_str())
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// 83.4 -> "1:23.4"
pub fn clock(s: f32) -> String {
    let s = s.max(0.0);
    let m = (s / 60.0).floor();
    format!("{}:{:04.1}", m as u32, s - m * 60.0)
}

fn speed_label(s: f32) -> String {
    if s >= 1.0 {
        format!("{}×", s.round() as u32)
    } else {
        let v = format!("{s:.3}");
        format!("{}×", v.trim_end_matches('0'))
    }
}

impl ReplayUi {
    /// `drivers`: (replay car, model) in list order.
    pub fn new(rep: &Replay, drivers: &[(usize, String)]) -> ReplayUi {
        let span = rep.captured().max(0.001);
        let n = rep.cars.len();
        let mut names = vec![String::new(); n];
        let mut limits = vec![8000.0f32; n];
        let mut engine = vec![false; n];
        let mut you = None;
        for (ci, model) in drivers {
            let ci = *ci;
            if ci >= n {
                continue;
            }
            names[ci] = pretty(model);
            // the highest rpm the car reaches is near enough its limiter
            // for the shift lights; only the recording car has wheel speed
            let mut top = 0.0f32;
            let mut low = f32::MAX;
            let mut spin = 0.0f32;
            for k in 0..400 {
                if let Some(tl) = rep.telemetry(ci, span * k as f32 / 400.0) {
                    top = top.max(tl.rpm);
                    low = low.min(tl.rpm);
                    spin = spin.max(tl.speed_rads.abs());
                }
            }
            // some replays hold one frozen engine frame for the other cars
            engine[ci] = top - low > 50.0;
            if top > 1000.0 {
                limits[ci] = top;
            }
            if spin > 1.0 && you.is_none() {
                you = Some(ci);
            }
        }
        ReplayUi {
            show: true,
            list: drivers.len() > 1,
            idle: 0.0,
            alpha: 1.0,
            scrub: false,
            track: (0.0, 1.0),
            list_rect: (0.0, 0.0, 0.0, 0.0),
            scroll: 0,
            rows: 1,
            last_watched: usize::MAX,
            names,
            limits,
            engine,
            you,
        }
    }

    /// The mouse moved, or a replay key was pressed: bring the controls back.
    pub fn wake(&mut self) {
        self.idle = 0.0;
    }

    /// Whether clicks should reach the controls (they are not faded out).
    pub fn live(&self) -> bool {
        self.show && self.alpha > 0.05
    }

    /// The timeline fraction under cursor x (while dragging).
    pub fn seek_at(&self, x: f32) -> f32 {
        ((x - self.track.0) / self.track.1.max(1e-6)).clamp(0.0, 1.0)
    }

    /// The wheel over the car list scrolls it (true = used).
    pub fn wheel(&mut self, cur: Option<(f32, f32)>, dy: f32, cars: usize) -> bool {
        let (x, y, w, h) = self.list_rect;
        let Some((cx, cy)) = cur else { return false };
        if !(self.live() && self.list && cx >= x && cx <= x + w && cy >= y && cy <= y + h) {
            return false;
        }
        let max = cars.saturating_sub(self.rows);
        let step = if dy > 0.0 { -1 } else { 1 } * dy.abs().ceil().max(1.0) as isize;
        self.scroll = (self.scroll as isize + step).clamp(0, max as isize) as usize;
        self.wake();
        true
    }

    pub fn name(&self, ci: usize) -> String {
        let n = self.names.get(ci).cloned().unwrap_or_default();
        if self.you == Some(ci) { format!("{n} (you)") } else { n }
    }

    /// The watched car's cluster reading.
    pub fn reading(&self, rep: &Replay, ci: usize, t: f32) -> Option<crate::cluster::Reading> {
        let tl = rep.telemetry(ci, t)?;
        let lim = self.limits.get(ci).copied().unwrap_or(8000.0);
        // frames where a car was not in the session read rpm 0, gear 0
        let engine = self.engine.get(ci).copied().unwrap_or(false) && (tl.rpm > 1.0 || tl.gear > 0);
        Some(crate::cluster::Reading {
            kmh: rep.speed(ci, t) * 3.6,
            rpm: tl.rpm,
            limiter: lim,
            // stored + 1: 0 reverse, 1 neutral
            gear: tl.gear as i32 - 1,
            throttle: tl.throttle,
            brake: tl.brake,
            clutch: 0.0,
            handbrake: 0.0,
            stalled: false,
            on_limiter: engine && tl.rpm > lim * 0.985,
            engine,
            aux: false,
        })
    }

    /// Build this frame's overlay. `cur`: the cursor, 0..1 screen.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&mut self, hud: &mut Hud, cluster: &mut crate::cluster::Cluster, rep: &Replay,
                drivers: &[(usize, String)], watched: usize, play_t: f32, playing: bool,
                speed: f32, camera: &str, cur: Option<(f32, f32)>, aspect: f32, dt: f32) {
        if !self.show {
            return;
        }
        let u = |v: f32| v / aspect;
        let span = rep.captured().max(0.001);
        let m = 0.022;

        // bar geometry first: whether the cursor is over it keeps it awake
        let fs = 0.0155;
        let pad = 0.012;
        let row1 = 0.026;
        let row2 = 0.046;
        let bh = pad + row1 + 0.006 + row2 + pad;
        let (bx, bw) = (u(m), 1.0 - 2.0 * u(m));
        let by = 1.0 - m - bh;
        let over_bar = cur.map_or(false, |(cx, cy)| cx >= bx && cx <= bx + bw && cy >= by - 0.02);
        let over_list = cur.map_or(false, |(cx, cy)| {
            let (x, y, w, h) = self.list_rect;
            self.list && cx >= x && cx <= x + w && cy >= y && cy <= y + h
        });
        self.idle += dt;
        let awake = self.idle < IDLE || !playing || self.scrub || over_bar || over_list;
        let target = if awake { 1.0 } else { 0.0 };
        let rate = if awake { 8.0 } else { 2.5 };
        self.alpha += (target - self.alpha) * (1.0 - (-rate * dt.max(0.0)).exp());
        if self.alpha < 0.004 {
            self.alpha = 0.0;
        }
        let a = self.alpha;
        let live = a > 0.05;

        // the watched car's cluster: above the bar while it shows, in the
        // corner once it fades
        if let Some(&(ci, _)) = drivers.get(watched) {
            if let Some(r) = self.reading(rep, ci, play_t) {
                let (_, _, cw, ch) = cluster.rect(aspect);
                let low = 1.0 - ch - 0.025;
                let high = by - ch - 0.014;
                cluster.x = 1.0 - cw - u(m);
                cluster.y = low + (high - low) * a;
                cluster.draw(hud, &r, aspect);
            }
        }
        if a <= 0.0 {
            return;
        }

        // ---- car list ----
        if self.list && drivers.len() > 0 {
            self.draw_list(hud, rep, drivers, watched, play_t, cur, aspect, a, live, by - 0.02);
        }

        // ---- the bar ----
        hud.panel(bx, by, bw, bh, 10.0, t::panel(0.78 * a), t::line(0.6 * a));
        let (ix, iw) = (bx + u(0.016), bw - u(0.032));

        // row 1: elapsed, the track, length
        let ty = by + pad;
        let el = clock(play_t);
        let tot = clock(span);
        let tw = hud.label_w("00:00.0", fs, Weight::Semibold);
        let lw = hud.label_w(&el, fs, Weight::Semibold);
        hud.label(&el, ix + tw - lw, ty + (row1 - hud.line_h(fs)) * 0.5, fs, Weight::Semibold, t::fg(a));
        let rw = hud.label_w(&tot, fs, Weight::Regular);
        hud.label(&tot, ix + iw - rw, ty + (row1 - hud.line_h(fs)) * 0.5, fs, Weight::Regular, t::dim(a));
        let gx = ix + tw + u(0.014);
        let gw = (ix + iw - tw - u(0.014)) - gx;
        self.track = (gx, gw);
        let hot_track = cur.map_or(false, |(cx, cy)| cx >= gx - u(0.006) && cx <= gx + gw + u(0.006) && cy >= ty && cy <= ty + row1);
        let th = hud.px_y(if hot_track || self.scrub { 6.0 } else { 4.0 });
        let tyc = ty + row1 * 0.5;
        hud.round_rect(gx, tyc - th * 0.5, gw, th, 3.0, t::well(0.95 * a));
        let f = (play_t / span).clamp(0.0, 1.0);
        // hover: where a click would land, and its time
        if let Some((cx, _)) = cur.filter(|_| hot_track && live) {
            let hf = ((cx - gx) / gw).clamp(0.0, 1.0);
            hud.round_rect(gx, tyc - th * 0.5, gw * hf, th, 3.0, t::fg(0.18 * a));
            let s = clock(hf * span);
            let sw = hud.label_w(&s, 0.012, Weight::Semibold) + u(0.012);
            let sx = (gx + gw * hf - sw * 0.5).clamp(bx, bx + bw - sw);
            let sh = hud.line_h(0.012) + 0.006;
            hud.panel(sx, by - sh - 0.008, sw, sh, 5.0, t::panel2(0.95 * a), t::line(a));
            hud.label(&s, sx + u(0.006), by - sh - 0.008 + 0.003, 0.012, Weight::Semibold, t::fg(a));
        }
        hud.round_rect(gx, tyc - th * 0.5, gw * f, th, 3.0, t::accent(a));
        let kn = if hot_track || self.scrub { 14.0 } else { 11.0 };
        let (kw, kh) = (hud.px_x(kn), hud.px_y(kn));
        hud.round_rect(gx + gw * f - kw * 0.5, tyc - kh * 0.5, kw, kh, kn * 0.5, t::fg(a));
        if live {
            hud.hotspot(gx - u(0.006), ty, gw + u(0.012), row1, Action::Seek(0.0));
        }

        // row 2: the car (left), transport + speed centred on the screen,
        // camera + list (right)
        let ry = ty + row1 + 0.006;
        let s = row2;
        let gap = u(0.004);
        let pw = u(s * 1.25);
        let sw = u(s * 0.8);
        let slw = hud.label_w("0.125×", fs, Weight::Semibold);
        // back / play / forward, the play button on the screen's centre line
        let group = 2.0 * u(s) + pw + 2.0 * gap;
        let group_x = 0.5 - group * 0.5;
        let mut x = group_x;
        let mut tips: Vec<(f32, f32, &'static str)> = Vec::new();
        let btn = |hud: &mut Hud, x: f32, w: f32, act: Action, primary: bool, on: bool, tip: &'static str,
                   tips: &mut Vec<(f32, f32, &'static str)>| -> bool {
            let hot = live && cur.map_or(false, |(cx, cy)| cx >= x && cx <= x + w && cy >= ry && cy <= ry + s);
            let fill = if primary {
                t::accent(if hot { 1.0 } else { 0.88 } * a)
            } else if on {
                t::panel2(a)
            } else if hot {
                t::panel2(0.95 * a)
            } else {
                t::well(0.0)
            };
            hud.round_rect(x, ry, w, s, 7.0, fill);
            if on && !primary {
                hud.round_rect(x, ry + s - hud.px_y(2.0), w, hud.px_y(2.0), 1.0, t::accent(a));
            }
            if live {
                hud.hotspot(x, ry, w, s, act);
            }
            if hot {
                tips.push((x + w * 0.5, ry, tip));
            }
            hot
        };
        let ic = t::fg(a);
        // icons inside a square of side `s`
        let icon_box = |x: f32| (x + u(s) * 0.3, ry + s * 0.3, u(s) * 0.4, s * 0.4);

        btn(hud, x, u(s), Action::Back, false, false, "Back 5 s  (Left, Shift 30 s)", &mut tips);
        let (ix0, iy0, w0, h0) = icon_box(x);
        hud.tri([(ix0 + w0 * 0.5, iy0), (ix0, iy0 + h0 * 0.5), (ix0 + w0 * 0.5, iy0 + h0)], ic);
        hud.tri([(ix0 + w0, iy0), (ix0 + w0 * 0.5, iy0 + h0 * 0.5), (ix0 + w0, iy0 + h0)], ic);
        x += u(s) + u(0.004);

        btn(hud, x, pw, Action::PlayPause, true, false, if playing { "Pause  (Space)" } else { "Play  (Space)" }, &mut tips);
        let (cx0, cy0) = (x + pw * 0.5, ry + s * 0.5);
        let (hw, hh) = (u(s) * 0.17, s * 0.2);
        if playing {
            let b = hw * 0.62;
            hud.rect(cx0 - hw, cy0 - hh, b, hh * 2.0, ic);
            hud.rect(cx0 + hw - b, cy0 - hh, b, hh * 2.0, ic);
        } else {
            hud.tri([(cx0 - hw * 0.8, cy0 - hh), (cx0 + hw * 1.2, cy0), (cx0 - hw * 0.8, cy0 + hh)], ic);
        }
        x += pw + u(0.004);

        btn(hud, x, u(s), Action::Fwd, false, false, "Forward 5 s  (Right, Shift 30 s)", &mut tips);
        let (ix0, iy0, w0, h0) = icon_box(x);
        hud.tri([(ix0, iy0), (ix0 + w0 * 0.5, iy0 + h0 * 0.5), (ix0, iy0 + h0)], ic);
        hud.tri([(ix0 + w0 * 0.5, iy0), (ix0 + w0, iy0 + h0 * 0.5), (ix0 + w0 * 0.5, iy0 + h0)], ic);

        // right: the car list and the camera
        let mut rx = ix + iw;
        if drivers.len() > 1 {
            let lw2 = hud.label_w("Cars", fs, Weight::Semibold) + u(0.044);
            rx -= lw2;
            btn(hud, rx, lw2, Action::ToggleList, false, self.list, "Car list  (Tab)", &mut tips);
            // three lines
            let (lx, ly) = (rx + u(0.013), ry + s * 0.5);
            for k in -1..=1 {
                hud.rect(lx, ly + k as f32 * 0.0065 - hud.px_y(1.0), u(0.012), hud.px_y(2.0), ic);
            }
            hud.label("Cars", rx + u(0.032), ry + (s - hud.line_h(fs)) * 0.5, fs, Weight::Semibold, t::fg(a));
            rx -= u(0.006);
        }
        let cam = {
            let mut c = camera.chars();
            c.next().map_or(String::new(), |f| f.to_uppercase().collect::<String>() + c.as_str())
        };
        let cw = hud.label_w("Onboard", fs, Weight::Semibold) + u(0.046);
        rx -= cw;
        btn(hud, rx, cw, Action::Camera, false, false, "Next camera  (C)", &mut tips);
        // a little camera: body + lens bump
        let (kx, ky) = (rx + u(0.013), ry + s * 0.5);
        hud.round_rect(kx, ky - 0.006, u(0.015), 0.013, 2.0, ic);
        hud.rect(kx + u(0.004), ky - 0.0095, u(0.006), 0.004, ic);
        hud.round_rect(kx + u(0.0055), ky - 0.0028, u(0.004), 0.0056, 2.0, t::panel(a));
        hud.label(&cam, rx + u(0.034), ry + (s - hud.line_h(fs)) * 0.5, fs, Weight::Semibold, t::fg(a));
        rx -= u(0.012);
        // speed: - 1x +
        let mut x = rx - (2.0 * sw + slw);
        btn(hud, x, sw, Action::Slower, false, false, "Slower  (Down)", &mut tips);
        hud.rect(x + sw * 0.32, ry + s * 0.5 - hud.px_y(1.0), sw * 0.36, hud.px_y(2.0), ic);
        x += sw;
        let sl = speed_label(speed);
        let lw = hud.label_w(&sl, fs, Weight::Semibold);
        hud.label(&sl, x + (slw - lw) * 0.5, ry + (s - hud.line_h(fs)) * 0.5, fs, Weight::Semibold,
                  if (speed - 1.0).abs() < 1e-3 { t::dim(a) } else { t::gold(a) });
        x += slw;
        btn(hud, x, sw, Action::Faster, false, false, "Faster  (Up)", &mut tips);
        hud.rect(x + sw * 0.32, ry + s * 0.5 - hud.px_y(1.0), sw * 0.36, hud.px_y(2.0), ic);
        hud.rect(x + sw * 0.5 - hud.px_x(1.0), ry + s * 0.5 - sw * 0.18 * aspect, hud.px_x(2.0), sw * 0.36 * aspect, ic);


        // left: < the watched car >
        if let Some(&(ci, _)) = drivers.get(watched) {
            let name = self.name(ci);
            let sub = format!("Car {} of {}", watched + 1, drivers.len());
            let avail = group_x - ix - u(0.02);
            let nw = hud.label_w(&name, fs, Weight::Semibold).max(hud.label_w(&sub, 0.0115, Weight::Regular));
            let aw = u(s * 0.8);
            let total = (nw + u(0.03) + aw * 2.0).min(avail);
            let mx = ix;
            if drivers.len() > 1 {
                btn(hud, mx, aw, Action::PrevCar, false, false, "Previous car  ([)", &mut tips);
                let (cx1, cy1) = (mx + aw * 0.5, ry + s * 0.5);
                hud.tri([(cx1 + u(0.004), cy1 - 0.008), (cx1 - u(0.004), cy1), (cx1 + u(0.004), cy1 + 0.008)], ic);
                btn(hud, mx + total - aw, aw, Action::NextCar, false, false, "Next car  (])", &mut tips);
                let cx2 = mx + total - aw * 0.5;
                hud.tri([(cx2 - u(0.004), cy1 - 0.008), (cx2 + u(0.004), cy1), (cx2 - u(0.004), cy1 + 0.008)], ic);
            }
            let midx = mx + total * 0.5;
            let nl = hud.label_w(&name, fs, Weight::Semibold);
            let y0 = ry + (s - hud.line_h(fs) - hud.line_h(0.0115) + 0.004) * 0.5;
            hud.label(&name, midx - nl * 0.5, y0, fs, Weight::Semibold, t::fg(a));
            let sl2 = hud.label_w(&sub, 0.0115, Weight::Regular);
            hud.label(&sub, midx - sl2 * 0.5, y0 + hud.line_h(fs) - 0.004, 0.0115, Weight::Regular, t::dim(a));
        }

        // tooltips last, over everything
        for (cx, top, s) in tips {
            let ts = 0.012;
            let w = hud.label_w(s, ts, Weight::Regular) + u(0.014);
            let h = hud.line_h(ts) + 0.006;
            let x = (cx - w * 0.5).clamp(0.002, 0.998 - w);
            let y = top - h - 0.012 - (pad + row1 + 0.006);
            hud.panel(x, y, w, h, 5.0, t::panel2(0.97 * a), t::line(a));
            hud.label(s, x + u(0.007), y + 0.003, ts, Weight::Regular, t::fg(a));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_list(&mut self, hud: &mut Hud, rep: &Replay, drivers: &[(usize, String)], watched: usize,
                 play_t: f32, cur: Option<(f32, f32)>, aspect: f32, a: f32, live: bool, bottom: f32) {
        let u = |v: f32| v / aspect;
        let fs = 0.0135;
        let pad = 0.012;
        let lh = hud.line_h(fs) + 0.009;
        let (x, y) = (u(0.022), 0.025);
        let w = u(0.34);
        let head = hud.line_h(fs) + 0.012;
        // at most 15 rows; the wheel scrolls the rest
        let fit = ((((bottom - y) - pad * 2.0 - head) / lh).floor().max(1.0) as usize).min(15);
        let n = drivers.len();
        self.rows = fit.min(n);
        // keep the watched car in view when it changes
        if watched != self.last_watched {
            self.last_watched = watched;
            if watched < self.scroll {
                self.scroll = watched;
            } else if watched >= self.scroll + self.rows {
                self.scroll = watched + 1 - self.rows;
            }
        }
        self.scroll = self.scroll.min(n.saturating_sub(self.rows));
        let h = pad * 2.0 + head + lh * self.rows as f32;
        self.list_rect = (x, y, w, h);
        hud.panel(x, y, w, h, 10.0, t::panel(0.74 * a), t::line(0.55 * a));
        let (ix, iw) = (x + u(0.013), w - u(0.026));
        hud.label("Cars", ix, y + pad, fs, Weight::Semibold, t::fg(a));
        let cnt = format!("{n}");
        let cw = hud.label_w(&cnt, fs, Weight::Regular);
        hud.label(&cnt, ix + u(0.045), y + pad, fs, Weight::Regular, t::dim2(a));
        let _ = cw;
        let unit = if crate::cluster::mph() { "mph" } else { "km/h" };
        let uw = hud.label_w(unit, 0.0115, Weight::Regular);
        hud.label(unit, ix + iw - uw, y + pad + 0.002, 0.0115, Weight::Regular, t::dim2(a));
        let mut yy = y + pad + head;
        for (i, &(ci, _)) in drivers.iter().enumerate().skip(self.scroll).take(self.rows) {
            let hot = live && cur.map_or(false, |(cx, cy)| cx >= x && cx <= x + w && cy >= yy && cy < yy + lh);
            let me = i == watched;
            if me || hot {
                hud.round_rect(x + u(0.004), yy, w - u(0.008), lh, 6.0, if me { t::panel2(a) } else { t::fg(0.09 * a) });
            }
            if me {
                hud.round_rect(x + u(0.004), yy + lh * 0.2, hud.px_x(3.0), lh * 0.6, 1.5, t::accent(a));
            }
            let ty = yy + (lh - hud.line_h(fs)) * 0.5;
            let num = format!("{}", i + 1);
            let nw = hud.label_w(&num, fs, Weight::Regular);
            hud.label(&num, ix + u(0.022) - nw, ty, fs, Weight::Regular, t::dim2(a));
            let name = self.name(ci);
            hud.label(&name, ix + u(0.034), ty, fs, if me { Weight::Semibold } else { Weight::Regular },
                      if me { t::fg(a) } else { t::fg(0.86 * a) });
            let kmh = rep.speed(ci, play_t) * 3.6;
            let v = if crate::cluster::mph() { kmh / 1.609_344 } else { kmh };
            let vs = format!("{v:.0}");
            let vw = hud.label_w(&vs, fs, Weight::Regular);
            hud.label(&vs, ix + iw - vw, ty, fs, Weight::Regular, t::dim(a));
            if live {
                hud.hotspot(x, yy, w, lh, Action::Watch(i));
            }
            yy += lh;
        }
        // more above / below
        if self.scroll > 0 {
            hud.round_rect(x + w * 0.5 - u(0.02), y + pad + head - 0.004, u(0.04), hud.px_y(2.0), 1.0, t::dim(0.6 * a));
        }
        if self.scroll + self.rows < n {
            hud.round_rect(x + w * 0.5 - u(0.02), y + h - pad + 0.003, u(0.04), hud.px_y(2.0), 1.0, t::dim(0.6 * a));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn names_and_clock() {
        assert_eq!(pretty("ks_audi_r8_lms_gt3_evo_2"), "Audi R8 LMS GT3 Evo 2");
        assert_eq!(pretty("ks_ferrari_f2004"), "Ferrari F2004");
        assert_eq!(pretty("ks_bmw_m4_gt3"), "BMW M4 GT3");
        assert_eq!(clock(83.44), "1:23.4");
        assert_eq!(clock(5.0), "0:05.0");
        assert_eq!(speed_label(0.25), "0.25×");
        assert_eq!(speed_label(4.0), "4×");
    }
}
