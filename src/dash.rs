//! The car's digital dash screens, laid out from the game's own pages.
//!
//! In the game a display is an HTML page (`<car>/displays/display*.html` +
//! its CSS, fonts and images) rendered by Coherent GameFace into a texture
//! that the display material (`CarActorData.displays[i].material`) shows. Its
//! values come from `data-bind-value="{{ModelCurrentCar...}}"` expressions,
//! and its JS toggles body classes (`lights-level1`, `direction-light-left`,
//! ...) from the car state (common_assets/displays/js/basecardisplay.js).
//!
//! This is a small, honest subset of that: an HTML tree, the compiled CSS
//! (display.css + common.css), absolutely-positioned boxes, text in the
//! car's own fonts, background colours and images (sprite atlases with
//! background-position), the body classes basecardisplay.js sets, and the
//! bindings evaluated against the viewer's car state. No JS runs; anything a
//! page draws only from script (gauges built in JS, canvas) is not shown.
//! Converted AC1 pages (acecm-convert) draw text as sprites from a bitmap
//! font strip (`data-ac1-font`, AC1_FONTS metrics in the page); that path is
//! handled too.

use crate::asset::TextureEntry;
use crate::fxstate::FxState;
use crate::load::Vfs;
use std::collections::HashMap;

// ---------------------------------------------------------------- HTML tree --

#[derive(Clone, Debug, Default)]
pub struct Node {
    pub tag: String,
    pub id: String,
    pub classes: Vec<String>,
    pub attrs: Vec<(String, String)>,
    pub text: String,
    pub children: Vec<Node>,
}

impl Node {
    fn attr(&self, k: &str) -> Option<&str> {
        self.attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
    }
}

const VOID: &[&str] = &["meta", "link", "br", "img", "input", "hr", "source", "base"];

fn parse_attrs(s: &str) -> Vec<(String, String)> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        while i < b.len() && (b[i].is_ascii_whitespace() || b[i] == b'/') {
            i += 1;
        }
        let ks = i;
        while i < b.len() && !b[i].is_ascii_whitespace() && b[i] != b'=' && b[i] != b'/' {
            i += 1;
        }
        let k = s[ks..i].to_lowercase();
        if k.is_empty() {
            i += 1;
            continue;
        }
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        let mut v = String::new();
        if i < b.len() && b[i] == b'=' {
            i += 1;
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < b.len() && (b[i] == b'"' || b[i] == b'\'') {
                let q = b[i];
                i += 1;
                let vs = i;
                while i < b.len() && b[i] != q {
                    i += 1;
                }
                v = s[vs..i.min(b.len())].to_string();
                i += 1;
            } else {
                let vs = i;
                while i < b.len() && !b[i].is_ascii_whitespace() {
                    i += 1;
                }
                v = s[vs..i].to_string();
            }
        }
        out.push((k, v));
    }
    out
}

/// A tolerant parser: enough for the display pages (no entities beyond the
/// common ones, no implied end tags beyond void elements).
pub fn parse_html(src: &str) -> Node {
    let mut stack: Vec<Node> = vec![Node { tag: "#root".into(), ..Default::default() }];
    let mut i = 0;
    let s = src;
    while i < s.len() {
        let rest = &s[i..];
        if rest.starts_with("<!--") {
            i += rest.find("-->").map_or(rest.len(), |e| e + 3);
            continue;
        }
        if rest.starts_with('<') {
            let Some(end) = rest.find('>') else { break };
            let inner = &rest[1..end];
            i += end + 1;
            if inner.starts_with('!') || inner.starts_with('?') {
                continue;
            }
            if let Some(name) = inner.strip_prefix('/') {
                let name = name.trim().to_lowercase();
                // pop to the matching open tag
                if let Some(pos) = stack.iter().rposition(|n| n.tag == name) {
                    while stack.len() > pos.max(1) {
                        let n = stack.pop().unwrap();
                        stack.last_mut().unwrap().children.push(n);
                    }
                }
                continue;
            }
            let self_close = inner.ends_with('/');
            let inner = inner.trim_end_matches('/');
            let (name, attrs) = match inner.find(|c: char| c.is_whitespace()) {
                Some(p) => (&inner[..p], &inner[p..]),
                None => (inner, ""),
            };
            let tag = name.to_lowercase();
            let attrs = parse_attrs(attrs);
            let mut n = Node { tag: tag.clone(), ..Default::default() };
            for (k, v) in &attrs {
                match k.as_str() {
                    "id" => n.id = v.clone(),
                    "class" => n.classes = v.split_whitespace().map(str::to_string).collect(),
                    _ => {}
                }
            }
            n.attrs = attrs;
            if tag == "script" || tag == "style" {
                // keep the raw content (AC1_FONTS lives in a script)
                let close = format!("</{tag}");
                let e = s[i..].to_lowercase().find(&close).map_or(s.len() - i, |e| e);
                n.text = s[i..i + e].to_string();
                i += e;
                if let Some(g) = s[i..].find('>') {
                    i += g + 1;
                }
                stack.last_mut().unwrap().children.push(n);
                continue;
            }
            if self_close || VOID.contains(&tag.as_str()) {
                stack.last_mut().unwrap().children.push(n);
            } else {
                stack.push(n);
            }
            continue;
        }
        let e = rest.find('<').unwrap_or(rest.len());
        let t = rest[..e].replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">");
        if !t.trim().is_empty() {
            let top = stack.last_mut().unwrap();
            top.text.push_str(t.trim());
        }
        i += e;
    }
    while stack.len() > 1 {
        let n = stack.pop().unwrap();
        stack.last_mut().unwrap().children.push(n);
    }
    stack.pop().unwrap()
}

// ---------------------------------------------------------------------- CSS --

#[derive(Clone, Debug)]
enum Part {
    Tag(String),
    Id(String),
    Class(String),
    Attr(String, Option<String>),
    Any,
    /// a pseudo-class or combinator we do not model: never matches
    Never,
}

#[derive(Clone, Debug)]
struct Compound {
    parts: Vec<Part>,
    /// combinator to the NEXT compound on the left: true = '>' (child)
    child: bool,
}

#[derive(Clone, Debug)]
struct Selector {
    /// right-most first
    chain: Vec<Compound>,
    spec: (u32, u32, u32),
}

#[derive(Clone, Debug)]
struct Rule {
    sels: Vec<Selector>,
    decls: Vec<(String, String, bool)>,
    order: usize,
}

fn parse_compound(s: &str) -> Vec<Part> {
    let mut parts = Vec::new();
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    let ident = |i: &mut usize| -> String {
        let st = *i;
        while *i < b.len() && (b[*i].is_alphanumeric() || b[*i] == '-' || b[*i] == '_') {
            *i += 1;
        }
        b[st..*i].iter().collect()
    };
    while i < b.len() {
        match b[i] {
            '#' => { i += 1; parts.push(Part::Id(ident(&mut i))); }
            '.' => { i += 1; parts.push(Part::Class(ident(&mut i))); }
            '*' => { i += 1; parts.push(Part::Any); }
            '[' => {
                let e = b[i..].iter().position(|&c| c == ']').map_or(b.len(), |p| i + p);
                let inner: String = b[i + 1..e].iter().collect();
                i = e + 1;
                match inner.split_once('=') {
                    Some((k, v)) => parts.push(Part::Attr(
                        k.trim().trim_end_matches(['~', '|', '^', '$', '*']).to_lowercase(),
                        Some(v.trim().trim_matches(['"', '\'']).to_string()))),
                    None => parts.push(Part::Attr(inner.trim().to_lowercase(), None)),
                }
            }
            ':' => {
                i += 1;
                if i < b.len() && b[i] == ':' {
                    i += 1;
                }
                let name = ident(&mut i);
                if i < b.len() && b[i] == '(' {
                    let e = b[i..].iter().position(|&c| c == ')').map_or(b.len(), |p| i + p);
                    i = e + 1;
                }
                // structural pseudo-classes: assume they match (layout is
                // per element anyway); states and pseudo-elements never do
                if !matches!(name.as_str(), "first-child" | "last-child" | "nth-child" | "not" | "root") {
                    parts.push(Part::Never);
                }
            }
            c if c.is_alphanumeric() => parts.push(Part::Tag(ident(&mut i).to_lowercase())),
            _ => i += 1,
        }
    }
    parts
}

fn parse_selector(s: &str) -> Option<Selector> {
    let s = s.replace('>', " > ");
    if s.contains('+') || s.contains('~') {
        return None;
    }
    let toks: Vec<&str> = s.split_whitespace().collect();
    let mut chain = Vec::new();
    let mut child_next = false;
    for t in toks.iter().rev() {
        if *t == ">" {
            child_next = true;
            if let Some(c) = chain.last_mut() {
                let c: &mut Compound = c;
                c.child = true;
            }
            continue;
        }
        let _ = child_next;
        chain.push(Compound { parts: parse_compound(t), child: false });
        child_next = false;
    }
    if chain.is_empty() {
        return None;
    }
    let mut spec = (0, 0, 0);
    for c in &chain {
        for p in &c.parts {
            match p {
                Part::Id(_) => spec.0 += 1,
                Part::Class(_) | Part::Attr(..) => spec.1 += 1,
                Part::Tag(_) => spec.2 += 1,
                _ => {}
            }
        }
    }
    Some(Selector { chain, spec })
}

fn strip_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while let Some(p) = s[i..].find("/*") {
        out.push_str(&s[i..i + p]);
        i += p + 2;
        i += s[i..].find("*/").map_or(s.len() - i, |e| e + 2);
    }
    out.push_str(&s[i..]);
    out
}

/// Rules + @font-face (family -> url).
fn parse_css(src: &str, order0: usize) -> (Vec<Rule>, Vec<(String, String)>) {
    let s = strip_comments(src);
    let mut rules = Vec::new();
    let mut faces = Vec::new();
    let b = s.as_bytes();
    let mut i = 0;
    let mut order = order0;
    while i < b.len() {
        let Some(ob) = s[i..].find('{') else { break };
        let head = s[i..i + ob].trim().to_string();
        // matching close brace (nested for @media / @keyframes)
        let mut depth = 0;
        let mut j = i + ob;
        while j < b.len() {
            match b[j] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            j += 1;
        }
        let body = &s[i + ob + 1..j.min(s.len())];
        i = j + 1;
        if head.starts_with("@font-face") {
            let mut fam = String::new();
            let mut url = String::new();
            for d in body.split(';') {
                if let Some((k, v)) = d.split_once(':') {
                    match k.trim() {
                        "font-family" => fam = v.trim().trim_matches(['"', '\'']).to_string(),
                        "src" => url = css_url(v).unwrap_or_default(),
                        _ => {}
                    }
                }
            }
            if !fam.is_empty() && !url.is_empty() {
                faces.push((fam, url));
            }
            continue;
        }
        if head.starts_with('@') {
            continue;
        }
        let sels: Vec<Selector> = head.split(',').filter_map(|x| parse_selector(x.trim())).collect();
        if sels.is_empty() {
            continue;
        }
        let mut decls = Vec::new();
        for d in body.split(';') {
            if let Some((k, v)) = d.split_once(':') {
                let v = v.trim();
                let imp = v.contains("!important");
                decls.push((k.trim().to_lowercase(), v.replace("!important", "").trim().to_string(), imp));
            }
        }
        rules.push(Rule { sels, decls, order });
        order += 1;
    }
    (rules, faces)
}

fn css_url(v: &str) -> Option<String> {
    let p = v.find("url(")?;
    let rest = &v[p + 4..];
    let e = rest.find(')')?;
    Some(rest[..e].trim().trim_matches(['"', '\'']).to_string())
}

fn matches_compound(n: &Node, parts: &[Part], body: &BodyState) -> bool {
    parts.iter().all(|p| match p {
        Part::Any => true,
        Part::Never => false,
        Part::Tag(t) => n.tag == *t,
        Part::Id(i) => n.id == *i,
        Part::Class(c) => n.classes.iter().any(|x| x == c) || (n.tag == "body" && body.classes.iter().any(|x| x == c)),
        Part::Attr(k, v) => {
            let got = if n.tag == "body" {
                body.attrs.get(k).map(|s| s.as_str()).or_else(|| n.attr(k))
            } else {
                n.attr(k)
            };
            match (got, v) {
                (Some(_), None) => true,
                (Some(g), Some(w)) => g == w,
                _ => false,
            }
        }
    })
}

fn matches(sel: &Selector, path: &[&Node], body: &BodyState) -> bool {
    let Some((me, anc)) = path.split_last() else { return false };
    if !matches_compound(me, &sel.chain[0].parts, body) {
        return false;
    }
    let mut k = anc.len();
    for ci in 1..sel.chain.len() {
        let child = sel.chain[ci - 1].child;
        let c = &sel.chain[ci];
        if child {
            if k == 0 || !matches_compound(anc[k - 1], &c.parts, body) {
                return false;
            }
            k -= 1;
        } else {
            let mut found = false;
            while k > 0 {
                k -= 1;
                if matches_compound(anc[k], &c.parts, body) {
                    found = true;
                    break;
                }
            }
            if !found {
                return false;
            }
        }
    }
    true
}

// --------------------------------------------------------------- the state --

/// The body classes / data attributes basecardisplay.js would set.
#[derive(Clone, Debug, Default)]
pub struct BodyState {
    pub classes: Vec<String>,
    pub attrs: HashMap<String, String>,
}

pub fn body_state(st: &FxState, page: &str, blink_on: bool, shift_rpm: f32, limiter_rpm: f32) -> BodyState {
    let mut c: Vec<&str> = vec!["cardisplay", "loaded", "already-on", "skip-intro", "has-abs", "has-tc"];
    let ignition = st.rpm > 0.0 || st.lights > 0;
    if ignition {
        c.extend(["ignition-on", "daylight-lights"]);
    }
    if st.rpm > 0.0 {
        c.push("engine-on");
    }
    if st.lights > 0 {
        c.push("lights-on");
        c.push(match st.lights { 1 => "lights-level1", 2 => "lights-level2", _ => "lights-level3" });
    }
    use crate::fxstate::Indicator::*;
    let (l, r) = match st.indicator { Left => (true, false), Right => (false, true), Hazard => (true, true), Off => (false, false) };
    if l && blink_on { c.push("direction-light-left"); }
    if r && blink_on { c.push("direction-light-right"); }
    if st.indicator == Hazard { c.push("warning-lights"); }
    if st.fog { c.push("rain-lights"); }
    if st.gear == 0 { c.push("neutral"); }
    if shift_rpm > 0.0 && st.rpm >= shift_rpm { c.push("shift-up"); }
    if limiter_rpm > 0.0 && st.rpm >= limiter_rpm { c.push("rpmlimiter"); }
    if st.pit_limiter { c.push("pitlimiter"); }
    // what the pages' JS toggles for ~10 s after a lap (ks_bmw_m4_gt3,
    // ferrari_488_challenge_evo ...: show-laptime, best-laptime)
    if let Some(l) = st.lap.filter(|l| l.laps > 0 && l.since_lap < 10.0) {
        c.push("show-laptime");
        if l.last_is_best { c.push("best-laptime"); }
    }
    let mut attrs = HashMap::new();
    attrs.insert("data-page".into(), page.to_string());
    attrs.insert("data-mode".into(), page.to_string());
    BodyState { classes: c.into_iter().map(str::to_string).collect(), attrs }
}

/// Evaluate a data-bind-value expression against the car state. Recognises
/// the ModelCurrentCar fields the shipped pages bind and the formatting
/// helpers wrapped around them; anything else returns None (the element
/// keeps its authored text).
pub fn eval_binding(expr: &str, st: &FxState) -> Option<String> {
    if let Some(v) = eval_call(expr.trim(), st) {
        return Some(v);
    }
    if expr.contains(".gear}}") || expr.trim_end().ends_with(".gear") {
        return Some(match st.gear { g if g < 0 => "R".into(), 0 => "N".into(), g => g.to_string() });
    }
    if let Some(v) = eval_number(expr, st) {
        if let Some(p) = expr.find("toFixed(") {
            let n: usize = expr[p + 8..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0);
            return Some(format!("{:.*}", n, v));
        }
        return Some(format!("{}", v.round() as i64));
    }
    let e = expr.replace(' ', "");
    let kmh = st.speed_mps * 3.6;
    let gear = match st.gear { g if g < 0 => "R".to_string(), 0 => "N".to_string(), g => g.to_string() };
    let num: Option<f64> = if e.contains("display_speed_kmh") || e.contains("speed_kmh") || e.contains("msToKmh") {
        Some(kmh as f64)
    } else if e.contains("display_speed_mph") || e.contains("speed_mph") {
        Some(kmh as f64 / 1.609344)
    } else if e.contains(".gear") || e.contains("gear_int") {
        return Some(gear);
    } else if e.contains(".rpm") {
        Some(st.rpm as f64)
    } else if e.contains("water_temperature") {
        Some(if st.rpm > 0.0 { 88.0 } else { 20.0 })
    } else if e.contains("oil_temperature") {
        Some(if st.rpm > 0.0 { 102.0 } else { 20.0 })
    } else if e.contains("oil_pressure") {
        Some(if st.rpm > 0.0 { 4.2 } else { 0.0 })
    } else if e.contains("fuel_liter_current_quantity") {
        Some(42.0)
    } else if e.contains("fuel_liter_per_lap") {
        Some(3.1)
    } else if e.contains("fuel_liter_used") {
        Some(0.0)
    } else if e.contains("total_lap_count") {
        Some(st.lap.map_or(0.0, |l| l.laps as f64))
    } else if e.contains("brake_bias") {
        Some(st.brake_bias.unwrap_or(0.565) as f64)
    } else if e.contains("tc_level") {
        Some(st.tc_level.unwrap_or(3.0) as f64)
    } else if e.contains("abs_level") {
        Some(st.abs_level.unwrap_or(3.0) as f64)
    } else if e.contains("esc_level") {
        Some(3.0)
    } else if e.contains("engine_map_level") {
        Some(0.0)
    } else if e.contains("tyre_pression") {
        // which wheel: the binding names it (fl / fr / rl / rr, or 0..3)
        let i = ["fl", "fr", "rl", "rr"].iter().position(|w| e.to_lowercase().contains(w))
            .or_else(|| (0..4).find(|i| e.contains(&format!("[{i}]"))));
        Some(st.tyre_bar.map_or(1.85, |b| b[i.unwrap_or(0)]) as f64)
    } else if e.contains("brake_pressure") {
        Some(st.brake as f64 * 80.0)
    } else if e.contains("air_temperature") {
        Some(21.0)
    } else if e.contains("total_km") {
        Some(1234.0)
    } else if e.contains("laptime_ms") || e.contains("formatTime") {
        return Some("--:--.--".into());
    } else if e.contains("throttle") || e.contains("gas") {
        Some(st.throttle as f64 * 100.0)
    } else {
        None
    };
    let mut v = num?;
    // arithmetic tails the pages use
    if let Some(p) = e.find("*100") {
        if p > 0 { v *= 100.0; }
    }
    if e.contains("+1") && !e.contains("+1)") || e.ends_with("+1") {
        v += 1.0;
    }
    // formatting
    if let Some(p) = e.find("toFixed(") {
        let n: usize = e[p + 8..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0);
        return Some(format!("{:.*}", n, v));
    }
    if let Some(p) = e.find("pad(") {
        let tail = &e[p + 4..];
        let w: usize = tail.rsplit(',').next().and_then(|t| t.trim_end_matches(')').parse().ok()).unwrap_or(0);
        return Some(format!("{:0w$}", v.abs().floor() as i64, w = w));
    }
    Some(format!("{}", v.round() as i64))
}

// ------------------------------------------------------------------ layout --

#[derive(Clone, Debug)]
pub enum Draw {
    Rect { x: f32, y: f32, w: f32, h: f32, color: [f32; 4] },
    /// a car texture or a page image, uv in [0,1]
    Image { x: f32, y: f32, w: f32, h: f32, img: usize, uv: [f32; 4], alpha: f32 },
    /// `x`, `w`: the box the text is aligned in (0 left, 1 centre, 2 right);
    /// `y`: baseline. The renderer measures with the real font.
    Text { x: f32, y: f32, w: f32, align: u8, size: f32, font: usize, text: String, color: [f32; 4], spacing: f32 },
}

/// A page image: a car texture index (BC, uploaded through TexCache) or
/// decoded RGBA8 pixels from a PNG.
#[derive(Clone, Debug)]
pub enum Image {
    Tex(usize, u32, u32),
    Rgba(u32, u32, std::sync::Arc<Vec<u8>>),
}

impl Image {
    pub fn size(&self) -> (f32, f32) {
        match self {
            Image::Tex(_, w, h) | Image::Rgba(w, h, _) => (*w as f32, *h as f32),
        }
    }
}

/// An AC1 sprite font: a 4096x64 strip, u ranges per ASCII 32..126.
#[derive(Clone, Debug)]
pub struct SpriteFont {
    pub img: usize,
    pub u: Vec<Option<(f32, f32)>>,
    pub w: f32,
    pub h: f32,
}

#[derive(Clone, Debug)]
pub struct Page {
    pub html_path: String,
    pub material: String,
    pub width: u32,
    pub height: u32,
    pub root: Node,
    rules: Vec<Rule>,
    /// family (lowercase) -> index into `fonts`
    pub font_names: HashMap<String, usize>,
    pub fonts: Vec<std::sync::Arc<Vec<u8>>>,
    /// url (lowercase, as written) -> index into `images`
    image_ix: HashMap<String, usize>,
    pub images: Vec<Image>,
    sprite_fonts: HashMap<String, SpriteFont>,
    /// first `.page` id, what data-page selects
    pub default_page: String,
    /// bound elements, for --check
    pub bindings: Vec<String>,
    pub missing: Vec<String>,
    /// style_of results by (path, body) fingerprint: the cascade over every
    /// rule for every node was most of a redraw, and redrawing every frame
    /// (no 10 Hz lag) made it ~0.7 ms of CPU a frame
    style_cache: std::sync::Arc<std::sync::Mutex<HashMap<u64, Style>>>,
}

#[derive(Clone, Debug, Default)]
struct Style {
    m: HashMap<String, (String, (u32, u32, u32), usize, bool)>,
}

impl Style {
    fn get(&self, k: &str) -> Option<&str> {
        self.m.get(k).map(|v| v.0.as_str())
    }
    fn set(&mut self, k: &str, v: &str, spec: (u32, u32, u32), order: usize, imp: bool) {
        let replace = match self.m.get(k) {
            None => true,
            Some((_, s0, o0, i0)) => (imp, spec, order) >= (*i0, *s0, *o0),
        };
        if replace {
            self.m.insert(k.to_string(), (v.to_string(), spec, order, imp));
        }
    }
}

fn px(v: &str, rel: f32, em: f32) -> Option<f32> {
    let v = v.trim();
    if let Some(n) = v.strip_suffix("px") {
        return n.trim().parse().ok();
    }
    if let Some(n) = v.strip_suffix('%') {
        return n.trim().parse::<f32>().ok().map(|p| p * rel / 100.0);
    }
    if let Some(n) = v.strip_suffix("rem") {
        return n.trim().parse::<f32>().ok().map(|p| p * 16.0);
    }
    if let Some(n) = v.strip_suffix("em") {
        return n.trim().parse::<f32>().ok().map(|p| p * em);
    }
    if v == "0" {
        return Some(0.0);
    }
    v.parse().ok()
}

pub fn parse_color(v: &str) -> Option<[f32; 4]> {
    let v = v.trim().to_lowercase();
    let srgb = |c: f32| c; // pages author in display sRGB; the target is sRGB too
    if let Some(h) = v.strip_prefix('#') {
        let h: String = h.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
        let n = |s: &str| u8::from_str_radix(s, 16).ok().map(|x| x as f32 / 255.0);
        return match h.len() {
            3 | 4 => {
                let ch: Vec<f32> = h.chars().map(|c| n(&format!("{c}{c}")).unwrap_or(0.0)).collect();
                Some([srgb(ch[0]), srgb(ch[1]), srgb(ch[2]), *ch.get(3).unwrap_or(&1.0)])
            }
            6 | 8 => Some([n(&h[0..2])?, n(&h[2..4])?, n(&h[4..6])?,
                          if h.len() == 8 { n(&h[6..8])? } else { 1.0 }]),
            _ => None,
        };
    }
    if let Some(p) = v.find('(') {
        let inner = v[p + 1..].trim_end_matches(')');
        let parts: Vec<f32> = inner.split(',').filter_map(|x| x.trim().trim_end_matches('%').parse().ok()).collect();
        if parts.len() >= 3 {
            return Some([parts[0] / 255.0, parts[1] / 255.0, parts[2] / 255.0, *parts.get(3).unwrap_or(&1.0)]);
        }
        return None;
    }
    Some(match v.as_str() {
        "white" => [1.0, 1.0, 1.0, 1.0],
        "black" => [0.0, 0.0, 0.0, 1.0],
        "red" => [1.0, 0.0, 0.0, 1.0],
        "green" => [0.0, 0.5, 0.0, 1.0],
        "lime" => [0.0, 1.0, 0.0, 1.0],
        "blue" => [0.0, 0.0, 1.0, 1.0],
        "yellow" => [1.0, 1.0, 0.0, 1.0],
        "orange" => [1.0, 0.65, 0.0, 1.0],
        "grey" | "gray" => [0.5, 0.5, 0.5, 1.0],
        "transparent" => [0.0, 0.0, 0.0, 0.0],
        _ => return None,
    })
}

struct Ctx<'a> {
    page: &'a Page,
    st: &'a FxState,
    body: &'a BodyState,
    out: Vec<(i32, Draw)>,
    /// overflow:hidden clip rect (x0, y0, x1, y1) for rects and images
    clip: (f32, f32, f32, f32),
}

/// Clip a rect / image draw to `c`, trimming its uv to match.
fn clip_draw(d: Draw, c: (f32, f32, f32, f32)) -> Draw {
    match d {
        Draw::Image { x, y, w, h, img, uv, alpha } => {
            let (x0, y0) = (x.max(c.0), y.max(c.1));
            let (x1, y1) = ((x + w).min(c.2), (y + h).min(c.3));
            if x1 <= x0 || y1 <= y0 || w <= 0.0 || h <= 0.0 {
                return Draw::Image { x, y, w: 0.0, h: 0.0, img, uv, alpha: 0.0 };
            }
            let fu = |px: f32| uv[0] + (uv[2] - uv[0]) * (px - x) / w;
            let fv = |py: f32| uv[1] + (uv[3] - uv[1]) * (py - y) / h;
            Draw::Image { x: x0, y: y0, w: x1 - x0, h: y1 - y0, img, uv: [fu(x0), fv(y0), fu(x1), fv(y1)], alpha }
        }
        Draw::Rect { x, y, w, h, color } => {
            let (x0, y0) = (x.max(c.0), y.max(c.1));
            let (x1, y1) = ((x + w).min(c.2), (y + h).min(c.3));
            Draw::Rect { x: x0, y: y0, w: (x1 - x0).max(0.0), h: (y1 - y0).max(0.0), color }
        }
        t => t,
    }
}

impl Page {
    fn style_of(&self, path: &[&Node], body: &BodyState) -> Style {
        use std::hash::{Hash, Hasher};
        // everything selector matching and the inline style can read
        let mut h = std::collections::hash_map::DefaultHasher::new();
        body.classes.hash(&mut h);
        let mut ba: Vec<_> = body.attrs.iter().collect();
        ba.sort();
        ba.hash(&mut h);
        for n in path {
            (&n.tag, &n.id, &n.classes, &n.attrs).hash(&mut h);
        }
        let key = h.finish();
        if let Some(s) = self.style_cache.lock().ok().and_then(|c| c.get(&key).cloned()) {
            return s;
        }
        let s = self.style_of_uncached(path, body);
        if let Ok(mut c) = self.style_cache.lock() {
            if c.len() > 8192 {
                c.clear();
            }
            c.insert(key, s.clone());
        }
        s
    }

    fn style_of_uncached(&self, path: &[&Node], body: &BodyState) -> Style {
        let mut s = Style::default();
        for r in &self.rules {
            let mut best: Option<(u32, u32, u32)> = None;
            for sel in &r.sels {
                if matches(sel, path, body) {
                    best = Some(best.map_or(sel.spec, |b| b.max(sel.spec)));
                }
            }
            if let Some(spec) = best {
                for (k, v, imp) in &r.decls {
                    s.set(k, v, spec, r.order, *imp);
                }
            }
        }
        // inline style wins over sheets
        if let Some(inl) = path.last().and_then(|n| n.attr("style")) {
            for d in inl.split(';') {
                if let Some((k, v)) = d.split_once(':') {
                    s.set(&k.trim().to_lowercase(), v.trim(), (9, 0, 0), usize::MAX, false);
                }
            }
        }
        s
    }

    /// The page's own background (body background-color), sRGB.
    pub fn background(&self, body: &BodyState) -> [f32; 4] {
        let root = &self.root;
        let html = root.children.iter().find(|n| n.tag == "html").unwrap_or(root);
        let Some(b) = find_tag(html, "body") else { return [0.0, 0.0, 0.0, 1.0] };
        let st = self.style_of(&[html, b], body);
        st.get("background-color").and_then(parse_color).unwrap_or([0.0, 0.0, 0.0, 1.0])
    }

    /// Lay the page out for this car state.
    pub fn draw_list(&self, st: &FxState, body: &BodyState) -> Vec<Draw> {
        let mut cx = Ctx { page: self, st, body, out: Vec::new(), clip: (-1e9, -1e9, 1e9, 1e9) };
        let dynamic = apply_dynamics(&self.root, st);
        let root = &dynamic;
        let html = root.children.iter().find(|n| n.tag == "html").unwrap_or(root);
        let Some(bnode) = find_tag(html, "body") else { return Vec::new() };
        let path: Vec<&Node> = vec![html, bnode];
        let w = self.width as f32;
        let h = self.height as f32;
        let inh = Inherit { color: [1.0; 4], size: 16.0, font: 0, align: 0, spacing: 0.0, opacity: 1.0, visible: true, z: 0 };
        node(&mut cx, &path, (0.0, 0.0, w, h), &inh);
        let mut v = cx.out;
        v.sort_by_key(|(z, _)| *z);
        v.into_iter().map(|(_, d)| d).collect()
    }
}

fn find_tag<'a>(n: &'a Node, tag: &str) -> Option<&'a Node> {
    if n.tag == tag {
        return Some(n);
    }
    n.children.iter().find_map(|c| find_tag(c, tag))
}

#[derive(Clone)]
struct Inherit {
    color: [f32; 4],
    size: f32,
    font: usize,
    /// 0 left, 1 center, 2 right
    align: u8,
    spacing: f32,
    opacity: f32,
    visible: bool,
    /// the stacking level: a box without z-index paints at its parent's
    /// (M4 GT3's lap pop-up text over its z-index 101 panel)
    z: i32,
}

/// Rough advance of a string at `size` px, for boxes without a width.
fn text_width(text: &str, size: f32, spacing: f32) -> f32 {
    text.chars().count() as f32 * (size * 0.6 + spacing)
}

fn node(cx: &mut Ctx, path: &[&Node], parent: (f32, f32, f32, f32), inh: &Inherit) {
    let n = *path.last().unwrap();
    if matches!(n.tag.as_str(), "script" | "style" | "head" | "link" | "meta" | "title") {
        return;
    }
    let s = cx.page.style_of(path, cx.body);
    if s.get("display") == Some("none") {
        return;
    }
    let mut me = inh.clone();
    if let Some(c) = s.get("color").and_then(parse_color) {
        me.color = c;
    }
    if let Some(f) = s.get("font-size").and_then(|v| px(v, inh.size, inh.size)) {
        me.size = f;
    }
    if let Some(f) = s.get("font-family") {
        let fam = f.split(',').next().unwrap_or("").trim().trim_matches(['"', '\'']).to_lowercase();
        if let Some(&i) = cx.page.font_names.get(&fam) {
            me.font = i;
        }
    }
    match s.get("text-align") {
        Some("center") => me.align = 1,
        Some("right") => me.align = 2,
        Some("left") => me.align = 0,
        _ => {}
    }
    let flex = matches!(s.get("display"), Some("flex") | Some("inline-flex"));
    if flex {
        match s.get("justify-content") {
            Some("center") => me.align = 1,
            Some("flex-end") | Some("end") => me.align = 2,
            _ => {}
        }
    }
    if let Some(v) = s.get("letter-spacing").and_then(|v| px(v, 0.0, me.size)) {
        me.spacing = v;
    }
    if let Some(o) = s.get("opacity").and_then(|v| v.trim().parse::<f32>().ok()) {
        me.opacity *= o;
    }
    if s.get("visibility") == Some("hidden") {
        me.visible = false;
    } else if s.get("visibility") == Some("visible") {
        me.visible = true;
    }
    if me.opacity <= 0.01 {
        return;
    }

    // ---- the box ----
    let (px0, py0, pw, ph) = parent;
    let em = me.size;
    let abs = matches!(s.get("position"), Some("absolute") | Some("fixed")) || n.tag == "body";
    let g = |k: &str, rel: f32| s.get(k).and_then(|v| px(v, rel, em));
    let (left, right, top, bottom) = (g("left", pw), g("right", pw), g("top", ph), g("bottom", ph));
    let mut w = g("width", pw);
    let mut h = g("height", ph);
    let text = node_text(cx, n);
    let auto_w = text.as_ref().map_or(0.0, |t| text_width(t, em, me.spacing));
    let x = if abs {
        match (left, right, w) {
            (Some(l), Some(r), None) => { w = Some(pw - l - r); px0 + l }
            (Some(l), _, _) => px0 + l,
            (None, Some(r), Some(ww)) => px0 + pw - r - ww,
            (None, Some(r), None) => { w = Some(auto_w); px0 + pw - r - auto_w }
            _ => px0,
        }
    } else {
        px0
    };
    let y = if abs {
        match (top, bottom, h) {
            (Some(t), Some(b), None) => { h = Some(ph - t - b); py0 + t }
            (Some(t), _, _) => py0 + t,
            (None, Some(b), Some(hh)) => py0 + ph - b - hh,
            (None, Some(b), None) => py0 + ph - b - em * 1.2,
            _ => py0,
        }
    } else {
        py0
    };
    let w = w.unwrap_or(if abs && left.is_some() && right.is_none() { auto_w.max(0.0) } else { pw });
    let h = h.unwrap_or(if abs && top.is_some() && bottom.is_none() && text.is_some() { em * 1.2 } else { ph });
    // transform: translate(-50%, -100%) - the AC1 converter anchors with it
    let (mut x, mut y) = (x, y);
    if let Some(t) = s.get("transform") {
        if let Some(p) = t.find("translate(") {
            let args: Vec<&str> = t[p + 10..].split(')').next().unwrap_or("").split(',').collect();
            if let Some(a) = args.first().and_then(|a| px(a, w, em)) { x += a; }
            if let Some(b) = args.get(1).and_then(|b| px(b, h, em)) { y += b; }
        }
    }
    let z: i32 = s.get("z-index").and_then(|v| v.trim().parse().ok()).unwrap_or(inh.z);
    me.z = z;

    if me.visible {
        if let Some(c) = s.get("background-color").and_then(parse_color) {
            if c[3] > 0.0 && n.tag != "body" {
                cx.out.push((z, clip_draw(Draw::Rect { x, y, w, h, color: [c[0], c[1], c[2], c[3] * me.opacity] }, cx.clip)));
            }
        }
        if n.tag == "img" {
            if let Some(&img) = n.attr("src").and_then(|u| cx.page.image_ix.get(&u.to_lowercase())) {
                cx.out.push((z, clip_draw(Draw::Image { x, y, w, h, img, uv: [0.0, 0.0, 1.0, 1.0], alpha: me.opacity }, cx.clip)));
            }
        }
        if let Some(url) = s.get("background-image").and_then(css_url) {
            if let Some(&img) = cx.page.image_ix.get(&url.to_lowercase()) {
                let (iw, ih) = cx.page.images[img].size();
                let (mut ox, mut oy) = (0.0, 0.0);
                if let Some(bp) = s.get("background-position") {
                    let a: Vec<f32> = bp.split_whitespace().filter_map(|v| px(v, 0.0, em)).collect();
                    if a.len() == 2 { ox = a[0]; oy = a[1]; }
                }
                let cover = matches!(s.get("background-size"), Some("cover") | Some("100% 100%") | Some("contain"));
                let uv = if cover || (ox == 0.0 && oy == 0.0 && (iw - w).abs() < 2.0) {
                    [0.0, 0.0, 1.0, 1.0]
                } else {
                    [-ox / iw, -oy / ih, (-ox + w) / iw, (-oy + h) / ih]
                };
                cx.out.push((z, clip_draw(Draw::Image { x, y, w, h, img, uv, alpha: me.opacity }, cx.clip)));
            }
        }
        if let Some(t) = text {
            if !t.is_empty() {
                let ac1 = n.attr("data-ac1-font").map(str::to_string);
                if let Some(sf) = ac1.as_ref().and_then(|f| cx.page.sprite_fonts.get(f)) {
                    let hh: f32 = n.attr("data-ac1-h").and_then(|v| v.parse().ok()).unwrap_or(em);
                    sprite_text(cx, sf, &t, x, y + h - hh, hh, me.align, w, z, me.opacity);
                } else {
                    let centre_v = flex && matches!(s.get("align-items"), Some("center"));
                    let ty = if centre_v { y + (h - em) * 0.5 + em * 0.8 } else { y + em * 0.9 };
                    let mut col = me.color;
                    col[3] *= me.opacity;
                    cx.out.push((z + 1, Draw::Text { x, y: ty, w, align: me.align, size: em, font: me.font, text: t, color: col, spacing: me.spacing }));
                }
            }
        }
    }
    let saved = cx.clip;
    if matches!(s.get("overflow"), Some("hidden")) && n.tag != "body" && n.tag != "html" {
        cx.clip = (cx.clip.0.max(x), cx.clip.1.max(y), cx.clip.2.min(x + w), cx.clip.3.min(y + h));
    }
    let mut p = path.to_vec();
    for c in &n.children {
        // AC1 fields: the hidden source span is drawn by its parent
        if c.classes.iter().any(|k| k == "ac1src" || k == "ac1out") {
            continue;
        }
        p.push(c);
        node(cx, &p, (x, y, w, h), &me);
        p.pop();
    }
    cx.clip = saved;
}

#[allow(clippy::too_many_arguments)]
fn sprite_text(cx: &mut Ctx, sf: &SpriteFont, t: &str, x: f32, y: f32, hh: f32, align: u8, w: f32, z: i32, alpha: f32) {
    let scale = hh / sf.h;
    let glyphs: Vec<(f32, f32)> = t.chars().filter_map(|c| {
        let i = (c as u32).checked_sub(32)? as usize;
        sf.u.get(i).copied().flatten()
    }).collect();
    let total: f32 = glyphs.iter().map(|(a, b)| (b - a) * sf.w * scale).sum();
    let mut gx = match align { 1 => x + (w - total) * 0.5, 2 => x + w - total, _ => x };
    // the converter anchors with translate(-100%, ...) against a zero-width
    // box: right-align at x when there is no width
    if w <= 1.0 {
        gx = x - total;
    }
    for (u0, u1) in glyphs {
        let gw = (u1 - u0) * sf.w * scale;
        cx.out.push((z + 1, clip_draw(Draw::Image { x: gx, y, w: gw, h: hh, img: sf.img, uv: [u0, 0.0, u1, 1.0], alpha }, cx.clip)));
        gx += gw;
    }
}

/// The text an element shows: its binding evaluated, else its own text.
fn node_text(cx: &Ctx, n: &Node) -> Option<String> {
    // an AC1 field: the bound value sits in its hidden .ac1src child
    if n.attr("data-ac1-font").is_some() {
        let src = n.children.iter().find(|c| c.classes.iter().any(|k| k == "ac1src"))?;
        let inner = src.children.first().unwrap_or(src);
        return Some(inner.attr("data-bind-value").and_then(|b| eval_binding(b, cx.st)).unwrap_or_else(|| inner.text.clone()));
    }
    if let Some(b) = n.attr("data-bind-value") {
        return Some(eval_binding(b, cx.st).unwrap_or_else(|| n.text.clone()));
    }
    if n.text.is_empty() { None } else { Some(n.text.clone()) }
}

fn collect_bindings(n: &Node, out: &mut Vec<String>) {
    if let Some(b) = n.attr("data-bind-value") {
        out.push(b.to_string());
    }
    for c in &n.children {
        collect_bindings(c, out);
    }
}

fn first_page(n: &Node) -> Option<String> {
    if n.classes.iter().any(|c| c == "page") && !n.id.is_empty() {
        return Some(n.id.clone());
    }
    n.children.iter().find_map(first_page)
}

fn scripts(n: &Node, out: &mut String) {
    if n.tag == "script" {
        out.push_str(&n.text);
        out.push('\n');
    }
    for c in &n.children {
        scripts(c, out);
    }
}

/// `var AC1_FONTS = {...}` from a converted page.
fn ac1_fonts(js: &str) -> Vec<(String, String, f32, f32, Vec<Option<(f32, f32)>>)> {
    let Some(p) = js.find("AC1_FONTS") else { return Vec::new() };
    let Some(b) = js[p..].find('{') else { return Vec::new() };
    let start = p + b;
    let mut depth = 0;
    let mut end = start;
    for (i, ch) in js[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => { depth -= 1; if depth == 0 { end = start + i + 1; break; } }
            _ => {}
        }
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&js[start..end]) else { return Vec::new() };
    let mut out = Vec::new();
    if let Some(o) = v.as_object() {
        for (name, f) in o {
            let w = f["w"].as_f64().unwrap_or(4096.0) as f32;
            let h = f["h"].as_f64().unwrap_or(64.0) as f32;
            let src = f["src"].as_str().unwrap_or("").to_string();
            let u = f["u"].as_array().map(|a| a.iter().map(|e| {
                e.as_array().and_then(|p| Some((p.first()?.as_f64()? as f32, p.get(1)?.as_f64()? as f32)))
            }).collect()).unwrap_or_default();
            out.push((name.clone(), src, w, h, u));
        }
    }
    out
}

fn decode_png(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut r = dec.read_info().ok()?;
    let mut buf = vec![0u8; r.output_buffer_size()];
    let info = r.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width, info.height);
    let px = (w * h) as usize;
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf[..px * 4].to_vec(),
        png::ColorType::Rgb => buf[..px * 3].chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => buf[..px * 2].chunks_exact(2).flat_map(|c| [c[0], c[0], c[0], c[1]]).collect(),
        png::ColorType::Grayscale => buf[..px].iter().flat_map(|&g| [g, g, g, 255]).collect(),
        _ => return None,
    };
    Some((w, h, rgba))
}

fn dir_of(p: &str) -> String {
    let p = p.replace('\\', "/");
    match p.rfind('/') {
        Some(i) => p[..i].to_string(),
        None => String::new(),
    }
}

fn join(dir: &str, rel: &str) -> String {
    let rel = rel.replace('\\', "/");
    let rel = rel.trim_start_matches('/');
    let mut parts: Vec<&str> = dir.split('/').filter(|s| !s.is_empty()).collect();
    for seg in rel.split('/') {
        match seg {
            ".." => { parts.pop(); }
            "." | "" => {}
            s => parts.push(s),
        }
    }
    parts.join("/")
}

/// Read every display page the actor names.
pub fn load(vfs: &mut Vfs, displays: &[crate::fxdata::DisplayDef], blob: &mut Vec<u8>,
            textures: &mut Vec<TextureEntry>, tex_index: &mut HashMap<String, usize>,
            problems: &mut Vec<String>) -> Vec<Page> {
    let mut pages = Vec::new();
    for d in displays {
        let Some(hk) = vfs.find(&d.html, None) else {
            problems.push(format!("display page not found: {}", d.html));
            continue;
        };
        let html = String::from_utf8_lossy(&vfs.read(&hk)).into_owned();
        let root = parse_html(&html);
        let dir = dir_of(&d.html);
        // stylesheets: <link href=...>
        let mut rules = Vec::new();
        let mut faces = Vec::new();
        let mut links = Vec::new();
        fn walk_links(n: &Node, out: &mut Vec<String>) {
            if n.tag == "link" {
                if let Some(h) = n.attr("href") { out.push(h.to_string()); }
            }
            for c in &n.children { walk_links(c, out); }
        }
        walk_links(&root, &mut links);
        let mut missing = Vec::new();
        for l in &links {
            let p = join(&dir, l);
            match vfs.find(&p, None) {
                Some(k) if k.ends_with(&p.to_lowercase().rsplit('/').next().unwrap_or("").to_string()) => {
                    let css = String::from_utf8_lossy(&vfs.read(&k)).into_owned();
                    let (r, f) = parse_css(&css, rules.len());
                    rules.extend(r);
                    faces.extend(f.into_iter().map(|(fam, url)| (fam, join(&dir_of(&p), &url))));
                }
                _ => missing.push(format!("css {p}")),
            }
        }
        // inline <style>
        fn walk_style(n: &Node, out: &mut String) {
            if n.tag == "style" { out.push_str(&n.text); }
            for c in &n.children { walk_style(c, out); }
        }
        let mut inline = String::new();
        walk_style(&root, &mut inline);
        if !inline.is_empty() {
            let (r, f) = parse_css(&inline, rules.len() + 100000);
            rules.extend(r);
            faces.extend(f.into_iter().map(|(fam, url)| (fam, join(&dir, &url))));
        }
        // fonts
        let mut fonts = Vec::new();
        let mut font_names = HashMap::new();
        for (fam, url) in &faces {
            match vfs.find(url, None) {
                Some(k) => {
                    let b = vfs.read(&k);
                    if ab_glyph::FontRef::try_from_slice(&b).is_ok() {
                        font_names.insert(fam.to_lowercase(), fonts.len());
                        fonts.push(std::sync::Arc::new(b));
                    } else {
                        missing.push(format!("font unreadable {url}"));
                    }
                }
                None => missing.push(format!("font {url}")),
            }
        }
        if fonts.is_empty() {
            // the shared display fonts, so text still shows
            if let Some(k) = vfs.find("content/cars/common_assets/displays/fonts/8bitoperatorplus-regular.ttf", None) {
                fonts.push(std::sync::Arc::new(vfs.read(&k)));
            }
        }
        // images: every url() in the sheets
        let mut urls: Vec<String> = Vec::new();
        for r in &rules {
            for (k, v, _) in &r.decls {
                if k.starts_with("background") {
                    if let Some(u) = css_url(v) { urls.push(u); }
                }
            }
        }
        let mut image_ix = HashMap::new();
        let mut images = Vec::new();
        let mut load_img = |vfs: &mut Vfs, url: &str, images: &mut Vec<Image>, missing: &mut Vec<String>| -> Option<usize> {
            let p = join(&dir, url);
            let low = p.to_lowercase();
            let png_path = if low.ends_with(".texture") { p[..p.len() - 8].to_string() + ".png" } else { p.clone() };
            if low.ends_with(".texture") {
                if let Some(i) = crate::load::push_fx_texture(vfs, &p, blob, textures, tex_index) {
                    let t = &textures[i];
                    images.push(Image::Tex(i, t.width, t.height));
                    return Some(images.len() - 1);
                }
            }
            if let Some(k) = vfs.find(&png_path, None) {
                if let Some((w, h, px)) = decode_png(&vfs.read(&k)) {
                    images.push(Image::Rgba(w, h, std::sync::Arc::new(px)));
                    return Some(images.len() - 1);
                }
            }
            missing.push(format!("image {p}"));
            None
        };
        fn walk_img(n: &Node, out: &mut Vec<String>) {
            if n.tag == "img" {
                if let Some(s) = n.attr("src") { out.push(s.to_string()); }
            }
            for c in &n.children { walk_img(c, out); }
        }
        walk_img(&root, &mut urls);
        for u in urls {
            if image_ix.contains_key(&u.to_lowercase()) {
                continue;
            }
            if let Some(i) = load_img(vfs, &u, &mut images, &mut missing) {
                image_ix.insert(u.to_lowercase(), i);
            }
        }
        let mut js = String::new();
        scripts(&root, &mut js);
        let mut sprite_fonts = HashMap::new();
        for (name, src, w, h, u) in ac1_fonts(&js) {
            if let Some(img) = load_img(vfs, &src, &mut images, &mut missing) {
                sprite_fonts.insert(name, SpriteFont { img, u, w, h });
            }
        }
        let mut bindings = Vec::new();
        collect_bindings(&root, &mut bindings);
        let default_page = first_page(&root).unwrap_or_else(|| "race".into());
        for m in &missing {
            problems.push(format!("display {}: {m}", crate::load::base_of(&hk)));
        }
        let (mut w, mut h) = (d.width.max(16), d.height.max(16));
        if w > 2048 || h > 2048 {
            let k = 2048.0 / w.max(h) as f32;
            w = (w as f32 * k) as u32;
            h = (h as f32 * k) as u32;
        }
        let _ = (w, h);
        pages.push(Page {
            html_path: d.html.clone(),
            material: crate::load::norm_mat_path(&d.material),
            width: d.width.max(16),
            height: d.height.max(16),
            root,
            rules,
            font_names,
            fonts,
            image_ix,
            images,
            sprite_fonts,
            default_page,
            bindings,
            missing,
            style_cache: Default::default(),
        });
    }
    pages
}

// ------------------------------------------------- expressions + dynamics --

/// A ModelCurrentCar field as a number, for the car state.
fn field(name: &str, st: &FxState) -> Option<f64> {
    let kmh = (st.speed_mps * 3.6) as f64;
    let ign = st.rpm > 0.0 || st.lights > 0;
    let n = name.rsplit('.').next().unwrap_or(name);
    Some(match n {
        "rpm" => st.rpm as f64,
        "display_speed_kmh" | "speed_kmh" | "speed" => kmh,
        "display_speed_mph" | "speed_mph" => kmh / 1.609344,
        "display_speed_ms" | "speed_ms" => st.speed_mps as f64,
        "gear_int" => (st.gear + 1) as f64,
        "gas" | "throttle" | "gas_percent" => st.throttle as f64,
        "brake" | "brake_percent" => st.brake as f64,
        "is_ignition_on" => if ign { 1.0 } else { 0.0 },
        "is_engine_running" => if st.rpm > 0.0 { 1.0 } else { 0.0 },
        "main_light_stage" => st.lights as f64,
        "water_temperature_c" => if st.rpm > 0.0 { 88.0 } else { 20.0 },
        "oil_temperature" | "oil_temperature_c" => if st.rpm > 0.0 { 102.0 } else { 20.0 },
        "fuel_liter_current_quantity" => 42.0,
        "km_per_fuel_liter" => 9.5,
        "current_lap_time_ms" => st.lap.map_or(0.0, |l| l.current_ms),
        "last_laptime_ms" => st.lap.map_or(0.0, |l| l.last_ms),
        "best_laptime_ms" => st.lap.map_or(0.0, |l| l.best_ms),
        "delta_time_ms" => st.lap.map_or(0.0, |l| l.delta_ms),
        "predicted_lap_time_ms" => st.lap.map_or(0.0, |l| l.predicted_ms),
        "total_lap_count" => st.lap.map_or(0.0, |l| l.laps as f64),
        "is_last_lap" => 0.0,
        "handbrake_percent" => 0.0,
        _ => return None,
    })
}

struct Expr<'a> {
    s: &'a [u8],
    i: usize,
    st: &'a FxState,
}

impl<'a> Expr<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        if self.i < self.s.len() && self.s[self.i] == c {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn sum(&mut self) -> Option<f64> {
        let mut v = self.prod()?;
        loop {
            if self.eat(b'+') {
                v += self.prod()?;
            } else if self.eat(b'-') {
                v -= self.prod()?;
            } else {
                return Some(v);
            }
        }
    }
    fn prod(&mut self) -> Option<f64> {
        let mut v = self.unary()?;
        loop {
            if self.eat(b'*') {
                v *= self.unary()?;
            } else if self.eat(b'/') {
                let d = self.unary()?;
                v = if d == 0.0 { 0.0 } else { v / d };
            } else {
                return Some(v);
            }
        }
    }
    fn unary(&mut self) -> Option<f64> {
        if self.eat(b'-') {
            return Some(-self.unary()?);
        }
        self.atom()
    }
    fn args(&mut self) -> Option<Vec<f64>> {
        let mut a = Vec::new();
        if self.eat(b')') {
            return Some(a);
        }
        loop {
            a.push(self.sum()?);
            if self.eat(b')') {
                return Some(a);
            }
            if !self.eat(b',') {
                return None;
            }
        }
    }
    fn atom(&mut self) -> Option<f64> {
        self.ws();
        if self.i >= self.s.len() {
            return None;
        }
        if self.eat(b'(') {
            let v = self.sum()?;
            return if self.eat(b')') { Some(v) } else { None };
        }
        let st = self.i;
        if self.s[self.i].is_ascii_digit() || self.s[self.i] == b'.' {
            while self.i < self.s.len() && (self.s[self.i].is_ascii_digit() || self.s[self.i] == b'.') {
                self.i += 1;
            }
            return std::str::from_utf8(&self.s[st..self.i]).ok()?.parse().ok();
        }
        if self.s[self.i..].starts_with(b"{{") {
            let e = self.s[self.i..].windows(2).position(|w| w == b"}}")? + self.i;
            let mut name = String::from_utf8_lossy(&self.s[self.i + 2..e]).into_owned();
            self.i = e + 2;
            // {{ModelCurrentCar}}.field.sub
            while self.i < self.s.len() && self.s[self.i] == b'.' {
                let fs = self.i + 1;
                let mut j = fs;
                while j < self.s.len() && (self.s[j].is_ascii_alphanumeric() || self.s[j] == b'_') {
                    j += 1;
                }
                let part = String::from_utf8_lossy(&self.s[fs..j]).into_owned();
                if part == "toFixed" || j == fs {
                    break;
                }
                name.push('.');
                name.push_str(&part);
                self.i = j;
            }
            return field(&name, self.st);
        }
        while self.i < self.s.len()
            && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_' || self.s[self.i] == b'.')
        {
            self.i += 1;
        }
        let id = std::str::from_utf8(&self.s[st..self.i]).ok()?.to_string();
        if !self.eat(b'(') {
            return None;
        }
        let a = self.args()?;
        let g = |k: usize| a.get(k).copied().unwrap_or(0.0);
        Some(match id.as_str() {
            "ksUI.clamp" | "clamp" => g(0).max(g(1)).min(g(2)),
            "Math.round" => g(0).round(),
            "Math.floor" => g(0).floor(),
            "Math.ceil" => g(0).ceil(),
            "Math.abs" => g(0).abs(),
            "Math.min" => g(0).min(g(1)),
            "Math.max" => g(0).max(g(1)),
            "CarDisplay.delayValue" => g(0),
            "CarDisplay.convert.msToKmh" => g(0) * 3.6,
            _ => return None,
        })
    }
}

/// The top-level comma-separated arguments of `f(...)` when `s` is exactly
/// one call of `name`.
fn call_args<'a>(s: &'a str, name: &str) -> Option<Vec<&'a str>> {
    let rest = s.strip_prefix(name)?.trim_start().strip_prefix('(')?;
    let (mut depth, mut quote, mut start) = (0i32, None::<char>, 0usize);
    let mut out = Vec::new();
    for (i, c) in rest.char_indices() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), _) => {}
            (None, '\'' | '"') => quote = Some(c),
            (None, '(' | '[' | '{') => depth += 1,
            (None, ',') if depth == 0 => {
                out.push(rest[start..i].trim());
                start = i + 1;
            }
            (None, ')' | ']' | '}') if depth > 0 => depth -= 1,
            (None, ')') => {
                // the call must end the expression
                if !rest[i + 1..].trim().is_empty() {
                    return None;
                }
                out.push(rest[start..i].trim());
                return Some(out);
            }
            _ => {}
        }
    }
    None
}

/// A literal argument: 'text' / "text".
fn str_arg(a: Option<&&str>) -> Option<String> {
    let a = a?.trim();
    let q = a.chars().next()?;
    (matches!(q, '\'' | '"') && a.len() >= 2 && a.ends_with(q)).then(|| a[1..a.len() - 1].to_string())
}

/// A number / boolean argument (false = 0, true = 1).
fn num_arg(a: Option<&&str>, st: &FxState) -> Option<f64> {
    match a?.trim() {
        "false" | "null" | "undefined" => Some(0.0),
        "true" => Some(1.0),
        x => eval_number(x, st),
    }
}

fn pad(v: u64, n: usize) -> String {
    format!("{v:0n$}")
}

/// ksUI.formatTimeExt (uiresources/js/components.js), for ms.
fn format_time_ext(t: f64, format: i64, msec: usize, prefix: &str, sep: &str, whenzero: Option<String>) -> String {
    if t.is_nan() {
        return "-".into();
    }
    if let (Some(z), true) = (whenzero, t == 0.0) {
        return z;
    }
    let mut t = t.round();
    let mut negative = t < 0.0;
    if msec < 3 {
        if msec == 1 {
            t = (t / 100.0).round() * 100.0;
        }
        if msec == 2 {
            t = (t / 10.0).round() * 10.0;
        }
        negative = t + 20.0 / (msec as f64) < 0.0;
    }
    let t = t.abs() as u64;
    let (h, s, m) = (t / 3_600_000, t / 1000, t / 60_000);
    // as the JS: seconds / minutes padded when a bigger unit leads, or format 1
    let mut ss = if m != 0 || format == 1 { pad(s % 60, 2) } else { s.to_string() };
    let mv = if h != 0 || format == 1 { m % 60 } else { m };
    let mut mm = if h != 0 || format == 1 { pad(mv, 2) } else { mv.to_string() };
    let mut ms = pad(t % 1000, 3);
    ms.truncate(msec.min(3));
    let mut out: Vec<String> = Vec::new();
    if h != 0 {
        out.push(h.to_string());
    }
    if format == 3 {
        mm = pad(mv, 2);
        ss = pad(s % 60, 2);
        out.push(mm);
    } else if mv != 0 || format == 2 {
        out.push(mm);
        ss = pad(s % 60, 2);
    }
    out.push(ss);
    let sign = if negative { "-" } else { prefix };
    if format != 1 && msec > 0 {
        out.push(ms);
        let j = out.join(sep);
        let k = j.rfind(sep).unwrap_or(0);
        return format!("{sign}{}.{}", &j[..k], &j[k + sep.len()..]);
    }
    format!("{sign}{}", out.join(sep))
}

/// ksUI.formatTime (same file), for ms.
fn format_time(t: f64, hidemsec: bool, msec: usize, prefix: &str, msec_sep: &str, pad_all: bool) -> String {
    if t.is_nan() {
        return "-".into();
    }
    let negative = t < 0.0;
    let t = t.abs().round() as u64;
    let (h, s, m) = (t / 3_600_000, t / 1000, t / 60_000);
    let ss = if m != 0 || hidemsec || pad_all { pad(s % 60, 2) } else { s.to_string() };
    let mv = if h != 0 || hidemsec || pad_all { m % 60 } else { m };
    let mm = if h != 0 || hidemsec || pad_all { pad(mv, 2) } else { mv.to_string() };
    let mut ms = pad(t % 1000, 3);
    ms.truncate(msec.min(3));
    let mut out: Vec<String> = Vec::new();
    if h != 0 {
        out.push(h.to_string());
    }
    if mv != 0 || hidemsec || pad_all {
        out.push(mm);
    }
    out.push(ss);
    let sign = if negative { "-" } else { prefix };
    if hidemsec {
        return format!("{sign}{}", out.join(":"));
    }
    out.push(ms);
    let j = out.join(":");
    let k = j.rfind(':').unwrap_or(0);
    format!("{sign}{}{msec_sep}{}", &j[..k], &j[k + 1..])
}

/// The time formatters the pages wrap around lap times: ksUI.formatTimeExt
/// / formatTime / pad, the per-car CarDisplay.laptime / delta (Lotus Emira's
/// display.js), CarDisplay.delayValue (shows its first argument).
fn eval_call(e: &str, st: &FxState) -> Option<String> {
    if let Some(a) = call_args(e, "CarDisplay.delayValue") {
        return eval_binding(a.first()?, st);
    }
    if let Some(a) = call_args(e, "ksUI.formatTimeExt") {
        let t = num_arg(a.first(), st)?;
        let format = num_arg(a.get(1), st).unwrap_or(1.0) as i64;
        let msec = num_arg(a.get(3), st).unwrap_or(3.0) as usize;
        return Some(format_time_ext(t, format, msec, &str_arg(a.get(4)).unwrap_or_default(),
                                    &str_arg(a.get(5)).unwrap_or(":".into()), str_arg(a.get(6))));
    }
    if let Some(a) = call_args(e, "ksUI.formatTime") {
        let t = num_arg(a.first(), st)?;
        let hide = num_arg(a.get(1), st).map_or(true, |v| v != 0.0);
        let msec = num_arg(a.get(3), st).unwrap_or(3.0) as usize;
        let pad_all = a.get(6).map_or(false, |x| !matches!(x.trim(), "false" | "0" | "null"));
        return Some(format_time(t, hide, msec, &str_arg(a.get(4)).unwrap_or_default(),
                                &str_arg(a.get(5)).unwrap_or(".".into()), pad_all));
    }
    if let Some(a) = call_args(e, "ksUI.pad") {
        let v = num_arg(a.first(), st)?;
        let n = num_arg(a.get(1), st).unwrap_or(2.0) as usize;
        return Some(pad(v.abs().floor() as u64, n));
    }
    if let Some(a) = call_args(e, "CarDisplay.laptime") {
        let t = num_arg(a.first(), st)?;
        if num_arg(a.get(1), st).map_or(false, |v| v != 0.0) && t == 0.0 {
            return Some("--".into());
        }
        return Some(format_time(t, false, 3, "", ".", false));
    }
    if let Some(a) = call_args(e, "CarDisplay.delta") {
        let t = num_arg(a.first(), st)?;
        if t == 0.0 {
            return Some(" ".into());
        }
        return Some(format!("{}{}", if t > 0.0 { "+" } else { "" }, format_time(t, false, 3, "", ".", false)));
    }
    None
}

#[cfg(test)]
mod time_tests {
    use super::*;

    #[test]
    fn ksui_formats() {
        let st = FxState::default();
        // 1:57.812 the ways the pages ask for it
        assert_eq!(format_time_ext(117812.0, 2, 2, "", ":", None), "1:57.81");
        assert_eq!(format_time_ext(117812.0, 3, 3, "", ":", None), "01:57.812");
        assert_eq!(format_time_ext(57812.0, 2, 2, "", ":", None), "0:57.81");
        assert_eq!(format_time_ext(0.0, 2, 2, "", ":", Some("--".into())), "--");
        assert_eq!(format_time(117812.0, false, 3, "", ".", false), "1:57.812");
        assert_eq!(format_time(-320.0, false, 3, "+", ".", false), "-0.320");
        assert_eq!(format_time(320.0, false, 2, "+", ".", false), "+0.32");
        assert_eq!(eval_call("ksUI.pad({{ModelCurrentCar}}.low_frequency.total_lap_count+1, 2)", &st).as_deref(), Some("01"));
        assert_eq!(eval_call("CarDisplay.delayValue(ksUI.formatTimeExt({{ModelCurrentCar}}.low_frequency.last_laptime_ms, 2, false, 2),1,50)", &st).as_deref(), Some("0:00.00"));
    }
}

/// A full arithmetic evaluation of a binding (fields, + - * /, clamp,
/// Math.*), ignoring a trailing `.toFixed(n)`.
pub fn eval_number(expr: &str, st: &FxState) -> Option<f64> {
    let t = expr.trim();
    let t = t.split(".toFixed(").next().unwrap_or(t);
    let mut e = Expr { s: t.as_bytes(), i: 0, st };
    let v = e.sum()?;
    e.ws();
    if e.i == e.s.len() { Some(v) } else { None }
}

/// The tree as this car state shows it: `data-bind-class-toggle` classes
/// added, `data-bind-style-<prop>` written into the inline style. (Anything
/// else the page script does is not modelled.)
pub fn apply_dynamics(n: &Node, st: &FxState) -> Node {
    let mut o = n.clone();
    if let Some(t) = n.attr("data-bind-class-toggle") {
        for part in t.split(';') {
            if let Some((cls, ex)) = part.split_once(':') {
                if eval_number(ex, st).map_or(false, |v| v != 0.0) {
                    o.classes.push(cls.trim().to_string());
                }
            }
        }
    }
    let mut extra = String::new();
    for (k, v) in &n.attrs {
        if let Some(prop) = k.strip_prefix("data-bind-style-") {
            if let Some(x) = eval_number(v, st) {
                extra.push_str(&format!(";{prop}:{x}px"));
            }
        }
    }
    if !extra.is_empty() {
        let base = o.attr("style").unwrap_or("").to_string();
        o.attrs.retain(|(k, _)| k != "style");
        o.attrs.push(("style".into(), base + &extra));
    }
    o.children = n.children.iter().map(|c| apply_dynamics(c, st)).collect();
    o
}
