//! Colour grading: exposure, then a 32^3 grading LUT baked at startup.
//!
//! The game grades through a LUT built from its `.postprocessing` presets.
//! This module bakes an equivalent LUT from those presets' numbers with an
//! independent implementation built from public definitions:
//!   * white balance: CIE daylight chromaticity for the colour temperature,
//!     adapted to D65 with the Bradford transform (ICC / CIE CAT);
//!   * colour correction per tonal range (shadows / midtones / highlights,
//!     blended by luma): saturation about luma, contrast about 18% grey,
//!     gamma, gain and offset - the usual grading controls the presets name;
//!   * tone curve: Hajime Uchimura's "GT" curve (CEDEC 2017, "HDR theory and
//!     practice"), with the preset's contrast, mid-section and black level;
//!   * highlights running to white: Timothy Lottes' crosstalk / "path to
//!     white" (GDC 2016, "Advanced Techniques and Optimization of HDR Color
//!     Pipelines"), driven by the preset's pathToWhite pair;
//!   * the LUT is addressed through our own log2 shaper (`shaper`), so 16
//!     stops of HDR fit 32 texels; the WGSL side applies the same formula.
//! Luma uses the ACES AP1 luminance weights.
//!
//! Two LUTs are stacked along z in one 32x32x64 texture: the studio grade in
//! the lower half, the outdoor grade in the upper. `Camera.grade.z` picks one,
//! so the same texture serves a car and a track without a rebind.
//!
//! The preset numbers below are the game's own settings (a handful of floats
//! per preset, read from the presets with its protobuf schema), not game
//! files or code.

/// Colour-correction inputs for one tonal range, as the presets store them:
/// an RGB triple and a scalar `.w` that scales it (offset: added).
#[derive(Clone, Copy)]
pub struct Cc {
    pub sat: [f32; 4],
    pub con: [f32; 4],
    pub gam: [f32; 4],
    pub gain: [f32; 4],
    pub off: [f32; 4],
}

const CC_ID: Cc = Cc {
    sat: [1.0; 4],
    con: [1.0; 4],
    gam: [1.0; 4],
    gain: [1.0; 4],
    off: [0.0; 4],
};

#[derive(Clone, Copy)]
pub struct Preset {
    /// tonemapper: contrast (a), mid_section_start (m), mid_section_length
    /// (l), black_level (c). maxBrightness P is 1 for SDR.
    pub contrast: f32,
    pub mid_start: f32,
    pub mid_len: f32,
    pub black: f32,
    /// colorGrading.pathToWhite: {pathToWhite, overExposeSaturation}
    pub cross_talk: f32,
    pub cross_sat: f32,
    /// whiteBalance {temperature, tint}, in the editor's units (see wb_factor)
    pub wb_temp: f32,
    pub wb_tint: f32,
    pub master: Cc,
    pub shadows: Cc,
    pub mid: Cc,
    pub high: Cc,
    pub max_shadows: f32,
    pub min_highlights: f32,
}

/// natural1.postprocessing - the most common track preset (the track grade
/// until 2026-11-01; EVOVIEW_GRADE_TRACK=natural1). Its path to white is
/// 100 / 10, i.e. none: bright colours stay saturated, which the game with
/// Natural 7 does not do (no scene pixel brighter than V 0.85 above HSV S
/// 0.55 in Josh's Oct-8 cockpit shot; natural1 allows S 1.0 there).
pub const NATURAL1: Preset = Preset {
    contrast: 1.0,
    mid_start: 0.4,
    mid_len: 0.1,
    black: 1.2,
    cross_talk: 100.0,
    cross_sat: 10.0,
    wb_temp: 0.0,
    wb_tint: 0.0,
    master: Cc { gam: [1.0, 1.0, 1.0, 0.963_788_45], ..CC_ID },
    shadows: CC_ID,
    mid: CC_ID,
    high: CC_ID,
    max_shadows: 0.1,
    min_highlights: 0.5,
};

/// natural6/natural7.postprocessing: the in-game "Natural 7" preset
/// (VideoSettings.graphics.post_processing_preset; Josh's setting). Linear
/// toe (black 1.0) from 0.15, shoulder from ~0.4, a strong path to white
/// (4.5 / 1: bright colours run to white), shadow gamma 0.8 and midtone
/// gamma 0.9 (applied as 1/gamma: deeper, richer darks).
/// ⚠ White balance 0, not the file's static -12: the preset's
/// dynamicSettings drive temperature, tint, master gain and gamma by sun
/// angle (natural6/white_balance.curve etc., sky::curves), and those curves
/// sit at 0 / 1 all day. They replace the static values (the exposure
/// curve's -1.2 by day likewise stands in for the static -1.5). The dusk /
/// night values are not baked in (one LUT per run).
pub const NATURAL7: Preset = Preset {
    contrast: 1.0,
    mid_start: 0.15,
    mid_len: 0.3,
    black: 1.0,
    cross_talk: 4.5,
    cross_sat: 1.0,
    wb_temp: 0.0,
    wb_tint: 0.0,
    master: CC_ID,
    shadows: Cc { gam: [1.0, 1.0, 1.0, 0.8], ..CC_ID },
    mid: Cc { gam: [1.0, 1.0, 1.0, 0.9], ..CC_ID },
    high: CC_ID,
    max_shadows: 0.1,
    min_highlights: 0.5,
};

/// The car thumbnails / studio. `system/thumbnails/thumbnails.graphicsoverride`
/// sets exposure and white balance (-5, -5) but leaves the tonemapper and the
/// ColorCorrect ranges unset, so those are the engine defaults:
/// `editor/default.postprocessing` minus its per-range grade (the thumbnail
/// override's colour-grading block is all-unset).
///
/// ⚠ `black` is FITTED, not the engine default 1.0: the thumbnails crush the
/// darkest 2% of the car (tyres, vents) to pure black, and 1.0 left them at
/// ~12/255. 1.3 brought lookcmp p2 from +12 to +5 with everything else flat.
pub const STUDIO: Preset = Preset {
    contrast: 1.0,
    mid_start: 0.15,
    mid_len: 0.3,
    black: 1.3,
    cross_talk: 50.0,
    cross_sat: 100.0,
    wb_temp: -5.0,
    wb_tint: -5.0,
    master: CC_ID,
    shadows: CC_ID,
    mid: CC_ID,
    high: CC_ID,
    max_shadows: 0.1,
    min_highlights: 0.5,
};

/// A preset by name, for EVOVIEW_GRADE_STUDIO / EVOVIEW_GRADE_TRACK.
pub fn by_name(n: &str) -> Option<Preset> {
    match n.trim().to_lowercase().as_str() {
        "natural1" => Some(NATURAL1),
        "natural7" | "natural 7" => Some(NATURAL7),
        "studio" | "thumbnail" => Some(STUDIO),
        "default" | "editor" => Some(Preset {
            master: Cc { sat: [1.0, 1.0, 1.0, 1.1], ..CC_ID },
            shadows: Cc { gam: [1.0, 1.0, 1.0, 0.8], gain: [0.9, 0.9, 1.5, 1.0], ..CC_ID },
            mid: Cc { gam: [1.0, 1.0, 1.0, 0.9], ..CC_ID },
            wb_temp: -12.0,
            wb_tint: 0.0,
            ..STUDIO
        }),
        // evoview's own outdoor look: Natural 7 with more colour and
        // contrast and a gentler path to white (Natural 7 bleached sunlit
        // ground - Laguna's sand read as snow). EVOVIEW_VIVID="sat,contrast,
        // path_to_white" tunes it
        "vivid" => {
            let f: Vec<f32> = crate::envc::var("EVOVIEW_VIVID").unwrap_or_default().split(',').filter_map(|v| v.trim().parse().ok()).collect();
            let g = |i: usize, d: f32| f.get(i).copied().unwrap_or(d);
            Some(Preset {
                contrast: g(1, 1.15),
                cross_talk: g(2, 12.0),
                master: Cc { sat: [1.0, 1.0, 1.0, g(0, 1.25)], ..CC_ID },
                ..NATURAL7
            })
        }
        "flat" => Some(Preset {
            contrast: 1.0, mid_start: 0.22, mid_len: 0.1, black: 1.0,
            cross_talk: 50.0, cross_sat: 100.0, wb_temp: 0.0, wb_tint: 0.0,
            ..STUDIO
        }),
        _ => None,
    }
}

/// The master grade applies on top of each range: multiply the gains,
/// add the offsets (per channel, scalar `.w` folded in).
fn combine(m: &Cc, r: &Cc) -> [[f32; 3]; 5] {
    let scaled = |v: [f32; 4]| [v[0] * v[3], v[1] * v[3], v[2] * v[3]];
    let shifted = |v: [f32; 4]| [v[0] + v[3], v[1] + v[3], v[2] + v[3]];
    let mul = |a: [f32; 4], b: [f32; 4]| {
        let (x, y) = (scaled(a), scaled(b));
        [x[0] * y[0], x[1] * y[1], x[2] * y[2]]
    };
    let add = |a: [f32; 4], b: [f32; 4]| {
        let (x, y) = (shifted(a), shifted(b));
        [x[0] + y[0], x[1] + y[1], x[2] + y[2]]
    };
    [mul(m.sat, r.sat), mul(m.con, r.con), mul(m.gam, r.gam), mul(m.gain, r.gain), add(m.off, r.off)]
}

fn mat3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Linear sRGB (D65) <-> CIE XYZ (IEC 61966-2-1).
const SRGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.412_456_4, 0.357_576_1, 0.180_437_5],
    [0.212_672_9, 0.715_152_2, 0.072_175_0],
    [0.019_333_9, 0.119_192_0, 0.950_304_1],
];
const XYZ_TO_SRGB: [[f32; 3]; 3] = [
    [3.240_454_2, -1.537_138_5, -0.498_531_4],
    [-0.969_266_0, 1.876_010_8, 0.041_556_0],
    [0.055_643_4, -0.204_025_9, 1.057_225_2],
];
/// The Bradford cone response matrix (CIE CAT, as ICC uses it).
const BRADFORD: [[f32; 3]; 3] = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// CIE daylight locus: chromaticity of the D-illuminant at `t` kelvin.
fn daylight_xy(t: f32) -> (f32, f32) {
    let t = t.clamp(4000.0, 25000.0);
    let x = if t <= 7000.0 {
        -4.6070e9 / (t * t * t) + 2.9678e6 / (t * t) + 0.09911e3 / t + 0.244063
    } else {
        -2.0064e9 / (t * t * t) + 1.9018e6 / (t * t) + 0.24748e3 / t + 0.237040
    };
    (x, -3.0 * x * x + 2.87 * x - 0.275)
}

fn xy_to_xyz(x: f32, y: f32) -> [f32; 3] {
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// White balance as a matrix on linear sRGB: a source white shifted by the
/// preset (temperature in hundreds of kelvin from 6500 K, NEGATIVE = cooler
/// output; tint as tint/1000 in y, negative = greener; signs chosen by
/// matching the game's car thumbnails, EVOVIEW_WB_SIGN=-1 flips them),
/// adapted to D65 by Bradford.
fn wb_matrix(temp: f32, tint: f32) -> Option<[[f32; 3]; 3]> {
    if temp == 0.0 && tint == 0.0 {
        return None;
    }
    let sg: f32 = crate::envc::var("EVOVIEW_WB_SIGN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let (temp, tint) = (temp * sg, tint * sg);
    let (sx, sy) = daylight_xy(6500.0 + temp * 100.0);
    let src = mat3(&BRADFORD, xy_to_xyz(sx, sy + tint * 0.001));
    let dst = mat3(&BRADFORD, xy_to_xyz(0.312_7, 0.329_0));
    let g = [dst[0] / src[0], dst[1] / src[1], dst[2] / src[2]];
    // M = XYZ_TO_SRGB . inv(B) . diag(g) . B . SRGB_TO_XYZ, built column by column
    let inv_b = inv3(&BRADFORD);
    let mut m = [[0.0f32; 3]; 3];
    for c in 0..3 {
        let mut e = [0.0f32; 3];
        e[c] = 1.0;
        let cone = mat3(&BRADFORD, mat3(&SRGB_TO_XYZ, e));
        let col = mat3(&XYZ_TO_SRGB, mat3(&inv_b, [cone[0] * g[0], cone[1] * g[1], cone[2] * g[2]]));
        for r in 0..3 {
            m[r][c] = col[r];
        }
    }
    Some(m)
}

fn inv3(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let [a, b, c] = m[0];
    let [d, e, f] = m[1];
    let [g, h, i] = m[2];
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let k = 1.0 / det;
    [
        [(e * i - f * h) * k, (c * h - b * i) * k, (b * f - c * e) * k],
        [(f * g - d * i) * k, (a * i - c * g) * k, (c * d - a * f) * k],
        [(d * h - e * g) * k, (b * g - a * h) * k, (a * e - b * d) * k],
    ]
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    if e1 <= e0 {
        return if x >= e1 { 1.0 } else { 0.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// ACES AP1 luminance weights.
const LUMA: [f32; 3] = [0.272_228_7, 0.674_081_7, 0.053_689_5];
const GREY: f32 = 0.18;

/// One grading range applied to a colour: saturation about the luma, then
/// per channel contrast about mid grey, gamma (the presets store 1/gamma
/// style values, so it divides the exponent), gain and offset.
fn correct(c: [f32; 3], luma: f32, k: &[[f32; 3]; 5]) -> [f32; 3] {
    let [sat, con, gam, gain, off] = *k;
    let mut o = [0.0f32; 3];
    for i in 0..3 {
        let saturated = (luma + (c[i] - luma) * sat[i]).max(0.0);
        let contrasted = GREY * (saturated / GREY).powf(con[i]);
        o[i] = contrasted.powf(gam[i].recip()) * gain[i] + off[i];
    }
    o
}

/// Uchimura's GT tone curve for one channel: a power toe below `m`, a line
/// of slope `a` from `m`, an exponential shoulder to the peak `p` = 1.
fn gt_curve(x: f32, a: f32, m: f32, l: f32, black: f32) -> f32 {
    let p = 1.0f32;
    let l0 = (p - m) * l / a;
    let s0 = m + l0;
    let s1 = m + a * l0;
    let c2 = a * p / (p - s1);
    let toe = m * (x / m).powf(black);
    let linear = m + a * (x - m);
    let shoulder = p - (p - s1) * (-c2 * (x - s0) / p).exp();
    let w_toe = 1.0 - smoothstep(0.0, m, x);
    let w_shoulder = if x < s0 { 0.0 } else { 1.0 };
    toe * w_toe + linear * (1.0 - w_toe - w_shoulder) + shoulder * w_shoulder
}

/// Lottes' crosstalk: each channel's ratio to the brightest one is pulled
/// toward 1 (white) as that peak grows, `strength` sharpening when it
/// happens and `sat` how saturated the colour stays on the way.
fn path_to_white(t: [f32; 3], strength: f32, sat: f32) -> [f32; 3] {
    let peak = t[0].max(t[1]).max(t[2]).max(1e-6);
    // the pull starts a touch above black
    let pull = (peak - 0.05).clamp(0.0, 1.0).powf(strength).clamp(0.0, 1.0);
    t.map(|v| {
        let ratio = (v / peak).max(0.0).powf(sat.recip());
        let toward_white = ratio + (1.0 - ratio) * pull;
        (toward_white.powf(sat) * peak).clamp(0.0, 1.0)
    })
}

/// One LUT texel: `lin` is linear HDR colour (already exposed).
pub fn grade(p: &Preset, lin: [f32; 3]) -> [f32; 3] {
    let mut c = match wb_matrix(p.wb_temp, p.wb_tint) {
        Some(m) => mat3(&m, lin),
        None => lin,
    };
    let luma = c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2];
    let w_shadow = 1.0 - smoothstep(0.0, p.max_shadows, luma);
    let w_high = smoothstep(p.min_highlights, 1.0, luma);
    let w_mid = 1.0 - w_shadow - w_high;
    let s = correct(c, luma, &combine(&p.master, &p.shadows));
    let m = correct(c, luma, &combine(&p.master, &p.mid));
    let h = correct(c, luma, &combine(&p.master, &p.high));
    for i in 0..3 {
        c[i] = (s[i] * w_shadow + m[i] * w_mid + h[i] * w_high).max(0.001);
    }
    let t = c.map(|x| gt_curve(x, p.contrast, p.mid_start, p.mid_len, p.black));
    path_to_white(t, p.cross_talk, p.cross_sat)
}

/// The LUT's log2 shaper: linear HDR <-> 0..1 LUT coordinate over 16 stops
/// (2^-10 .. 64). The WGSL side (shader / track / post / fx) does the same:
/// `(log2(x + 1/1024) + 10) / 16`.
pub const SHAPER_STOPS: f32 = 16.0;
pub const SHAPER_FLOOR: f32 = 1.0 / 1024.0;

pub fn shaper(x: f32) -> f32 {
    ((x + SHAPER_FLOOR).log2() + 10.0) / SHAPER_STOPS
}

pub const EDGE: u32 = 32;

/// The shaper's inverse at a texel centre: LUT coordinate -> linear.
fn texel_to_linear(i: u32) -> f32 {
    let u = i as f32 / (EDGE as f32 - 1.0);
    (u * SHAPER_STOPS - 10.0).exp2() - SHAPER_FLOOR
}

fn f16_bits(v: f32) -> u16 {
    // finite, non-negative, <= 1 here: no inf/nan handling needed
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - exp);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    let r = ((exp as u32) << 10) | (man >> 13);
    // round to nearest
    sign | (r + ((man >> 12) & 1)) as u16
}

/// Studio (z 0..32) and outdoor (z 32..64) LUTs as RGBA16F bytes.
pub fn bake_pair(studio: &Preset, outdoor: &Preset) -> Vec<u8> {
    let n = EDGE as usize;
    let mut out = Vec::with_capacity(n * n * n * 2 * 8);
    for p in [studio, outdoor] {
        for b in 0..EDGE {
            for g in 0..EDGE {
                for r in 0..EDGE {
                    let c = grade(p, [texel_to_linear(r), texel_to_linear(g), texel_to_linear(b)]);
                    for v in [c[0], c[1], c[2], 1.0] {
                        out.extend_from_slice(&f16_bits(v).to_le_bytes());
                    }
                }
            }
        }
    }
    out
}

/// Presets in use, honouring EVOVIEW_GRADE_STUDIO / EVOVIEW_GRADE_TRACK.
pub fn presets() -> (Preset, Preset) {
    let pick = |var: &str, dflt: Preset| {
        crate::envc::var(var).ok().and_then(|v| by_name(&v)).unwrap_or(dflt)
    };
    let mut s = pick("EVOVIEW_GRADE_STUDIO", STUDIO);
    // EVOVIEW_TONE=contrast,mid_start,mid_len,black overrides the studio
    // tonemapper (tuning)
    if let Ok(v) = crate::envc::var("EVOVIEW_TONE") {
        let f: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if f.len() == 4 {
            s.contrast = f[0];
            s.mid_start = f[1];
            s.mid_len = f[2];
            s.black = f[3];
        }
    }
    // EVOVIEW_WB=temp,tint overrides the studio white balance (tuning)
    if let Ok(v) = crate::envc::var("EVOVIEW_WB") {
        let f: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if f.len() == 2 {
            s.wb_temp = f[0];
            s.wb_tint = f[1];
        }
    }
    // the track grade is the player's VideoSettings post_processing_preset;
    // Josh's is "Natural 7"
    // evoview's vivid look by default; EVOVIEW_GRADE_TRACK=natural7 is the
    // game's own grade (flatter: the user found it 'too bright, not
    // saturated enough, dead')
    (s, pick("EVOVIEW_GRADE_TRACK", by_name("vivid").unwrap()))
}

pub fn upload(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let (s, o) = presets();
    let data = bake_pair(&s, &o);
    let size = wgpu::Extent3d { width: EDGE, height: EDGE, depth_or_array_layers: EDGE * 2 };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("grading lut"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(EDGE * 8),
            rows_per_image: Some(EDGE),
        },
        size,
    );
    tex.create_view(&Default::default())
}

/// Exposure multipliers (studio, outdoor), applied before the LUT.
///
/// The game divides by an adapted average luminance clamped to the preset's
/// [min, max] brightness and multiplies by 2^compensation. Our scene radiance
/// is not in the game's absolute units (no physical sun or sky), so the
/// adapted value cannot be measured the game's way; these constants are
/// CALIBRATED against the game's own renders instead (car thumbnails for the
/// studio). EVOVIEW_EV adds stops on top, for comparisons.
pub fn exposure() -> (f32, f32) {
    let ev: f32 = crate::envc::var("EVOVIEW_EV").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let k = ev.exp2();
    // outdoors only: half a stop down by default (EVOVIEW_EV_TRACK). With
    // the lower ambient and the vivid grade it brings sunlit ground back
    // from white (Laguna's sand read as snow). A whole stop (with clarity
    // and shafts on) read as overcast and gloomy next to the game.
    let t: f32 = crate::envc::var("EVOVIEW_EV_TRACK").ok().and_then(|v| v.parse().ok()).unwrap_or(-0.5);
    (STUDIO_EXPOSURE * k, OUTDOOR_EXPOSURE * k * t.exp2())
}

/// -0.7 EV, fitted with tools/lookcmp.py at the final integration (with the
/// stronger thumbnail sun; mean car luminance delta ~+1).
pub const STUDIO_EXPOSURE: f32 = 0.615_572_8;
pub const OUTDOOR_EXPOSURE: f32 = 1.0;

/// Scale on the four studio softboxes. EVOVIEW_RIG overrides. Off in the
/// thumbnail studio: the game lights its thumbnails with sky and sun only.
pub fn rig() -> f32 {
    let dflt = if studio() == Studio::Thumbnail { 0.0 } else { RIG };
    crate::envc::var("EVOVIEW_RIG").ok().and_then(|v| v.parse().ok()).unwrap_or(dflt)
}

/// Which environment lights a car.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Studio {
    /// The game's thumbnail scene: dark sphere, sky dome, high sun (default).
    Thumbnail,
    /// The paint-studio cube (studio_1_sdr) + the softbox rig.
    Paint,
}

/// EVOVIEW_STUDIO=paint picks the paint-studio cube; default is the thumbnail.
pub fn studio() -> Studio {
    static S: std::sync::OnceLock<Studio> = std::sync::OnceLock::new();
    *S.get_or_init(|| match crate::envc::var("EVOVIEW_STUDIO").as_deref() {
        Ok("paint") | Ok("cube") => Studio::Paint,
        _ => Studio::Thumbnail,
    })
}

/// Background (clear) colour of the car view, linear. In the thumbnail
/// studio: the sphere wall as the camera sees it past the floor.
pub fn studio_clear() -> [f64; 3] {
    match studio() {
        Studio::Thumbnail => {
            // exactly what the shader's floor tonemaps TH_BACKDROP to, so the
            // floor melts into the background with no horizon line
            // exposure-invariant: TH_BACKDROP is already the exposed value
            let e = 1.0;
            let mut b = TH_BACKDROP;
            if let Ok(v) = crate::envc::var("EVOVIEW_BACKDROP") {
                let f: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if f.len() == 3 {
                    b = [f[0], f[1], f[2]];
                }
            }
            let c = grade(&presets().0, [b[0] * e, b[1] * e, b[2] * e]);
            [c[0] as f64, c[1] as f64, c[2] as f64]
        }
        Studio::Paint => [0.012, 0.013, 0.015],
    }
}

/// `studio_clear` with the backdrop radiance scaled by `k` (the night
/// ambient of the effects, lights::fx_block). k = 1 is studio_clear exactly.
pub fn studio_clear_scaled(k: f32) -> [f64; 3] {
    if (k - 1.0).abs() < 1e-6 || studio() != Studio::Thumbnail {
        let c = studio_clear();
        return if (k - 1.0).abs() < 1e-6 { c } else { [c[0] * k as f64, c[1] * k as f64, c[2] * k as f64] };
    }
    let b = TH_BACKDROP;
    let c = grade(&presets().0, [b[0] * k, b[1] * k, b[2] * k]);
    [c[0] as f64, c[1] as f64, c[2] as f64]
}

/// HDR radiance of the thumbnail sphere's wall as the camera sees it. Must
/// equal TH_BACKDROP in shader.wgsl. Fitted to the thumbnails' backdrop.
pub const TH_BACKDROP: [f32; 3] = [0.0405, 0.0374, 0.0340];

pub const RIG: f32 = 1.0;

/// EVOVIEW_TONEMAP=aces keeps the old ACES curve (for before/after shots).
pub fn use_lut() -> bool {
    !matches!(crate::envc::var("EVOVIEW_TONEMAP").as_deref(), Ok("aces"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn monotonic_and_bounded() {
        let mut last = -1.0;
        for i in 0..EDGE {
            let x = texel_to_linear(i);
            let y = grade(&NATURAL1, [x, x, x])[1];
            assert!(y >= last - 1e-5 && (0.0..=1.0).contains(&y), "{i} {x} {y}");
            last = y;
        }
    }
    #[test]
    fn half_floats() {
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits(0.0), 0);
    }
}
