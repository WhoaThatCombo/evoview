//! AC EVO's pit system on the driven car: pit limiter, pit lane, the box,
//! the stop with its menu and timings, the garage. All of it is the game's
//! own data and rules; where something is a guess it says so.
//!
//! TRACK (content\tracks\<id>\containers, Imola checked):
//! - pitlane_zones*.scene: ZoneActorData (actor f112) {f1000 ZoneData {f1
//!   geometrics [{f1 points: TransformData}], f2 instances [{f101 pitlane
//!   {f1 type 0 Entry / 1 Exit / 2 Average / 3 Main}, f111 speed_limit {f1
//!   command_name}}]}}. Imola: Pitlane_Main (Main + speed limit "pitlane"),
//!   Pitlane_Entry, Pitlane_Exit - XZ polygons, local to their actor.
//! - spawnpoints_pitlane*.scene: StartingPositionData (actor f104) {f6
//!   types (4 Pit, 5 Box), f2 slot_number}: pit_N is the car's stop spot in
//!   the working lane, box_N the same car parked in its garage. The
//!   container is placed by the track scene (ContainerActorData f121; at
//!   Imola rotated (159.6, -77.5, -160.8)) - without that the spots are
//!   200 m off.
//! - layouts\*.pitlane.aisplinedata: the AI's line through the lane (its
//!   payloads' 13.8 m/s is a hard-coded AI default, 0x415ccccd in the exe).
//!
//! SPEED LIMIT AND PENALTIES: the session's PenaltyInvestigations trigger
//! whose command_name is the zone's ("pitlane"): Speeding {1 required_speed
//! km/h, 2 check_if_higher} with checks {1 data {1 PenaltyType, 2 ms}, 2
//! weight}. content\data\test_drive / hot_lap / race_weekend practice:
//! 60 km/h, weight 62 Warning, 70 TeleportToPit (10 s). instant_race /
//! hot_stint / the race: 50 km/h, 62 Drive-through, 80 Disqualified.
//! timeattack.penaltyinvestigations is empty. ⚠ A check's weight is read
//! here as the peak speed (km/h) that triggers it - a guess that fits both
//! tables (EVOVIEW_PIT_RULES=practice|race|off picks; EVOVIEW_PIT_SPEED=n
//! overrides the limit).
//!
//! STOP TIMES (MfdCommunication, client 20260907: FUN_141629ef0 predicted
//! time, FUN_141628740 repairs, FUN_1416288d0 refuel rate):
//! - refuel = 0.2 s per litre (at most the tank's room) + 3 s, when any;
//! - tyres = 15 s per axle changed (front then rear: 30 s for four);
//! - refuel and tyres run side by side: total = max(refuel, tyres) +
//!   repairs; each rounded to whole seconds;
//! - body repair = the 5 body zones' damage summed / 1275 x 180 s;
//!   suspension = the 4 corners' mean x 120 s; the engine has no time there.
//!   evoview has no damage yet, so both read 0 ("no damage").
//! - the menu is CarPitStopValues: change front / rear tyres, a compound per
//!   axle (one for both when the setup limits say use_single_compound), the
//!   four pressures, fuel to add, repair body / suspension / engine, serve a
//!   stop-and-go. Nothing of the setup: that is the garage's (back to pit).
//!
//! LIMITER: InputAction 138 toggles it (Josh's Moza: button 33); the dash
//! shows `.pitlimiter`, the car's lights category 40, LED group 2. ⚠ Its
//! speed is the lane's limit here: the exe's player limiter speed is not
//! decoded (the AI's lane speed is its data+0x54, km/h).
//!
//! Every car change goes through `apply` (one Change at a time): that is the
//! single place a merge with the setups work hooks in.

use crate::proto::{fields, text, Val};
use crate::trackdrive::{Controls, TrackDrive};
use glam::{Mat4, Vec3, Vec4};
use std::collections::HashMap;

// ------------------------------------------------------------- track data --

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ZoneKind {
    Entry,
    Exit,
    Average,
    Main,
}

#[derive(Clone, Debug)]
pub struct Zone {
    pub kind: ZoneKind,
    /// ZoneData.SpeedLimit command_name ("pitlane"): the zone the limit holds in
    pub limit: Option<String>,
    pub poly: Vec<Vec3>,
    y: f32,
}

impl Zone {
    pub fn contains(&self, p: Vec3) -> bool {
        if (p.y - self.y).abs() > 8.0 || self.poly.len() < 3 {
            return false;
        }
        // even-odd in XZ
        let mut inside = false;
        let n = self.poly.len();
        for i in 0..n {
            let (a, b) = (self.poly[i], self.poly[(i + n - 1) % n]);
            if (a.z > p.z) != (b.z > p.z) && p.x < (b.x - a.x) * (p.z - a.z) / (b.z - a.z) + a.x {
                inside = !inside;
            }
        }
        inside
    }
}

/// A starting position: where the car stands, facing `yaw` (atan2(x, z)).
#[derive(Clone, Copy, Debug)]
pub struct Spot {
    pub slot: i32,
    pub pos: Vec3,
    pub yaw: f32,
}

impl Spot {
    pub fn fwd(&self) -> Vec3 {
        Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos())
    }
}

#[derive(Clone, Debug, Default)]
pub struct PitLane {
    pub zones: Vec<Zone>,
    /// the stop spots in the lane (StartingPositionType_Pit), by slot
    pub pits: Vec<Spot>,
    /// the garages (StartingPositionType_Box), by slot
    pub boxes: Vec<Spot>,
    /// the AI's line through the lane, entry to exit
    pub spline: Vec<Vec3>,
    /// the race grid (spawnpoints_grid*.scene: StartingPositionType_Grid
    /// alone), by slot - race.rs
    pub grid: Vec<Spot>,
}

/// Each actor of a scene: its node matrix and its ActorData (f50) range.
fn scene_actors(d: &[u8], mut f: impl FnMut(Mat4, usize, usize)) {
    fields(d, 0, d.len(), |fno, v| {
        let (2, Some((s, e))) = (fno, v.range()) else { return };
        let mut node = Mat4::IDENTITY;
        let mut body = None;
        fields(d, s, e, |g, w| match (g, w.range()) {
            (4, Some((a, b))) => {
                let (p, r, sc) = crate::load::parse_transform(d, a, b);
                node = crate::load::trs(p, r, sc);
            }
            (50, Some(r)) => body = Some(r),
            _ => {}
        });
        if let Some((a, b)) = body {
            f(node, a, b);
        }
    });
}

fn vec3(d: &[u8], a: usize, b: usize) -> Vec3 {
    let mut v = [0.0f32; 3];
    fields(d, a, b, |f, x| {
        if (1..=3).contains(&f) {
            v[f as usize - 1] = x.f32();
        }
    });
    Vec3::from(v)
}

/// The aisplinedata's points: 1 {1 {1 {x y z}}}.
fn spline_points(d: &[u8]) -> Vec<Vec3> {
    let mut pts = Vec::new();
    fields(d, 0, d.len(), |f, v| {
        let (1, Some((a, b))) = (f, v.range()) else { return };
        fields(d, a, b, |f2, v2| {
            let (1, Some((c, e))) = (f2, v2.range()) else { return };
            fields(d, c, e, |f3, v3| {
                if let (1, Some((g, h))) = (f3, v3.range()) {
                    pts.push(vec3(d, g, h));
                }
            });
        });
    });
    pts
}

/// A track's files by "content\\tracks\\<t>\\..." key: the game's package, or
/// a loose mod track under Saved Games/ACE/mods/content/tracks/<t>.
enum TrackFiles {
    Pkg(crate::pkg::Package),
    Dir(HashMap<String, std::path::PathBuf>),
}

impl TrackFiles {
    fn open(pkg: &std::path::Path, track: &str) -> Option<TrackFiles> {
        let pre = format!("content\\tracks\\{}\\", track.to_lowercase());
        if let Some(p) = crate::pkg::Package::open_filtered(pkg, &[pre.as_str()]).ok().filter(|p| !p.by_path.is_empty()) {
            return Some(TrackFiles::Pkg(p));
        }
        let dir = std::path::Path::new(&crate::envc::var_os("USERPROFILE")?)
            .join("Saved Games").join("ACE").join("mods").join("content").join("tracks").join(track);
        let mut files = HashMap::new();
        let mut stack = vec![dir.clone()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).ok()?.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if let Ok(rel) = p.strip_prefix(&dir) {
                    let rel = rel.to_string_lossy().to_lowercase().replace('/', "\\");
                    files.insert(format!("{pre}{rel}"), p);
                }
            }
        }
        (!files.is_empty()).then_some(TrackFiles::Dir(files))
    }

    fn keys(&self) -> Vec<String> {
        match self {
            TrackFiles::Pkg(p) => p.by_path.keys().cloned().collect(),
            TrackFiles::Dir(f) => f.keys().cloned().collect(),
        }
    }

    fn read(&self, k: &str) -> Vec<u8> {
        match self {
            TrackFiles::Pkg(p) => p.by_path.get(k).map(|e| p.read(*e)).unwrap_or_default(),
            TrackFiles::Dir(f) => f.get(k).and_then(|p| std::fs::read(p).ok()).unwrap_or_default(),
        }
    }
}

impl PitLane {
    /// The track's pit lane: from content.kspkg, or a loose mod track's
    /// folder (None: the track has none).
    pub fn load(pkg: &std::path::Path, track: &str) -> Option<PitLane> {
        let p = TrackFiles::open(pkg, track)?;
        let all_keys = p.keys();
        let norm = |s: &str| s.to_lowercase().replace('/', "\\");
        let mut keys: Vec<String> = all_keys.iter().filter(|k| k.ends_with(".scene")).cloned().collect();
        keys.sort();
        // where the track scene puts each container (one level, like cams.rs)
        let mut placed: HashMap<String, Mat4> = HashMap::new();
        for k in keys.iter().filter(|k| !k.contains("\\containers\\")) {
            let d = p.read(k);
            scene_actors(&d, |node, a, b| {
                fields(&d, a, b, |h, v| {
                    let (121, Some((ca, cb))) = (h, v.range()) else { return };
                    fields(&d, ca, cb, |m, z| {
                        if let (1, Some(r)) = (m, z.range()) {
                            placed.insert(norm(&text(&d, r)), node);
                        }
                    });
                });
            });
        }
        // every pit container's zones and spots, per scene
        let mut zone_sets: Vec<(String, Vec<Zone>)> = Vec::new();
        let mut spot_sets: Vec<(String, Vec<Spot>, Vec<Spot>)> = Vec::new();
        let mut grid_sets: Vec<(String, Vec<Spot>)> = Vec::new();
        for k in keys.iter().filter(|k| k.contains("\\containers\\") && (k.contains("pit") || k.contains("zone") || k.contains("spawn"))) {
            let d = p.read(k);
            let cont = placed.get(&norm(k)).copied().unwrap_or(Mat4::IDENTITY);
            let (mut zones, mut pits, mut boxes, mut grid) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            scene_actors(&d, |node, a, b| {
                let m = cont * node;
                fields(&d, a, b, |h, v| {
                    let Some((x, y)) = v.range() else { return };
                    match h {
                        112 => fields(&d, x, y, |g, w| {
                            let (1000, Some((za, zb))) = (g, w.range()) else { return };
                            let (mut polys, mut kind, mut limit) = (Vec::new(), None, None);
                            fields(&d, za, zb, |zf, zv| match (zf, zv.range()) {
                                (1, Some((ga, gb))) => {
                                    let mut poly = Vec::new();
                                    fields(&d, ga, gb, |pf, pv| {
                                        if let (1, Some((ta, tb))) = (pf, pv.range()) {
                                            let (pp, _, _) = crate::load::parse_transform(&d, ta, tb);
                                            poly.push(m.transform_point3(Vec3::from(pp)));
                                        }
                                    });
                                    polys.push(poly);
                                }
                                (2, Some((ia, ib))) => fields(&d, ia, ib, |inf, iv| match (inf, iv.range()) {
                                    (101, Some((pa, pb))) => {
                                        let mut t = 0u64;
                                        fields(&d, pa, pb, |q, qv| if let (1, Val::Varint(n)) = (q, qv) { t = n });
                                        kind = Some(match t { 1 => ZoneKind::Exit, 2 => ZoneKind::Average, 3 => ZoneKind::Main, _ => ZoneKind::Entry });
                                    }
                                    (111, Some((sa, sb))) => fields(&d, sa, sb, |q, qv| {
                                        if let (1, Some(r)) = (q, qv.range()) {
                                            limit = Some(text(&d, r));
                                        }
                                    }),
                                    _ => {}
                                }),
                                _ => {}
                            });
                            if let Some(kind) = kind {
                                for poly in polys {
                                    let y = poly.iter().map(|p| p.y).sum::<f32>() / poly.len().max(1) as f32;
                                    zones.push(Zone { kind, limit: limit.clone(), poly, y });
                                }
                            }
                        }),
                        104 => {
                            let (mut types, mut slot) = (Vec::new(), 0i32);
                            fields(&d, x, y, |g, w| match (g, w) {
                                (6, Val::Varint(t)) => types.push(t),
                                (6, Val::Bytes(pa, pb)) => types.extend(crate::proto::varints(&d, pa, pb).into_iter().map(|t| t as u64)),
                                (2, Val::Varint(s)) => slot = s as i32,
                                _ => {}
                            });
                            let f = m.transform_vector3(Vec3::Z);
                            let s = Spot { slot, pos: m.w_axis.truncate(), yaw: f.x.atan2(f.z) };
                            // a hotlap spot is Grid + Pit + Box at once: not a pit
                            if types.contains(&3) {
                                if !types.contains(&4) && !types.contains(&5) {
                                    grid.push(s);
                                }
                                return;
                            }
                            if types.contains(&4) {
                                pits.push(s);
                            } else if types.contains(&5) {
                                boxes.push(s);
                            }
                        }
                        _ => {}
                    }
                });
            });
            if !zones.is_empty() {
                zone_sets.push((k.clone(), zones));
            }
            if !pits.is_empty() || !boxes.is_empty() {
                spot_sets.push((k.clone(), pits, boxes));
            }
            if !grid.is_empty() {
                grid_sets.push((k.clone(), grid));
            }
        }
        // several layouts' containers: the running layout's (EVOVIEW_LAYOUT,
        // which an online session sets: spawnpoints_pitlane_<layout>.scene),
        // else the GP one, else the plainest name
        let lay = crate::envc::var("EVOVIEW_LAYOUT").ok().map(|l| l.to_lowercase()).filter(|l| !l.is_empty());
        let rank = |k: &String| (!lay.as_ref().map_or(false, |l| k.contains(l.as_str())), !k.contains("gp"), k.len());
        zone_sets.sort_by_key(|z| rank(&z.0));
        spot_sets.sort_by_key(|s| rank(&s.0));
        grid_sets.sort_by_key(|s| rank(&s.0));
        let mut lane = PitLane::default();
        if let Some((k, z)) = zone_sets.into_iter().next() {
            println!("pits: {} zone(s) from {k}", z.len());
            lane.zones = z;
        }
        if let Some((k, mut pits, mut boxes)) = spot_sets.into_iter().next() {
            pits.sort_by_key(|s| s.slot);
            boxes.sort_by_key(|s| s.slot);
            println!("pits: {} pit spot(s), {} garage(s) from {k}", pits.len(), boxes.len());
            lane.pits = pits;
            lane.boxes = boxes;
        }
        if let Some((k, mut g)) = grid_sets.into_iter().next() {
            g.sort_by_key(|s| s.slot);
            println!("pits: {} grid slot(s) from {k}", g.len());
            lane.grid = g;
        }
        let mut sp: Vec<&String> = all_keys.iter().filter(|k| k.ends_with(".pitlane.aisplinedata")).collect();
        sp.sort_by_key(|k| rank(*k));
        if let Some(k) = sp.first() {
            lane.spline = spline_points(&p.read(k));
        }
        (!lane.zones.is_empty() || !lane.pits.is_empty()).then_some(lane)
    }

    pub fn in_lane(&self, p: Vec3) -> bool {
        self.zones.iter().any(|z| z.contains(p))
    }

    /// Inside the zone the speed limit holds in (the Main zone carrying a
    /// SpeedLimit; any Main zone when none carries one).
    pub fn in_limit(&self, p: Vec3) -> bool {
        let any_limit = self.zones.iter().any(|z| z.limit.is_some());
        self.zones.iter().any(|z| (if any_limit { z.limit.is_some() } else { z.kind == ZoneKind::Main }) && z.contains(p))
    }
}

// ------------------------------------------------------------------ rules --

#[derive(Clone, Copy, Debug)]
pub struct Check {
    /// ⚠ read as the peak speed (km/h) that triggers it
    pub weight: f32,
    /// PenaltyType: 0 Warning, 9 Drive-through, 10 Stop and go, 250
    /// Disqualified, 260 Teleport to pit
    pub kind: u32,
    pub secs: f32,
}

#[derive(Clone, Debug)]
pub struct Rules {
    pub name: String,
    pub limit_kmh: f32,
    pub checks: Vec<Check>,
}

pub fn penalty_name(kind: u32) -> &'static str {
    match kind {
        0 => "warning",
        1 => "reprimand",
        6 => "time penalty",
        9 => "drive-through",
        10 => "stop and go",
        250 => "disqualified",
        260 => "back to the pits",
        _ => "penalty",
    }
}

fn varint_at(d: &[u8], mut i: usize) -> Option<(u64, usize)> {
    let mut r = 0u64;
    for s in 0..10 {
        let b = *d.get(i)?;
        i += 1;
        r |= ((b & 0x7f) as u64) << (7 * s);
        if b < 0x80 {
            return Some((r, i));
        }
    }
    None
}

impl Rules {
    /// The pit lane trigger of a seasondefinition (inside its sessions'
    /// TimeAttack.Specialization Any, so found by its command name).
    fn from_season(d: &[u8], name: &str) -> Option<Rules> {
        let at = d.windows(9).position(|w| w == b"\x12\x07pitlane")?;
        // the enclosing Trigger: a length-delimited field 1 just before
        for st in (at.saturating_sub(6)..at).rev() {
            if d[st] != 0x0a {
                continue;
            }
            let Some((len, q)) = varint_at(d, st + 1) else { continue };
            let end = q + len as usize;
            if q > at || end < at + 9 || end > d.len() {
                continue;
            }
            let mut r = Rules { name: name.into(), limit_kmh: 0.0, checks: Vec::new() };
            fields(d, q, end, |f, v| match (f, v.range()) {
                (38, Some((a, b))) => fields(d, a, b, |g, w| if let (1, Val::Varint(s)) = (g, w) { r.limit_kmh = s as f32 }),
                (100, Some((a, b))) => {
                    let mut c = Check { weight: 0.0, kind: 0, secs: 0.0 };
                    fields(d, a, b, |g, w| match (g, w) {
                        (1, Val::Bytes(x, y)) => fields(d, x, y, |h, u| match (h, u) {
                            (1, Val::Varint(t)) => c.kind = t as u32,
                            (2, Val::Varint(ms)) => c.secs = ms as f32 * 0.001,
                            _ => {}
                        }),
                        (2, Val::Varint(wt)) => c.weight = wt as f32,
                        _ => {}
                    });
                    r.checks.push(c);
                }
                _ => {}
            });
            if r.limit_kmh > 0.0 {
                r.checks.sort_by(|a, b| a.weight.total_cmp(&b.weight));
                return Some(r);
            }
        }
        None
    }

    /// EVOVIEW_PIT_RULES: practice (default: the test drive's), race (the
    /// instant race's), off (the limit, no penalties).
    pub fn load(pkg: &std::path::Path) -> Rules {
        let mode = crate::envc::var("EVOVIEW_PIT_RULES").unwrap_or_default();
        let file = if mode == "race" { "instant_race" } else { "test_drive" };
        let key = format!("content\\data\\{file}.seasondefinition");
        let mut r = crate::pkg::Package::open_filtered(pkg, &["content\\data\\"]).ok()
            .and_then(|p| p.by_path.get(&key).map(|e| p.read(*e)))
            .and_then(|d| Rules::from_season(&d, file))
            .unwrap_or_else(|| Rules {
                name: "default".into(),
                limit_kmh: 60.0,
                checks: vec![Check { weight: 62.0, kind: 0, secs: 0.0 }, Check { weight: 70.0, kind: 260, secs: 10.0 }],
            });
        if mode == "off" {
            r.checks.clear();
            r.name += " (no penalties)";
        }
        if let Some(v) = crate::envc::var("EVOVIEW_PIT_SPEED").ok().and_then(|v| v.parse::<f32>().ok()) {
            r.limit_kmh = v;
        }
        r
    }
}

// --------------------------------------------------------------- the car --

pub struct Compound {
    /// the .compatibletyres entry's f1 (TyreCompound: the M4 GT3's 20
    /// Pirelli GTWC, 21 its wet)
    pub kind: u32,
    /// what the menu calls it, from the .tyre file's folder / name
    pub name: String,
    pub fit: [Option<(crate::carphys::TyreModel, Option<crate::tyretherm::Data>)>; 2],
}

/// A compound's menu name from its front .tyre path.
fn compound_name_of(path: &str) -> String {
    let p = path.to_lowercase();
    let pick = [("inter", "INTER"), ("wet", "WET"), ("rain", "WET"), ("soft", "SOFT"), ("medium", "MEDIUM"), ("hard", "HARD"),
        ("slick", "SLICK"), ("semi", "SEMI SLICK"), ("street", "ROAD"), ("road", "ROAD"), ("vintage", "VINTAGE")];
    pick.iter().find(|(k, _)| p.contains(k)).map_or_else(|| {
        let f = p.rsplit(['\\', '/']).next().unwrap_or("tyre");
        f.trim_end_matches(".tyre").to_uppercase()
    }, |(_, n)| n.to_string())
}

/// Damage the repairs work from: the game's 5 body zones (0..255 each) and
/// 4 suspension corners (0..1). evoview has no damage model yet: zeros.
#[derive(Clone, Copy, Debug, Default)]
pub struct Damage {
    pub body: [f32; 5],
    pub susp: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub min: f32,
    pub max: f32,
    pub step: f32,
}

pub struct PitCar {
    pub max_fuel: f32,
    pub kg_per_l: f32,
    pub fuel: f32,
    pub compounds: Vec<Compound>,
    /// the compound on each axle now
    pub fitted: [usize; 2],
    /// CarSetupDataLimits f100: one compound for both axles
    pub single: bool,
    pub psi: Range,
    /// the setup's pressures (ours: FR FL RR RL), the menu's start
    pub setup_psi: [f32; 4],
    pub arb: [Option<Range>; 2],
    pub damage: Damage,
    /// the setup's aero now: front / rear target ride height mm, front /
    /// rear wing deg (CarSetupData f6 {2, 3, 4, 5}); NaN = not in the setup
    pub aero_now: [f32; 4],
    /// their .carsetuplimits ranges (the same f6 numbers), None = fixed
    pub aero_lim: [Option<Range>; 4],
    /// the car's files and physics, to rebuild the aero (lapsim::aero) when
    /// the garage changes a wing or a ride height
    pub aero_src: Option<(crate::load::Vfs, crate::carphys::Physics)>,
}

fn sub(d: &[u8], a: usize, b: usize, want: u32) -> Option<(usize, usize)> {
    let mut hit = None;
    fields(d, a, b, |f, v| {
        if f == want && hit.is_none() {
            hit = v.range();
        }
    });
    hit
}

fn f32_in(d: &[u8], a: usize, b: usize, want: u32) -> Option<f32> {
    let mut hit = None;
    fields(d, a, b, |f, v| {
        if f == want {
            hit = Some(v.f32());
        }
    });
    hit
}

fn range_of(d: &[u8], (a, b): (usize, usize)) -> Range {
    Range { step: f32_in(d, a, b, 1).unwrap_or(0.1), min: f32_in(d, a, b, 2).unwrap_or(0.0), max: f32_in(d, a, b, 3).unwrap_or(0.0) }
}

impl PitCar {
    pub fn load(vfs: &mut crate::load::Vfs, p: &crate::carphys::Physics) -> PitCar {
        let cd = vfs.read(&p.source);
        // CarData f1 general {5 fuel, 6 maxFuel, 8 kgPerLiter}, f16 setupLimits
        let g = sub(&cd, 0, cd.len(), 1);
        let gf = |f: u32| g.and_then(|(a, b)| f32_in(&cd, a, b, f)).filter(|x| *x > 0.0);
        let max_fuel = gf(6).unwrap_or(60.0);
        let kg_per_l = gf(8).unwrap_or(0.75);
        let mut lim_path = None;
        fields(&cd, 0, cd.len(), |f, v| if let (16, Some(r)) = (f, v.range()) { lim_path = Some(text(&cd, r)) });
        // the setup: f7 fuel_strategy {1 fuel}, f4 alignments x4 {1 pressure}
        // (FL FR RL RR)
        let (mut fuel, mut psi_fl) = (None, Vec::new());
        if !p.setup_source.is_empty() {
            let sd = vfs.read(&p.setup_source);
            fuel = sub(&sd, 0, sd.len(), 7).and_then(|(a, b)| f32_in(&sd, a, b, 1));
            fields(&sd, 0, sd.len(), |f, v| {
                if let (4, Some((a, b))) = (f, v.range()) {
                    psi_fl.push(f32_in(&sd, a, b, 1).unwrap_or(0.0));
                }
            });
        }
        // the tank as carphys loaded it (the player's setup's fuel when one
        // is on, already in the rig's mass): pit changes add to that
        let fuel = Some(p.fuel_l).filter(|l| *l > 0.0).or(fuel).or(gf(5)).unwrap_or(max_fuel * 0.5).min(max_fuel);
        let kg_per_l = if p.kg_per_l > 0.0 { p.kg_per_l } else { kg_per_l };
        let mut setup_psi = if psi_fl.len() >= 4 { [psi_fl[1], psi_fl[0], psi_fl[3], psi_fl[2]] } else { [0.0; 4] };
        // the player's setup's pressures (carphys, per axle) over the stock's
        if !p.user_setup.is_empty() {
            for k in 0..4 {
                if let Some(v) = p.pressure[k / 2] {
                    setup_psi[k] = v;
                }
            }
        }
        let aero_now = [p.ride_h[0].map(|m| m * 1000.0), p.ride_h[1].map(|m| m * 1000.0), p.wing[0], p.wing[1]]
            .map(|v| v.unwrap_or(f32::NAN));
        let mut aero_lim = [None; 4];
        let (mut psi, mut arb, mut single) = (Range { min: 15.0, max: 40.0, step: 0.1 }, [None, None], false);
        if let Some(k) = lim_path.and_then(|l| vfs.find(&l, None)) {
            let ld = vfs.read(&k);
            if let Some(r) = sub(&ld, 0, ld.len(), 4).and_then(|(a, b)| sub(&ld, a, b, 1)) {
                psi = range_of(&ld, r);
            }
            if let Some((a, b)) = sub(&ld, 0, ld.len(), 1) {
                let mut i = 0;
                fields(&ld, a, b, |f, v| {
                    if let (1, Some(r)) = (f, v.range()) {
                        if i < 2 {
                            arb[i] = Some(range_of(&ld, r)).filter(|r| r.max > r.min);
                        }
                        i += 1;
                    }
                });
            }
            fields(&ld, 0, ld.len(), |f, v| if let (100, Val::Varint(x)) = (f, v) { single = x != 0 });
            // f6 aero {2 / 3 front / rear target height mm, 4 / 5 wing deg}
            if let Some((a, b)) = sub(&ld, 0, ld.len(), 6) {
                for (i, f) in [2u32, 3, 4, 5].into_iter().enumerate() {
                    aero_lim[i] = sub(&ld, a, b, f).map(|r| range_of(&ld, r)).filter(|r| r.max > r.min);
                }
            }
        }
        let compounds: Vec<Compound> = crate::carphys::compatible_tyres(vfs, &p.source).into_iter().map(|(kind, paths)| Compound {
            kind,
            name: compound_name_of(&paths[0]),
            fit: [0, 1].map(|ax| crate::carphys::read_tyre(vfs, &paths[ax])),
        }).collect();
        // which one the car has on: the same grip model
        let fitted = [0, 1].map(|ax| {
            compounds.iter().position(|c| match (&c.fit[ax], p.tyres[ax]) {
                (Some((m, _)), Some(t)) => m.dy0 == t.dy0 && m.fz0 == t.fz0 && m.ls_y == t.ls_y,
                _ => false,
            }).unwrap_or(0)
        });
        let names: Vec<&str> = compounds.iter().map(|c| c.name.as_str()).collect();
        if crate::envc::var_os("EVOVIEW_TYRE_LOG").is_some() {
            for c in &compounds {
                println!("TYRE compound {} kind {}: {:?}", c.name, c.kind, c.fit.iter().map(|f| f.as_ref().map(|(m, _)| (m.dy0, m.dx0, m.fz0, m.ls_y))).collect::<Vec<_>>());
            }
        }
        println!("pits: fuel {fuel:.0} / {max_fuel:.0} L ({kg_per_l} kg/L), tyres {names:?} (fitted {:?}{}), pressures {:.1}-{:.1} psi",
            fitted, if single { ", one compound" } else { "" }, psi.min, psi.max);
        PitCar { max_fuel, kg_per_l, fuel, compounds, fitted, single, psi, setup_psi, arb, damage: Damage::default(),
                 aero_now, aero_lim, aero_src: None }
    }

    /// The game's repair times, s: body, suspension (FUN_141628740).
    pub fn repair_secs(&self) -> (f32, f32) {
        ((self.damage.body.iter().sum::<f32>() / 1275.0 * 180.0).round(),
         (self.damage.susp.iter().sum::<f32>() * 0.25 * 120.0).round())
    }
}

// ------------------------------------------------------- apply a change --

/// One change to the car, from a pit stop or the garage.
#[derive(Clone, Debug)]
pub enum Change {
    /// fresh tyres of compound `compound` on `axle` (0 front, 1 rear) at
    /// `psi` [right, left]
    Tyres { axle: usize, compound: usize, psi: [f32; 2] },
    /// litres into (or out of) the tank
    Fuel(f32),
    /// anti-roll bar rate (the setup's units)
    Arb { axle: usize, rate: f32 },
    /// aero setting `which` (0 / 1 front / rear ride height mm, 2 / 3 front
    /// / rear wing deg): the aero is rebuilt from the car's data
    Aero { which: usize, value: f32 },
    RepairBody,
    RepairSuspension,
}

/// THE place a pit stop or the garage changes the car. A loaded setup
/// (`--setup`, carphys.rs) sets fuel, pressures, wings and ride height at
/// load; here the same quantities change through the same setters
/// (Tyre::set_pressure, the rig's mass, lapsim::aero). Returns what it did.
pub fn apply(td: &mut TrackDrive, car: &mut PitCar, ch: &Change) -> String {
    let rig = td.rig.as_mut().map(|(rc, _)| rc);
    match ch {
        Change::Tyres { axle, compound, psi } => {
            let Some(c) = car.compounds.get(*compound) else { return "no such compound".into() };
            let (fit, name) = (c.fit[*axle].clone(), c.name.clone());
            car.fitted[*axle] = *compound;
            match (rig, fit) {
                (Some(rc), Some((m, data))) => {
                    rc.fit_tyres(*axle, m.sctm, data, *psi);
                    format!("{} tyres: new {name} at {:.1} / {:.1} psi", ["front", "rear"][*axle], psi[1], psi[0])
                }
                _ => format!("{} tyres: {name} (the simple car model has no tyre state)", ["front", "rear"][*axle]),
            }
        }
        Change::Fuel(l) => {
            let l = l.clamp(-car.fuel, car.max_fuel - car.fuel);
            car.fuel += l;
            if let Some(rc) = rig {
                rc.add_mass(l * car.kg_per_l);
                format!("fuel {:+.1} L -> {:.1} L, body {:.1} kg", l, car.fuel, rc.body_mass())
            } else {
                format!("fuel {:+.1} L -> {:.1} L", l, car.fuel)
            }
        }
        Change::Arb { axle, rate } => match rig {
            Some(rc) => {
                rc.set_arb(*axle, *rate);
                format!("{} anti-roll bar {:.0}", ["front", "rear"][*axle], rate)
            }
            None => "anti-roll bars: rigid-body car only".into(),
        },
        Change::Aero { which, value } => {
            let w = (*which).min(3);
            car.aero_now[w] = *value;
            let what = ["front ride height", "rear ride height", "front wing", "rear wing"][w];
            let unit = if w < 2 { "mm" } else { "deg" };
            let Some((vfs, ph)) = car.aero_src.as_mut() else { return format!("{what} {value:.1} {unit} (no aero data)") };
            if w < 2 {
                ph.ride_h[w] = Some(value / 1000.0);
            } else {
                ph.wing[w - 2] = Some(*value);
            }
            let k = crate::lapsim::aero(vfs, ph);
            td.set_aero(k);
            format!("{what} {value:.1} {unit}")
        }
        Change::RepairBody => {
            car.damage.body = [0.0; 5];
            "bodywork repaired".into()
        }
        Change::RepairSuspension => {
            car.damage.susp = [0.0; 4];
            "suspension repaired".into()
        }
    }
}

// ----------------------------------------------------------- the stop --

/// What the menu asks for (CarPitStopValues, plus the garage's bars).
#[derive(Clone, Debug)]
pub struct Plan {
    /// change front / rear tyres
    pub tyres: [bool; 2],
    pub compound: [usize; 2],
    /// new tyres' pressures, ours: FR FL RR RL
    pub psi: [f32; 4],
    pub fuel: f32,
    pub body: bool,
    pub susp: bool,
    pub arb: [f32; 2],
    /// garage: PitCar::aero_now's settings
    pub aero: [f32; 4],
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Stage {
    Driving,
    /// stopped in the box, the menu up
    Stopped,
    /// the crew at work
    Service,
    /// done: free to go (until the car leaves the box)
    Released,
    /// in the garage (Home): the setup menu
    Garage,
    /// a penalty holds the car
    Held,
}

struct Service {
    plan: Plan,
    t: f32,
    litres: f32,
    refuel: f32,
    tyres: f32,
    repair: f32,
    hold: f32,
    total: f32,
    fuel_in: f32,
    done: [bool; 3],
}

/// The pit stop times for a plan, s: (refuel, tyres, repairs, total).
pub fn stop_times(car: &PitCar, plan: &Plan) -> (f32, f32, f32, f32) {
    let room = (car.max_fuel - car.fuel).max(0.0);
    let litres = if plan.fuel > room { room.ceil() } else { plan.fuel };
    let refuel = if plan.fuel > 0.0 { (0.2 * litres.min(car.max_fuel) + 3.0).round() } else { 0.0 };
    let tyres = 15.0 * (plan.tyres[0] as u32 + plan.tyres[1] as u32) as f32;
    let (b, s) = car.repair_secs();
    let repair = if plan.body { b } else { 0.0 } + if plan.susp { s } else { 0.0 };
    (refuel, tyres, repair, refuel.max(tyres) + repair)
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Row {
    Tyres,
    /// 0 front, 1 rear, 2 both
    Compound(usize),
    /// ours: 0 FR 1 FL 2 RR 3 RL
    Psi(usize),
    Fuel,
    Body,
    Susp,
    Arb(usize),
    /// PitCar::aero_now index
    Aero(usize),
    Go,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Key {
    Limiter,
    Menu,
    Up,
    Down,
    Left(bool),
    Right(bool),
    Ok,
    Garage,
}

pub struct Pits {
    pub lane: Option<PitLane>,
    pub rules: Rules,
    pub car: PitCar,
    pub limiter: bool,
    /// the game's automatic_pit_limiter assist (EVOVIEW_PIT_AUTOLIMITER=1)
    auto_limiter: bool,
    pub in_lane: bool,
    pub in_limit: bool,
    pub speeding: bool,
    /// this visit's top speed in the limit zone, and the checks it fired
    peak: f32,
    fired: usize,
    /// index into lane.pits / lane.boxes
    pub box_i: usize,
    pub stage: Stage,
    held_left: f32,
    pub menu: bool,
    row: usize,
    pub plan: Plan,
    service: Option<Service>,
    /// a drive-through / stop-and-go to serve (race rules)
    pub drive_through: bool,
    pub stop_go: f32,
    through_clean: bool,
    msgs: Vec<(String, f32, [f32; 4])>,
    pub time: f32,
    /// the right edge of the pit-lane sign (top right; the lap panel moves
    /// it left of itself)
    pub sign_right: f32,
}

const AMBER: [f32; 4] = [1.0, 0.72, 0.1, 0.95];
const RED: [f32; 4] = [1.0, 0.22, 0.18, 0.95];
const GREEN: [f32; 4] = [0.3, 0.95, 0.4, 0.95];
const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 0.95];
const DIM: [f32; 4] = [1.0, 1.0, 1.0, 0.55];

impl Pits {
    pub fn new(lane: Option<PitLane>, rules: Rules, car: PitCar, rig_arb: [f32; 2]) -> Pits {
        let want = crate::envc::var("EVOVIEW_PIT_BOX").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(0);
        let box_i = lane.as_ref().and_then(|l| l.pits.iter().position(|s| s.slot == want)).unwrap_or(0);
        let psi = [0, 1, 2, 3].map(|k| {
            let s = car.setup_psi[k];
            let ideal = car.compounds.get(car.fitted[k / 2]).and_then(|c| c.fit[k / 2].as_ref())
                .and_then(|f| f.1.as_ref()).map_or(0.0, |d| d.ideal_p);
            if s > 0.0 { s } else { ideal }
        });
        println!("pits: lane limit {:.0} km/h ({} rules: {}), box {}",
            rules.limit_kmh, rules.name,
            rules.checks.iter().map(|c| format!("{:.0} km/h -> {}", c.weight, penalty_name(c.kind))).collect::<Vec<_>>().join(", "),
            lane.as_ref().and_then(|l| l.pits.get(box_i)).map_or(-1, |s| s.slot + 1));
        let plan = Plan { tyres: [true, true], compound: car.fitted, psi, fuel: 0.0, body: false, susp: false, arb: rig_arb, aero: car.aero_now };
        Pits {
            lane, rules, car, limiter: false,
            auto_limiter: crate::envc::var_os("EVOVIEW_PIT_AUTOLIMITER").is_some(),
            in_lane: false, in_limit: false, speeding: false, peak: 0.0, fired: 0,
            box_i, stage: Stage::Driving, held_left: 0.0, menu: false, row: 0, plan, service: None,
            drive_through: false, stop_go: 0.0, through_clean: false, msgs: Vec::new(), time: 0.0,
            sign_right: 0.98,
        }
    }

    fn say(&mut self, s: impl Into<String>, secs: f32, c: [f32; 4]) {
        let s = s.into();
        println!("PIT t {:6.1}  {s}", self.time);
        self.msgs.retain(|m| m.0 != s);
        self.msgs.push((s, secs, c));
    }

    pub fn spot(&self) -> Option<Spot> {
        self.lane.as_ref()?.pits.get(self.box_i).copied()
    }

    fn garage_spot(&self) -> Option<Spot> {
        let l = self.lane.as_ref()?;
        let slot = l.pits.get(self.box_i)?.slot;
        l.boxes.iter().find(|b| b.slot == slot).copied()
    }

    /// The car's place against the box: (along, across) m, heading error rad.
    fn box_error(&self, td: &TrackDrive) -> Option<(f32, f32, f32)> {
        let s = self.spot()?;
        let d = td.pos - s.pos;
        let f = s.fwd();
        let r = Vec3::new(f.z, 0.0, -f.x);
        let dy = ((td.yaw - s.yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)) - std::f32::consts::PI;
        Some((d.dot(f), d.dot(r), dy))
    }

    /// Whether the menu has the keyboard's arrows (the car's arrows are off).
    pub fn menu_takes_arrows(&self) -> bool {
        self.menu
    }

    /// The car may not move: stopped in the box, served, in the garage, held.
    pub fn holds(&self) -> bool {
        matches!(self.stage, Stage::Stopped | Stage::Service | Stage::Garage | Stage::Held)
    }

    /// The driver's inputs through the pit systems: the crew holds the car;
    /// the limiter caps the throttle at the lane's speed (a governor a hair
    /// under it, like a real one).
    pub fn filter(&self, c: &mut Controls, td: &TrackDrive) {
        if self.holds() {
            // the handbrake, not the pedal: held at a standstill the
            // keyboard automatic takes the brake pedal for reverse and then
            // drives on it (rigcar: reversing swaps the pedals)
            c.throttle = 0.0;
            c.brake = 0.0;
            c.handbrake = true;
            c.handbrake_v = Some(1.0);
            return;
        }
        if self.limiter {
            let target = (self.rules.limit_kmh - 0.8) / 3.6;
            let cap = ((target - td.vx.abs()) * 1.2 + 0.15).clamp(0.0, 1.0);
            c.throttle = c.throttle.min(cap);
        }
    }

    fn rows(&self) -> Vec<Row> {
        let mut r = vec![Row::Tyres];
        if self.car.compounds.len() > 1 {
            if self.car.single {
                r.push(Row::Compound(2));
            } else {
                r.extend([Row::Compound(0), Row::Compound(1)]);
            }
        }
        r.extend([Row::Psi(1), Row::Psi(0), Row::Psi(3), Row::Psi(2), Row::Fuel, Row::Body, Row::Susp]);
        if self.stage == Stage::Garage {
            for ax in 0..2 {
                if self.car.arb[ax].is_some() {
                    r.push(Row::Arb(ax));
                }
            }
            if self.car.aero_src.is_some() {
                for i in 0..4 {
                    if self.car.aero_lim[i].is_some() && self.car.aero_now[i].is_finite() {
                        r.push(Row::Aero(i));
                    }
                }
            }
        }
        r.push(Row::Go);
        r
    }

    fn adjust(&mut self, row: Row, dir: f32, big: bool) {
        let p = &mut self.plan;
        let n = self.car.compounds.len().max(1);
        match row {
            Row::Tyres => {
                // all / front / rear / none
                let i = match p.tyres { [true, true] => 0, [true, false] => 1, [false, true] => 2, _ => 3 };
                let i = (i as i32 + dir as i32).rem_euclid(4);
                p.tyres = [[true, true], [true, false], [false, true], [false, false]][i as usize];
            }
            Row::Compound(ax) => {
                for a in 0..2 {
                    if ax == 2 || ax == a {
                        p.compound[a] = (p.compound[a] as i32 + dir as i32).rem_euclid(n as i32) as usize;
                        // a new compound starts at its own ideal pressure
                        if let Some(d) = self.car.compounds[p.compound[a]].fit[a].as_ref().and_then(|f| f.1.as_ref()) {
                            if d.ideal_p > 0.0 {
                                p.psi[a * 2] = d.ideal_p;
                                p.psi[a * 2 + 1] = d.ideal_p;
                            }
                        }
                    }
                }
            }
            Row::Psi(k) => {
                let r = self.car.psi;
                let step = if big { 1.0 } else { r.step.max(0.1) };
                p.psi[k] = ((p.psi[k] + dir * step) * 10.0).round() / 10.0;
                p.psi[k] = p.psi[k].clamp(r.min.min(r.max), r.max.max(r.min));
            }
            Row::Fuel => {
                let room = (self.car.max_fuel - self.car.fuel).max(0.0).ceil();
                p.fuel = (p.fuel + dir * if big { 10.0 } else { 1.0 }).clamp(0.0, room);
            }
            Row::Body => p.body = !p.body,
            Row::Susp => p.susp = !p.susp,
            Row::Arb(ax) => {
                if let Some(r) = self.car.arb[ax] {
                    let step = r.step.max((r.max - r.min) / 100.0) * if big { 10.0 } else { 1.0 };
                    p.arb[ax] = (p.arb[ax] + dir * step).clamp(r.min, r.max);
                }
            }
            Row::Aero(i) => {
                if let Some(r) = self.car.aero_lim[i] {
                    let step = r.step.max(0.1) * if big { 5.0 } else { 1.0 };
                    p.aero[i] = (p.aero[i] + dir * step).clamp(r.min, r.max);
                }
            }
            Row::Go => {}
        }
    }

    pub fn key(&mut self, k: Key, td: &mut TrackDrive) {
        match k {
            Key::Limiter => {
                self.limiter = !self.limiter;
                self.say(if self.limiter { "pit limiter on" } else { "pit limiter off" }, 1.5, AMBER);
            }
            Key::Menu => {
                // the garage's and the box's menu stay up while the car is there
                if !matches!(self.stage, Stage::Stopped | Stage::Garage) {
                    self.menu = !self.menu;
                }
            }
            Key::Garage => self.to_garage(td),
            _ if !self.menu => {}
            Key::Up | Key::Down => {
                let n = self.rows().len();
                self.row = (self.row as i32 + if k == Key::Up { -1 } else { 1 }).rem_euclid(n as i32) as usize;
            }
            Key::Left(big) | Key::Right(big) => {
                let rows = self.rows();
                let row = rows[self.row.min(rows.len() - 1)];
                self.adjust(row, if matches!(k, Key::Left(_)) { -1.0 } else { 1.0 }, big);
            }
            Key::Ok => match self.stage {
                Stage::Stopped => self.start_service(),
                Stage::Garage => self.garage_go(td),
                _ => self.menu = false,
            },
        }
    }

    /// The wheel's pit buttons, pressed this frame (wheel::PIT_ROLES order:
    /// limiter, menu, up, down, left, right, ok, back to the pits). Stopped
    /// in the box or the garage, the limiter button is the menu's OK.
    pub fn wheel(&mut self, pressed: [bool; 8], td: &mut TrackDrive) {
        let stopped = matches!(self.stage, Stage::Stopped | Stage::Garage);
        let keys = [if stopped { Key::Ok } else { Key::Limiter }, Key::Menu, Key::Up, Key::Down, Key::Left(false), Key::Right(false), Key::Ok, Key::Garage];
        for (i, k) in keys.into_iter().enumerate() {
            if pressed[i] {
                self.key(k, td);
            }
        }
    }

    fn start_service(&mut self) {
        let plan = self.plan.clone();
        let (refuel, tyres, repair, total) = stop_times(&self.car, &plan);
        let room = (self.car.max_fuel - self.car.fuel).max(0.0);
        let litres = plan.fuel.min(room.ceil()).min(room);
        let hold = self.stop_go;
        self.stop_go = 0.0;
        self.say(format!("pit stop: {:.0} s (refuel {refuel:.0} s for {litres:.0} L, tyres {tyres:.0} s, repairs {repair:.0} s{})",
            total + hold, if hold > 0.0 { format!(", stop and go {hold:.0} s") } else { String::new() }), 3.0, WHITE);
        self.service = Some(Service { plan, t: 0.0, litres, refuel, tyres, repair, hold, total: total + hold, fuel_in: 0.0, done: [false; 3] });
        self.stage = Stage::Service;
        self.menu = false;
    }

    fn to_garage(&mut self, td: &mut TrackDrive) {
        let Some(s) = self.garage_spot().or(self.spot()) else {
            self.say("no garage on this track", 2.0, RED);
            return;
        };
        if !td.place_from(s.pos, s.yaw, 1.5) {
            self.say("garage: no floor to stand on", 2.0, RED);
            return;
        }
        self.stage = Stage::Garage;
        self.service = None;
        self.menu = true;
        self.row = 0;
        if let Some((rc, _)) = td.rig.as_ref() {
            self.plan.arb = rc.arb();
        }
        self.plan.aero = self.car.aero_now;
        self.say("back in the pits: changes go in at once", 2.5, WHITE);
    }

    /// The garage's changes go in at once (no crew timing there).
    fn garage_go(&mut self, td: &mut TrackDrive) {
        let p = self.plan.clone();
        let mut done = Vec::new();
        if p.fuel != 0.0 {
            done.push(apply(td, &mut self.car, &Change::Fuel(p.fuel)));
        }
        for ax in 0..2 {
            if p.tyres[ax] {
                done.push(apply(td, &mut self.car, &Change::Tyres { axle: ax, compound: p.compound[ax], psi: [p.psi[ax * 2], p.psi[ax * 2 + 1]] }));
            }
            if self.car.arb[ax].is_some() {
                done.push(apply(td, &mut self.car, &Change::Arb { axle: ax, rate: p.arb[ax] }));
            }
        }
        for i in 0..4 {
            if self.car.aero_lim[i].is_some() && p.aero[i].is_finite() && p.aero[i] != self.car.aero_now[i] {
                done.push(apply(td, &mut self.car, &Change::Aero { which: i, value: p.aero[i] }));
            }
        }
        if p.body {
            done.push(apply(td, &mut self.car, &Change::RepairBody));
        }
        if p.susp {
            done.push(apply(td, &mut self.car, &Change::RepairSuspension));
        }
        for d in done {
            println!("PIT t {:6.1}  garage: {d}", self.time);
        }
        self.plan.fuel = 0.0;
        self.stage = Stage::Released;
        self.menu = false;
        self.say("setup applied: drive out", 2.5, GREEN);
    }

    /// Each frame, after the physics step.
    pub fn update(&mut self, td: &mut TrackDrive, dt: f32) {
        self.time += dt;
        for m in self.msgs.iter_mut() {
            m.1 -= dt;
        }
        self.msgs.retain(|m| m.1 > 0.0);
        let p = td.pos;
        let kmh = td.speed_kmh().abs();
        let (in_lane, in_limit) = self.lane.as_ref().map_or((false, false), |l| (l.in_lane(p), l.in_limit(p)));
        if in_lane != self.in_lane {
            self.in_lane = in_lane;
            if in_lane && self.time < 1.0 {
                self.through_clean = true;
            } else if in_lane {
                self.say(format!("pit lane: entered at {kmh:.0} km/h"), 2.0, WHITE);
                self.through_clean = true;
            } else {
                self.say(format!("pit lane: left at {kmh:.0} km/h (peak {:.1} km/h in the limit zone)", self.peak), 2.0, WHITE);
                if self.drive_through && self.through_clean {
                    self.drive_through = false;
                    self.say("drive-through served", 3.0, GREEN);
                }
                self.peak = 0.0;
                self.fired = 0;
            }
        }
        if in_limit != self.in_limit {
            self.in_limit = in_limit;
            if self.auto_limiter {
                self.limiter = in_limit;
            }
        }
        self.speeding = in_limit && kmh > self.rules.limit_kmh + 0.05;
        if in_limit {
            self.peak = self.peak.max(kmh);
            if self.speeding {
                self.through_clean = false;
            }
            // the worst check the peak has reached, once (a jump past two
            // fires only the heavier)
            let hit = (self.fired..self.rules.checks.len()).rev().find(|&i| self.peak >= self.rules.checks[i].weight);
            if let Some(i) = hit {
                self.fired = i + 1;
                let c = self.rules.checks[i];
                self.penalty(c, td);
            }
        }
        // the box
        let err = self.box_error(td);
        match self.stage {
            Stage::Driving => {
                if let Some((along, across, dyaw)) = err {
                    let still = td.vx.abs() < 0.15;
                    if still && in_lane && along.abs() < 1.2 && across.abs() < 1.2 && dyaw.abs() < 0.45 {
                        self.stage = Stage::Stopped;
                        self.menu = true;
                        self.row = self.rows().len() - 1;
                        self.through_clean = false;
                        self.say(format!("in the box ({along:+.2} m, {across:+.2} m): Enter starts the stop"), 3.0, GREEN);
                    } else if still && in_lane && along.abs() < 8.0 && across.abs() < 4.0 && dyaw.abs() < 0.6 {
                        if !self.msgs.iter().any(|m| m.0.starts_with("out of the box")) {
                            self.say(format!("out of the box: {along:+.1} m along, {across:+.1} m across"), 1.5, AMBER);
                        }
                    }
                }
            }
            Stage::Service => self.service_step(td, dt),
            Stage::Released => {
                let away = err.map_or(true, |(a, c, _)| a.abs() > 6.0 || c.abs() > 4.0);
                let garage_away = self.garage_spot().map_or(true, |g| (td.pos - g.pos).length() > 6.0);
                if away && garage_away {
                    self.stage = Stage::Driving;
                }
            }
            Stage::Held => {
                self.held_left -= dt;
                if self.held_left <= 0.0 {
                    self.stage = Stage::Released;
                    self.say("penalty served: go", 2.0, GREEN);
                }
            }
            Stage::Stopped | Stage::Garage => {}
        }
    }

    fn penalty(&mut self, c: Check, td: &mut TrackDrive) {
        let peak = self.peak;
        match c.kind {
            9 => self.drive_through = true,
            10 => self.stop_go = c.secs.max(10.0),
            260 => {
                // teleported to the box (the garage) and held there
                if let Some(s) = self.garage_spot().or(self.spot()) {
                    if td.place_from(s.pos, s.yaw, 1.5) {
                        self.stage = Stage::Held;
                        self.held_left = c.secs.max(1.0);
                        self.menu = false;
                    }
                }
            }
            _ => {}
        }
        self.say(format!("pit lane speeding at {peak:.0} km/h: {}", penalty_name(c.kind)), 4.0, RED);
    }

    fn service_step(&mut self, td: &mut TrackDrive, dt: f32) {
        let Some(mut s) = self.service.take() else { return };
        s.t += dt;
        // fuel flows after the 3 s hook-up, 5 L/s; all in by the refuel time
        if s.litres > 0.0 {
            let want = if s.t >= s.refuel { s.litres } else { ((s.t - 3.0) / 0.2).clamp(0.0, s.litres) };
            if want > s.fuel_in {
                let m = apply(td, &mut self.car, &Change::Fuel(want - s.fuel_in));
                s.fuel_in = want;
                if want >= s.litres {
                    println!("PIT t {:6.1}  refuelled {:.1} L: {m}", self.time, s.litres);
                }
            }
        }
        // tyres: front at 15 s, rear 15 s after
        let mut at = 0.0;
        for ax in 0..2 {
            if !s.plan.tyres[ax] {
                continue;
            }
            at += 15.0;
            if !s.done[ax] && s.t >= at {
                s.done[ax] = true;
                let m = apply(td, &mut self.car, &Change::Tyres { axle: ax, compound: s.plan.compound[ax], psi: [s.plan.psi[ax * 2], s.plan.psi[ax * 2 + 1]] });
                self.say(m, 2.0, WHITE);
            }
        }
        if !s.done[2] && s.t >= s.refuel.max(s.tyres) + s.repair {
            s.done[2] = true;
            if s.plan.body {
                let m = apply(td, &mut self.car, &Change::RepairBody);
                self.say(m, 2.0, WHITE);
            }
            if s.plan.susp {
                let m = apply(td, &mut self.car, &Change::RepairSuspension);
                self.say(m, 2.0, WHITE);
            }
        }
        if s.t >= s.total {
            self.stage = Stage::Released;
            self.plan.fuel = 0.0;
            self.say(format!("go, go, go ({:.1} s stationary)", s.t), 2.5, GREEN);
            return;
        }
        self.service = Some(s);
    }

    // ------------------------------------------------------------ the HUD --
    // ACECM's look (hud::theme): graphite panels with a hairline edge, Segoe
    // UI in sentence case, the red accent for the selection and the main
    // action only.

    /// The pit HUD: limiter, lane sign, speeding, the box marker (projected
    /// with the frame's `vp`), the stop countdown and the menu.
    pub fn draw(&self, hud: &mut crate::hud::Hud, td: &TrackDrive, vp: Mat4, aspect: f32) {
        use crate::hud::{theme as t, Weight};
        let fs = 0.0135;
        let kmh = td.speed_kmh().abs();
        let mut y = 0.06;
        // a centred chip: text on a panel with a coloured edge on the left
        let chip = |hud: &mut crate::hud::Hud, y: f32, s: &str, size: f32, fill: [f32; 4], fg: [f32; 4], bar: Option<[f32; 4]>| -> f32 {
            let w = hud.label_w(s, size, Weight::Semibold);
            let (padx, pady) = (size * 0.9 / aspect, size * 0.45);
            let (bw, bh) = (w + padx * 2.0, hud.line_h(size) + pady * 2.0);
            let x = 0.5 - bw * 0.5;
            hud.panel(x, y, bw, bh, 6.0, fill, t::line(fill[3]));
            if let Some(b) = bar {
                hud.round_rect(x, y, hud.px_x(3.0), bh, 1.5, b);
            }
            hud.label(s, x + padx, y + pady, size, Weight::Semibold, fg);
            bh
        };
        if self.limiter {
            let s = format!("Pit limiter  {}", crate::cluster::speed(kmh.min(999.0)));
            y += chip(hud, y, &s, fs, t::gold(0.95), t::hex(0x1a1e25, 1.0), None) + 0.008;
        }
        // the lane's speed sign, top right: a real round sign
        if self.in_lane {
            let r = 0.048;
            let rw = r / aspect;
            let (cx, cy) = (self.sign_right - rw, 0.03 + r);
            hud.round_rect(cx - rw, cy - r, rw * 2.0, r * 2.0, 999.0, t::hex(0xd9342b, 0.97));
            hud.round_rect(cx - rw * 0.78, cy - r * 0.78, rw * 1.56, r * 1.56, 999.0, t::hex(0xffffff, 0.97));
            let s = format!("{:.0}", self.rules.limit_kmh);
            let size = r * 0.95;
            let w = hud.label_w(&s, size, Weight::Semibold);
            hud.label(&s, cx - w * 0.5, cy - hud.line_h(size) * 0.5, size, Weight::Semibold, t::hex(0x111111, 1.0));
            let w = hud.label_w("Pit lane", fs * 0.85, Weight::Semibold);
            hud.label("Pit lane", cx - w * 0.5, cy + r + 0.006, fs * 0.85, Weight::Semibold, t::fg(0.95));
            if self.speeding && (self.time * 4.0).fract() < 0.6 {
                let s = format!("Speeding  {}", crate::cluster::speed(kmh));
                y += chip(hud, y, &s, fs, t::red(0.95), t::hex(0xffffff, 1.0), None) + 0.008;
            }
        }
        if self.drive_through {
            y += chip(hud, y, "Drive-through to serve", fs, t::panel(0.92), t::fg(1.0), Some(t::gold(1.0))) + 0.008;
        }
        // the box: a marker on the ground and its distance
        if let (Some(s), true) = (self.spot(), self.in_lane && matches!(self.stage, Stage::Driving)) {
            let d = (td.pos - s.pos).length();
            if d < 250.0 {
                let ok = self.box_error(td).map_or(false, |(a, c, _)| a.abs() < 1.2 && c.abs() < 1.2);
                box_marker(hud, vp, s, aspect, if ok { t::ok(0.95) } else { t::gold(0.95) });
                let txt = format!("Box {}  {:.0} m", s.slot + 1, d);
                y += chip(hud, y, &txt, fs, t::panel(0.92), t::fg(1.0), Some(if ok { t::ok(1.0) } else { t::gold(1.0) })) + 0.008;
            }
        }
        match self.stage {
            Stage::Service => {
                if let Some(sv) = self.service.as_ref() {
                    let left = (sv.total - sv.t).max(0.0);
                    let w = 0.30 / aspect;
                    let x = 0.5 - w * 0.5;
                    let pad = 0.014;
                    let lh = hud.line_h(fs);
                    let mut rows: Vec<(String, f32)> = Vec::new();
                    if sv.litres > 0.0 {
                        rows.push((format!("Fuel  {:.0} L", sv.litres), sv.fuel_in / sv.litres));
                    }
                    let mut at = 0.0;
                    for ax in 0..2 {
                        if sv.plan.tyres[ax] {
                            rows.push((["Front tyres", "Rear tyres"][ax].into(), ((sv.t - at) / 15.0).clamp(0.0, 1.0)));
                            at += 15.0;
                        }
                    }
                    if sv.repair > 0.0 {
                        rows.push(("Repairs".into(), ((sv.t - sv.refuel.max(sv.tyres)) / sv.repair).clamp(0.0, 1.0)));
                    }
                    if sv.hold > 0.0 {
                        rows.push(("Stop and go".into(), (sv.t / sv.total).clamp(0.0, 1.0)));
                    }
                    let big = fs * 1.9;
                    let h = pad * 2.0 + hud.line_h(big) + 0.006 + rows.len() as f32 * (lh + 0.004);
                    hud.panel(x, y, w, h, 6.0, t::panel(0.94), t::line(1.0));
                    let px = pad / aspect * 1.2;
                    hud.label("Pit stop", x + px, y + pad + (hud.line_h(big) - lh) * 0.6, fs, Weight::Semibold, t::dim(1.0));
                    let s = format!("{left:.1} s");
                    let sw = hud.label_w(&s, big, Weight::Semibold);
                    hud.label(&s, x + w - px - sw, y + pad, big, Weight::Semibold, t::fg(1.0));
                    let mut by = y + pad + hud.line_h(big) + 0.006;
                    for (what, f) in rows {
                        hud.label(&what, x + px, by, fs * 0.9, Weight::Regular, t::dim(1.0));
                        let bx = x + w * 0.45;
                        let bwid = w * 0.55 - px;
                        let bh = hud.px_y(6.0);
                        let byy = by + lh * 0.5 - bh * 0.5;
                        hud.round_rect(bx, byy, bwid, bh, 3.0, t::well(1.0));
                        hud.round_rect(bx, byy, bwid * f.clamp(0.0, 1.0), bh, 3.0, if f >= 1.0 { t::ok(1.0) } else { t::accent(1.0) });
                        by += lh + 0.004;
                    }
                    y += h + 0.008;
                }
            }
            Stage::Held => {
                let s = format!("Penalty: held for {:.1} s", self.held_left.max(0.0));
                y += chip(hud, y, &s, fs, t::panel(0.94), t::fg(1.0), Some(t::red(1.0))) + 0.008;
            }
            _ => {}
        }
        // messages, newest last, a coloured edge for their kind
        for (m, ttl, c) in self.msgs.iter().rev().take(3) {
            let a = ttl.min(1.0);
            let mut s = m.clone();
            if let Some(f) = s.get(0..1) {
                s = f.to_uppercase() + &s[1..];
            }
            y += chip(hud, y, &s, fs * 0.92, t::panel(0.9 * a), t::fg(a), Some([c[0], c[1], c[2], a])) + 0.006;
        }
        if self.menu {
            self.draw_menu(hud, td, aspect);
        }
    }

    fn draw_menu(&self, hud: &mut crate::hud::Hud, td: &TrackDrive, aspect: f32) {
        use crate::hud::{theme as t, Weight};
        let rows = self.rows();
        let fs = 0.0135;
        let small = 0.0115;
        let w = 0.36 / aspect;
        let x = 0.02 / aspect * 1.6;
        let padx = 0.018 / aspect;
        let pady = 0.016;
        let rh = 0.034;
        let lh = hud.line_h(fs);
        let p = &self.plan;
        let (refuel, tyres, repair, total) = stop_times(&self.car, p);
        let (rb, rs) = self.car.repair_secs();
        let tread = td.rig.as_ref().and_then(|(rc, _)| rc.tread_mm());
        let head = hud.line_h(fs * 1.25) + 0.004 + hud.line_h(small) * if tread.is_some() { 2.0 } else { 1.0 } + 0.012;
        let foot = 0.012 + hud.line_h(small) * if self.stage == Stage::Garage { 1.0 } else { 2.0 };
        let h = pady * 2.0 + head + rows.len() as f32 * rh + 0.01 + foot;
        let y0 = (0.5 - h * 0.5).max(0.08);
        hud.panel(x, y0, w, h, 6.0, t::panel(0.95), t::line(1.0));
        let (ix, iw) = (x + padx, w - padx * 2.0);
        let mut y = y0 + pady;
        // header: title, the box on the right, the car's state under it
        let title = match self.stage {
            Stage::Garage => "Garage",
            Stage::Stopped => "Pit stop",
            _ => "Pit strategy",
        };
        hud.label(title, ix, y, fs * 1.25, Weight::Semibold, t::fg(1.0));
        if let Some(s) = self.spot() {
            let b = format!("Box {}", s.slot + 1);
            let bw = hud.label_w(&b, small, Weight::Regular);
            hud.label(&b, ix + iw - bw, y + (hud.line_h(fs * 1.25) - hud.line_h(small)) * 0.6, small, Weight::Regular, t::dim(1.0));
        }
        y += hud.line_h(fs * 1.25) + 0.004;
        let s = format!("Fuel {:.1} of {:.0} L", self.car.fuel, self.car.max_fuel);
        hud.label(&s, ix, y, small, Weight::Regular, t::dim(1.0));
        y += hud.line_h(small);
        if let Some(tr) = tread {
            let s = format!("Tread  FL {:.1}  FR {:.1}  RL {:.1}  RR {:.1} mm", tr[1], tr[0], tr[3], tr[2]);
            hud.label(&s, ix, y, small, Weight::Regular, t::dim(1.0));
            y += hud.line_h(small);
        }
        y += 0.006;
        hud.rect(x, hud.snap_y(y), w, hud.px_y(1.0), t::line(1.0));
        y += 0.006;
        let sel_i = self.row.min(rows.len() - 1);
        for (i, r) in rows.iter().enumerate() {
            let sel = i == sel_i;
            let cname = |ax: usize| self.car.compounds.get(p.compound[ax]).map_or("-", |c| c.name.as_str());
            let yesno = |on: bool, secs: f32| if secs <= 0.0 { "No damage".to_string() } else if on { format!("Yes, {secs:.0} s") } else { "No".to_string() };
            let (name, value): (String, String) = match *r {
                Row::Tyres => ("Tyres".into(), match p.tyres { [true, true] => "Change all", [true, false] => "Front only", [false, true] => "Rear only", _ => "Keep" }.into()),
                Row::Compound(2) => ("Compound".into(), cname(0).into()),
                Row::Compound(ax) => (["Front compound", "Rear compound"][ax].into(), cname(ax).into()),
                Row::Psi(k) => (format!("Pressure {}", ["FR", "FL", "RR", "RL"][k]), format!("{:.1} psi", p.psi[k])),
                Row::Fuel => ("Add fuel".into(), if p.fuel > 0.0 { format!("{:.0} L  ({:.0} s)", p.fuel, refuel) } else { "None".into() }),
                Row::Body => ("Repair bodywork".into(), yesno(p.body, rb)),
                Row::Susp => ("Repair suspension".into(), yesno(p.susp, rs)),
                Row::Arb(ax) => (["Front anti-roll bar", "Rear anti-roll bar"][ax].into(), format!("{:.0}", p.arb[ax])),
                Row::Aero(i) => (["Front ride height", "Rear ride height", "Front wing", "Rear wing"][i].into(),
                                 if i < 2 { format!("{:.0} mm", p.aero[i]) } else { format!("{:.1}\u{b0}", p.aero[i]) }),
                Row::Go => (match self.stage {
                    Stage::Stopped => format!("Start pit stop  ({total:.0} s)"),
                    Stage::Garage => "Apply and drive out".into(),
                    _ => "Close".into(),
                }, String::new()),
            };
            let ty = y + (rh - lh) * 0.5;
            if *r == Row::Go {
                // the main action: an accent button
                let by = y + 0.008;
                let bh = rh;
                hud.round_rect(ix, by, iw, bh, 4.0, if sel { t::accent(1.0) } else { t::hex(0xb02a22, 0.9) });
                let nw = hud.label_w(&name, fs, Weight::Semibold);
                hud.label(&name, ix + iw * 0.5 - nw * 0.5, by + (bh - lh) * 0.5, fs, Weight::Semibold,
                          t::hex(0xffffff, if sel { 1.0 } else { 0.85 }));
                y += rh + 0.008;
                continue;
            }
            let off = matches!(r, Row::Psi(k) if !p.tyres[k / 2]) || matches!(r, Row::Body if rb <= 0.0) || matches!(r, Row::Susp if rs <= 0.0);
            if sel {
                hud.round_rect(x + hud.px_x(1.0), y, w - hud.px_x(2.0), rh, 0.0, t::panel2(1.0));
                hud.rect(x + hud.px_x(1.0), y, hud.px_x(2.0), rh, t::accent(1.0));
            }
            let c = if off { t::dim2(1.0) } else if sel { t::fg(1.0) } else { t::hex(0xc9ced6, 1.0) };
            hud.label(&name, ix, ty, fs, Weight::Regular, c);
            let vw = hud.label_w(&value, fs, Weight::Semibold);
            let aw = hud.label_w("\u{203a}", fs, Weight::Semibold);
            let vx = ix + iw - vw - if sel { aw * 2.2 } else { 0.0 };
            hud.label(&value, vx, ty, fs, Weight::Semibold, if off { t::dim2(1.0) } else { t::fg(1.0) });
            if sel {
                hud.label("\u{2039}", vx - aw * 2.2, ty, fs, Weight::Semibold, t::dim(1.0));
                hud.label("\u{203a}", ix + iw - aw, ty, fs, Weight::Semibold, t::dim(1.0));
            }
            y += rh;
        }
        y += 0.01;
        if self.stage != Stage::Garage {
            let s = format!("Crew  refuel {refuel:.0} s, tyres {tyres:.0} s, repairs {repair:.0} s");
            hud.label(&s, ix, y, small, Weight::Regular, t::dim(1.0));
            y += hud.line_h(small);
        }
        let close = if matches!(self.stage, Stage::Stopped | Stage::Garage) { "" } else { "   Tab close" };
        let s = format!("\u{2191}\u{2193} choose   \u{2190}\u{2192} change   Enter confirm{close}");
        hud.label(&s, ix, y, small, Weight::Regular, t::dim2(1.0));
    }
}

/// The box's footprint on the ground, projected: a translucent car-sized
/// rectangle (2.4 x 5.2 m) with an outline.
fn box_marker(hud: &mut crate::hud::Hud, vp: Mat4, s: Spot, aspect: f32, c: [f32; 4]) {
    let f = s.fwd();
    let r = Vec3::new(f.z, 0.0, -f.x);
    let corners = [(-1.2, -2.6), (1.2, -2.6), (1.2, 2.6), (-1.2, 2.6)].map(|(a, b)| s.pos + r * a + f * b + Vec3::Y * 0.05);
    let mut sc = [(0.0f32, 0.0f32); 4];
    for (i, p) in corners.iter().enumerate() {
        let q = vp * Vec4::new(p.x, p.y, p.z, 1.0);
        if q.w < 0.05 {
            return;
        }
        sc[i] = ((q.x / q.w + 1.0) * 0.5, (1.0 - q.y / q.w) * 0.5);
    }
    let fill = [c[0], c[1], c[2], 0.22];
    hud.tri([sc[0], sc[1], sc[2]], fill);
    hud.tri([sc[0], sc[2], sc[3]], fill);
    // outline: each edge a thin quad, square in pixels
    for i in 0..4 {
        let (a, b) = (sc[i], sc[(i + 1) % 4]);
        let (dx, dy) = ((b.0 - a.0) * aspect, b.1 - a.1);
        let l = (dx * dx + dy * dy).sqrt().max(1e-6);
        let t = 0.0025;
        let (nx, ny) = (-dy / l * t / aspect, dx / l * t);
        let q = [(a.0 + nx, a.1 + ny), (b.0 + nx, b.1 + ny), (b.0 - nx, b.1 - ny), (a.0 - nx, a.1 - ny)];
        hud.tri([q[0], q[1], q[2]], c);
        hud.tri([q[0], q[2], q[3]], c);
    }
}

// ------------------------------------------------- start-up and testing --

/// EVOVIEW_PIT_START=box | lane | garage puts the car there at load, free
/// to drive off; box-menu | garage-menu | service start with the stop's /
/// the garage's menu up or the crew at work (captures);
/// EVOVIEW_PIT_LIMITER=1 starts with the limiter on.
pub fn start(td: &mut TrackDrive, p: &mut Pits) {
    if crate::envc::var_os("EVOVIEW_PIT_LIMITER").is_some() {
        p.limiter = true;
    }
    // drive straight out of wherever the session starts; the box only
    // takes the car once it has left and come back (Released -> Driving)
    p.stage = Stage::Released;
    let Ok(at) = crate::envc::var("EVOVIEW_PIT_START") else { return };
    match at.as_str() {
        // a session starts free to drive out; the box takes the car once it
        // has left and come back
        "garage" => {
            if let Some(s) = p.garage_spot().or(p.spot()) {
                td.place_from(s.pos, s.yaw, 1.5);
            }
        }
        // the menus up at once, for captures: garage-menu = back to the pits
        "garage-menu" => p.to_garage(td),
        "lane" => {
            let pts = p.lane.as_ref().map(|l| l.spline.clone()).unwrap_or_default();
            if pts.len() > 20 {
                let i = pts.len() / 3;
                let d = pts[i + 3] - pts[i];
                td.place_from(pts[i], d.x.atan2(d.z), 1.5);
                p.limiter = true;
            }
        }
        "box" | "box-menu" | "service" => {
            if let Some(s) = p.spot() {
                td.place_from(s.pos, s.yaw, 1.5);
                if at == "box-menu" {
                    p.stage = Stage::Driving;
                }
                if at == "service" {
                    p.stage = Stage::Stopped;
                    p.plan.fuel = 40.0f32.min((p.car.max_fuel - p.car.fuel).max(0.0));
                    p.start_service();
                }
            }
        }
        _ => {}
    }
}

/// The scripted pit test (`--pit-test <secs> [--pit-script stop|lap]`):
/// - stop: from ~350 m before the pit entry on the racing line, into the
///   lane (limiter on at the entry, off at the exit), stop in the box,
///   change four tyres and add fuel, out to the track; prints the lane
///   events, the limiter's top speed, wear / mass / fuel before and after
///   and the stop's time against the game's formula.
/// - lap: the racing line with the pit system watching (nothing must fire).
pub fn test(pkg: &std::path::Path, track: &str, td: &mut TrackDrive, p: &mut Pits, secs: f32, script: &str) {
    use crate::trackdrive::LineDriver;
    let line = crate::trackdrive::racing_line(pkg, track);
    let Some(lane) = p.lane.clone() else {
        println!("PITTEST no pit lane on {track}");
        return;
    };
    let near = |pts: &[Vec3], q: Vec3| -> usize {
        (0..pts.len()).min_by(|&a, &b| pts[a].distance_squared(q).total_cmp(&pts[b].distance_squared(q))).unwrap_or(0)
    };
    let dt = 1.0 / 120.0;
    if script == "lap" {
        let mut drv = LineDriver::new(line.clone());
        let mut t = 0.0;
        let mut next = 0.0;
        while t < secs {
            let mut c = drv.controls(td, dt);
            p.filter(&mut c, td);
            td.step(&c, dt);
            p.update(td, dt);
            if t >= next {
                next += 10.0;
                println!("PITTEST lap t {t:5.1}  {:5.1} km/h  pos ({:.0}, {:.0})  in lane {}  stage {:?}", td.speed_kmh(), td.pos.x, td.pos.z, p.in_lane, p.stage);
            }
            t += dt;
        }
        println!("PITTEST lap done: {} message(s) up, stage {:?}", p.msgs.len(), p.stage);
        return;
    }
    let (Some(spot), true) = (p.spot(), lane.spline.len() > 10 && line.len() > 10) else {
        println!("PITTEST needs a pit spot, the pit spline and the racing line");
        return;
    };
    // the loop: the pit spline (swung through the box), then the racing
    // line from the pit exit round to the pit entry
    let mut pit = lane.spline.clone();
    let ib = near(&pit, spot.pos);
    let dir = (pit[(ib + 2).min(pit.len() - 1)] - pit[ib.saturating_sub(2)]).normalize_or_zero();
    let side = Vec3::new(dir.z, 0.0, -dir.x);
    let off = (spot.pos - pit[ib]).dot(side);
    let mut s_at = vec![0.0f32; pit.len()];
    for i in 1..pit.len() {
        s_at[i] = s_at[i - 1] + pit[i].distance(pit[i - 1]);
    }
    let sb = s_at[ib] + (spot.pos - pit[ib]).dot(dir);
    for i in 0..pit.len() {
        let d = (s_at[i] - sb).abs();
        let w = if d < 8.0 { 1.0 } else if d < 40.0 { 0.5 + 0.5 * (std::f32::consts::PI * (d - 8.0) / 32.0).cos() } else { 0.0 };
        pit[i] += side * off * w;
    }
    let ie = near(&line, lane.spline[0]);
    let ix = near(&line, *lane.spline.last().unwrap());
    let n = line.len();
    let mut path = pit.clone();
    let mut i = (ix + 3) % n;
    while i != ie {
        path.push(line[i]);
        i = (i + 1) % n;
    }
    // start ~350 m before the entry
    let (mut k, mut run) = (ie, 0.0);
    while run < 350.0 {
        let j = (k + n - 1) % n;
        run += line[k].distance(line[j]);
        k = j;
    }
    let d = line[(k + 3) % n] - line[k];
    td.place(line[k], d.x.atan2(d.z));
    let mut path2: Vec<Vec3> = (0..((ie + n - k) % n)).map(|j| line[(k + j) % n]).collect();
    path2.extend(path);
    let mut drv = LineDriver::new(path2);
    println!("PITTEST start ({:.0}, {:.0}), entry at line {ie}, box {} at ({:.1}, {:.1}, {:.1}) yaw {:.0} (lateral swing {off:+.1} m)",
        td.pos.x, td.pos.z, spot.slot + 1, spot.pos.x, spot.pos.y, spot.pos.z, spot.yaw.to_degrees());
    let wear = |td: &TrackDrive| td.rig.as_ref().and_then(|(rc, _)| rc.tread_mm());
    let mass = |td: &TrackDrive| td.rig.as_ref().map_or(0.0, |(rc, _)| rc.body_mass());
    let (mut t, mut next) = (0.0f32, 0.0f32);
    let (mut lim_peak, mut in_was) = (0.0f32, false);
    let mut service_t0 = None;
    let mut before = None;
    let mut last_steer = 0.0f32;
    while t < secs {
        // the driver's limiter: on at the lane's entry, off at the exit
        if p.in_lane != in_was {
            in_was = p.in_lane;
            p.limiter = p.in_lane;
        }
        // creeping onto the box the line driver would think itself stuck
        // (pushing, under 1 m/s) and back out: it only steers above that
        let approach = p.stage == Stage::Driving && p.in_lane && p.box_error(td).map_or(false, |(a, c, _)| a < 0.5 && a > -70.0 && c.abs() < 6.0);
        let mut c = if p.holds() {
            Controls::default()
        } else if approach && td.vx.abs() < 1.5 {
            Controls { steer: last_steer, analog: true, ..Default::default() }
        } else {
            drv.controls(td, dt)
        };
        last_steer = c.steer;
        // a limiter only cuts the throttle: the driver brakes to the limit
        // by the limit line
        let lim = (p.rules.limit_kmh - 2.0) / 3.6;
        if p.in_lane && td.vx > lim {
            c.throttle = 0.0;
            c.brake = c.brake.max(((td.vx - lim) * 0.25).clamp(0.2, 1.0));
        }
        if p.stage == Stage::Driving && p.in_lane {
            if let Some((along, across, _)) = p.box_error(td) {
                // the box ahead: brake to stop on it
                if along < 0.5 && along > -70.0 && across.abs() < 6.0 {
                    let target = (2.0 * 3.0 * (-along).max(0.0)).sqrt().min(p.rules.limit_kmh / 3.6);
                    let v = td.vx;
                    if v > target + 0.3 || along > -0.25 {
                        c.throttle = 0.0;
                        c.brake = if along > -0.25 { 1.0 } else { ((v - target) * 0.6).clamp(0.15, 1.0) };
                    } else {
                        c.throttle = c.throttle.min(0.35);
                        c.brake = 0.0;
                    }
                }
            }
        }
        p.filter(&mut c, td);
        td.step(&c, dt);
        p.update(td, dt);
        if p.in_limit {
            lim_peak = lim_peak.max(td.speed_kmh());
        }
        if p.stage == Stage::Stopped {
            before = Some((wear(td), mass(td), p.car.fuel));
            p.plan.tyres = [true, true];
            p.plan.fuel = 40.0f32.min((p.car.max_fuel - p.car.fuel).max(0.0));
            let (rf, ty, rp, tot) = stop_times(&p.car, &p.plan);
            println!("PITTEST plan: four tyres + {:.0} L -> game times refuel {rf} s, tyres {ty} s, repairs {rp} s, total {tot} s", p.plan.fuel);
            p.key(Key::Ok, td);
            service_t0 = Some(t);
        }
        if let (Some(t0), Stage::Released) = (service_t0, p.stage) {
            let (w0, m0, f0) = before.take().unwrap_or((None, 0.0, 0.0));
            println!("PITTEST service took {:.2} s", t - t0);
            println!("PITTEST tread mm before {:?} after {:?} (new {:?})", w0, wear(td), td.rig.as_ref().map(|(rc, _)| rc.tread_new()));
            println!("PITTEST fuel {f0:.1} -> {:.1} L, body mass {m0:.1} -> {:.1} kg", p.car.fuel, mass(td));
            println!("PITTEST tyre psi now {:?}", td.rig.as_ref().and_then(|(rc, _)| rc.tyre_psi()));
            service_t0 = None;
        }
        if t >= next {
            next += 1.0;
            println!("PITTEST t {t:5.1}  {:5.1} km/h  pos ({:6.1}, {:6.1})  lane {} limit-zone {} limiter {}  stage {:?}{}",
                td.speed_kmh(), td.pos.x, td.pos.z, p.in_lane as u8, p.in_limit as u8, p.limiter as u8, p.stage,
                p.box_error(td).map_or(String::new(), |(a, c, _)| format!("  box {a:+.1}/{c:+.1} m")));
        }
        t += dt;
    }
    println!("PITTEST top speed in the limit zone {lim_peak:.1} km/h (limit {:.0})", p.rules.limit_kmh);
}
