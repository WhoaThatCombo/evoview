//! A client of a real AC EVO dedicated server (EVOVIEW_MP=host:port).
//!
//! Protocol (reversed 2026-11-02, notes in memory evoview-mp-client):
//! TCP frames are `u16 len` (0xFFFF + u32 for big ones) over a packet; a
//! packet is `u16 kind` + body, kind 2 = protobuf: `u8 namelen`, the type's
//! FULL name, the message bytes. UDP datagrams are packets without the length.
//!
//! 1. TCP ClientConnectionRequest (within 0.5 s of connecting) ->
//!    ConnectToServerHandshakeResponse {4 connection_id}
//! 2. UDP `u16 9, u32 connection_id` -> TCP AssociateUDPSocketResponse
//! 3. TCP RequestPlayerCar {the car's .carfinalstate} -> ChangePlayerCarResponseEvent
//!    {3 car_id, 4 car_number, 5 spawn transform}; then ClientLoaded + PlayerReady
//! 4. UDP car state, packet 1 (63 bytes), ~20 Hz each way; the server relays
//!    every car's packets verbatim, keyed by the u16 slot.
//!
//! ⚠ Test against our own servers only.

use glam::Vec3;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpStream, UdpSocket};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// One car's state as it goes over the wire (world space, game conventions).
#[derive(Clone, Copy, Debug, Default)]
pub struct CarState {
    pub pos: Vec3,
    /// the game's yaw / pitch / roll, radians (game yaw = -evoview yaw)
    pub ypr: Vec3,
    pub vel: Vec3,
    pub rpm: u16,
    pub gear: u8,
    pub throttle: f32,
    pub brake: f32,
    /// -1..1, + = right
    pub steer: f32,
    /// wheel speed, rad/s
    pub wheel: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Remote {
    pub state: CarState,
    /// local time of the last packet
    pub at: Instant,
}

/// One remote car's recent packets, for playback a little in the past
/// (a jitter buffer): (its own clock, s; state), oldest first.
/// ⚠ Each car stamps its packets with ITS OWN clock (offsets between cars
/// on one server ran 1000-4700 s apart), so the local <-> packet clock
/// offset is per car: the smallest arrival - stamp seen (the least delayed
/// packet), let drift up slowly.
struct Hist {
    buf: std::collections::VecDeque<(f64, CarState)>,
    /// local seconds - packet seconds, for the quickest packet
    off: f64,
    /// the offset playback runs at, slewed toward `off` at up to 5% of
    /// real time (a quicker packet lowering `off` 60 ms jumped the car
    /// ~2 m in one frame); and the local time it was last moved
    play: Option<(f64, f64)>,
}

/// How far behind the newest packet remote cars are played (s): two
/// packet intervals (the game sends every 57 ms), so a late or lost packet
/// is still interpolated, not guessed. EVOVIEW_MP_DELAY overrides.
fn interp_delay() -> f64 {
    crate::envc::var("EVOVIEW_MP_DELAY").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.12)
}

impl Hist {
    fn push(&mut self, local: f64, t: f64, st: CarState) {
        let newest = self.buf.back().map(|b| b.0);
        // a clock jump (the sender rejoined, a reset): start over
        if newest.map_or(false, |n| t < n - 2.0 || t > n + 10.0) {
            self.buf.clear();
        }
        let o = local - t;
        if self.buf.is_empty() || o < self.off {
            self.off = o;
        } else {
            // route latency can grow: follow it, slowly (~2 s)
            self.off += (o - self.off) * 0.02;
        }
        // in order; a duplicate or a stale reorder is dropped
        match newest {
            Some(n) if t <= n => {
                if !self.buf.iter().any(|b| (b.0 - t).abs() < 1e-4) {
                    let i = self.buf.iter().position(|b| b.0 > t).unwrap_or(self.buf.len());
                    self.buf.insert(i, (t, st));
                }
            }
            _ => self.buf.push_back((t, st)),
        }
        while self.buf.len() > 32 {
            self.buf.pop_front();
        }
    }

    /// The car at local time `local` minus the playback delay: Hermite
    /// between the packets around it (their velocities as tangents), else
    /// carried on from the newest for at most 0.25 s.
    fn sample(&mut self, local: f64) -> Option<CarState> {
        let off = match self.play {
            Some((o, at)) if (o - self.off).abs() < 0.5 => {
                let k = (local - at).max(0.0) * 0.05;
                o + (self.off - o).clamp(-k, k)
            }
            _ => self.off,
        };
        self.play = Some((off, local));
        let t = local - off - interp_delay();
        let last = self.buf.back()?;
        if t >= last.0 {
            let mut s = last.1;
            let dt = (t - last.0).min(0.25) as f32;
            s.pos += s.vel * dt;
            return Some(s);
        }
        let i = self.buf.iter().position(|b| b.0 > t)?;
        if i == 0 {
            return Some(self.buf[0].1);
        }
        let ((t0, a), (t1, b)) = (self.buf[i - 1], self.buf[i]);
        let h = (t1 - t0).max(1e-4);
        let u = ((t - t0) / h).clamp(0.0, 1.0) as f32;
        let h = h as f32;
        let mut s = if u < 0.5 { a } else { b };
        s.pos = if h < 0.5 {
            let (u2, u3) = (u * u, u * u * u);
            a.pos * (2.0 * u3 - 3.0 * u2 + 1.0) + a.vel * (h * (u3 - 2.0 * u2 + u))
                + b.pos * (-2.0 * u3 + 3.0 * u2) + b.vel * (h * (u3 - u2))
        } else {
            // a gap (lost packets): straight between
            a.pos.lerp(b.pos, u)
        };
        let ang = |x: f32, y: f32| {
            let d = (y - x + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
            x + d * u
        };
        s.ypr = Vec3::new(ang(a.ypr.x, b.ypr.x), ang(a.ypr.y, b.ypr.y), ang(a.ypr.z, b.ypr.z));
        s.vel = a.vel.lerp(b.vel, u);
        s.steer = a.steer + (b.steer - a.steer) * u;
        s.throttle = a.throttle + (b.throttle - a.throttle) * u;
        s.brake = a.brake + (b.brake - a.brake) * u;
        s.wheel = a.wheel + (b.wheel - a.wheel) * u;
        s.rpm = (a.rpm as f32 + (b.rpm as f32 - a.rpm as f32) * u) as u16;
        Some(s)
    }
}

#[derive(Default)]
struct Shared {
    status: String,
    /// the pit box the server put us in: position, rotation (degrees)
    spawn: Option<(Vec3, Vec3)>,
    /// our car's state, to send
    me: Option<CarState>,
    remotes: HashMap<u16, Remote>,
    /// each remote car's recent packets (jitter buffer)
    hist: HashMap<u16, Hist>,
    /// local clock origin for `hist`
    epoch: Option<Instant>,
    slot: u16,
    /// the other cars by number: (car folder, driver), from CreateNetcarCommand
    roster: HashMap<u16, (String, String)>,
    /// their CarFinalStateData (model, design, paints) by number
    states: HashMap<u16, Vec<u8>>,
    /// every car number the session lists (the handshake's entry data) -
    /// a public server only streams the cars near us, so these are asked
    /// for up front
    listed: Vec<u16>,
    /// netcar requests sent, by number (re-asked after 2 s)
    asked: HashMap<u16, Instant>,
    /// the session's clock: start of day (minutes), time multiplier, date
    clock: Option<(f64, f64, (i32, u32, u32))>,
    /// server world time (s) at a local instant, and the game mode's start
    world: Option<(f64, Instant)>,
    mode_start: f64,
    /// opponent index -> the model loaded there and the car number drawn there
    ai: Vec<(String, Option<u16>)>,
    /// the server's chat: (when, sender, text, kind), newest last
    chat: std::collections::VecDeque<(Instant, String, String, u32)>,
    /// lines typed here, waiting for the network thread
    say: Vec<String>,
    /// our driver name (shown on our own lines)
    me_name: String,
    /// the spare slots' model (our car)
    spare: String,
    /// the session's scene containers (the layout the server runs)
    containers: Vec<String>,
    /// the session's track folder name
    track: Option<String>,
    /// car numbers with a slot of their own model and look, and those being
    /// loaded for (late joiners)
    loaded: std::collections::HashSet<u16>,
    loading: std::collections::HashSet<u16>,
}

pub struct Mp {
    sh: Arc<Mutex<Shared>>,
}

/// Packet `<HBiH3f3f3f4BHHBBBH4BB`, 63 bytes. ⚠ The car's NUMBER is the u16
/// at byte 56 (writer FUN_141234ae0's param+0x5c); the u16 at byte 7 is a
/// per-sender value that jitters by a few units (ping-like, >= 1) and is NOT
/// an id - keyed on it, cars swapped places every packet on a full server.
pub fn car_packet(seq: u8, ms: i32, number: u16, s: &CarState) -> Vec<u8> {
    let mut b = Vec::with_capacity(63);
    b.extend_from_slice(&1u16.to_le_bytes());
    b.push(seq);
    b.extend_from_slice(&ms.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    for v in [s.pos, s.ypr, s.vel] {
        for c in v.to_array() {
            b.extend_from_slice(&c.to_le_bytes());
        }
    }
    // wheel speeds: sign * round(log10(|w| + 1) * 20) + 100
    let w = s.wheel;
    let wb = (w.signum() * ((w.abs() + 1.0).log10() * 20.0).round() + 100.0).clamp(0.0, 255.0) as u8;
    b.extend_from_slice(&[wb; 4]);
    b.extend_from_slice(&((32767.0 + s.steer.clamp(-1.0, 1.0) * 32767.0) as u16).to_le_bytes());
    b.extend_from_slice(&s.rpm.max(1).to_le_bytes());
    b.push(s.gear);
    b.push((s.throttle.clamp(0.0, 1.0) * 255.0) as u8);
    b.push((s.brake.clamp(0.0, 1.0) * 255.0) as u8);
    b.extend_from_slice(&number.to_le_bytes());
    b.extend_from_slice(&[0; 4]);
    b.push(127);
    b
}

/// The other half of `car_packet`: (car number, state).
pub fn parse_car(p: &[u8]) -> Option<(u16, CarState)> {
    if p.len() < 63 || u16::from_le_bytes([p[0], p[1]]) != 1 {
        return None;
    }
    let f = |o: usize| f32::from_le_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]]);
    let v = |o: usize| Vec3::new(f(o), f(o + 4), f(o + 8));
    let slot = u16::from_le_bytes([p[56], p[57]]);
    let wb = p[45] as f32 - 100.0;
    let steer = (u16::from_le_bytes([p[49], p[50]]) as f32 - 32767.0) / 32767.0;
    Some((slot, CarState {
        pos: v(9),
        ypr: v(21),
        vel: v(33),
        wheel: wb.signum() * (10f32.powf(wb.abs() / 20.0) - 1.0),
        steer,
        rpm: u16::from_le_bytes([p[51], p[52]]),
        gear: p[53],
        throttle: p[54] as f32 / 255.0,
        brake: p[55] as f32 / 255.0,
    }))
}

// ---- a tiny protobuf writer ------------------------------------------------

fn varint(b: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        b.push((v as u8) | 0x80);
        v >>= 7;
    }
    b.push(v as u8);
}
fn f_varint(b: &mut Vec<u8>, n: u32, v: u64) {
    varint(b, (n as u64) << 3);
    varint(b, v);
}
fn f_bytes(b: &mut Vec<u8>, n: u32, v: &[u8]) {
    varint(b, ((n as u64) << 3) | 2);
    varint(b, v.len() as u64);
    b.extend_from_slice(v);
}
fn f_f32(b: &mut Vec<u8>, n: u32, v: f32) {
    varint(b, ((n as u64) << 3) | 5);
    b.extend_from_slice(&v.to_le_bytes());
}

/// A kind-2 (protobuf) packet body.
fn proto_packet(name: &str, msg: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(msg.len() + name.len() + 3);
    b.extend_from_slice(&2u16.to_le_bytes());
    b.push(name.len() as u8);
    b.extend_from_slice(name.as_bytes());
    b.extend_from_slice(msg);
    b
}

fn tcp_frame(body: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(body.len() + 6);
    if body.len() < 0xffff {
        f.extend_from_slice(&(body.len() as u16).to_le_bytes());
    } else {
        f.extend_from_slice(&0xffffu16.to_le_bytes());
        f.extend_from_slice(&(body.len() as u32 + 2).to_le_bytes());
    }
    f.extend_from_slice(body);
    f
}

fn pguid(a: u64, b: u64) -> Vec<u8> {
    let mut m = Vec::new();
    if a != 0 {
        f_varint(&mut m, 1, a);
    }
    if b != 0 {
        f_varint(&mut m, 2, b);
    }
    m
}

/// The car's .carfinalstate from the game package, and its mechanical
/// preset name (what the server's allowed list holds, e.g. preset_r8gt3_mech_1).
pub fn car_final_state(pkg: &std::path::Path, car: &str) -> Option<(Vec<u8>, String)> {
    let dir = format!("content\\cars\\{}\\presets\\", car.to_lowercase());
    // a mod car's presets are in its own package (Saved Games/ACE/mods/<car>.kspkg):
    // looked for only in the game's, a mod car joined as no one - single player
    let has = |p: &crate::pkg::Package| p.by_path.keys().any(|k| k.ends_with(".carfinalstate"));
    let p = match crate::pkg::Package::open_filtered(pkg, &[&dir]).ok().filter(|p| has(p)) {
        Some(p) => p,
        None => {
            let m = crate::envc::var_os("USERPROFILE")
                .map(|h| std::path::Path::new(&h).join("Saved Games").join("ACE").join("mods").join(format!("{car}.kspkg")))
                .filter(|m| m.exists())?;
            println!("mp: {car} is a mod: presets from {}", m.display());
            crate::pkg::Package::open_filtered(&m, &[&dir]).ok().filter(|p| has(p))?
        }
    };
    let want = crate::envc::var("EVOVIEW_MP_PRESET").ok();
    let mut keys: Vec<&String> = p.by_path.keys().filter(|k| k.ends_with(".carfinalstate")).collect();
    keys.sort_by_key(|k| (!k.contains("mech_1"), !k.contains("visual_1"), k.len()));
    let key = match want.as_deref() {
        Some(w) => match keys.iter().find(|k| k.contains(w)) {
            Some(k) => *k,
            None => {
                println!("mp: {car} has no preset {w}; joining with its first");
                *keys.first()?
            }
        },
        None => *keys.first()?,
    };
    let d = p.read(*p.by_path.get(key)?);
    // 10 body { 1 path ".../preset_X_mech_1.mechanicalcarpreset" }
    let mut mech = String::new();
    crate::proto::fields(&d, 0, d.len(), |n, v| {
        if let (10, Some((a, b))) = (n, v.range()) {
            crate::proto::fields(&d, a, b, |n2, v2| {
                if let (1, Some(r)) = (n2, v2.range()) {
                    let s = crate::proto::text(&d, r);
                    if s.ends_with(".mechanicalcarpreset") {
                        mech = s.rsplit(['\\', '/']).next().unwrap_or("").trim_end_matches(".mechanicalcarpreset").to_string();
                    }
                }
            });
        }
    });
    println!("mp: car {key} (mechanical preset {mech})");
    Some((d, mech))
}

impl Mp {
    /// Connect in the background; `fs` / `mech` from `car_final_state`.
    /// Not connected, and why: evoview drives offline under the red banner
    /// rather than looking like it joined.
    pub fn failed(why: String) -> Mp {
        eprintln!("mp: {why}");
        Mp { sh: Arc::new(Mutex::new(Shared { status: format!("disconnected: {why}"), ..Default::default() })) }
    }

    pub fn start(addr: String, user: String, name: String, fs: Vec<u8>, mech: String) -> Mp {
        let sh = Arc::new(Mutex::new(Shared { status: format!("connecting to {addr}"), me_name: name.clone(), ..Default::default() }));
        let s2 = sh.clone();
        std::thread::Builder::new().name("mp".into()).spawn(move || {
            if let Err(e) = run(&addr, &user, &name, &fs, &mech, &s2) {
                eprintln!("mp: {e}");
                s2.lock().unwrap().status = format!("disconnected: {e}");
            }
        }).ok();
        Mp { sh }
    }

    /// The pit box the server gave us, once it has: position, and yaw in
    /// evoview's convention.
    pub fn spawn(&self) -> Option<(Vec3, f32)> {
        // the transform's rotation: degrees, y the heading (game convention)
        self.sh.lock().unwrap().spawn.map(|(p, r)| (p, -r.y.to_radians()))
    }

    pub fn set_me(&self, s: CarState) {
        self.sh.lock().unwrap().me = Some(s);
    }

    /// Every other car heard from in the last 5 s.
    pub fn remotes(&self) -> Vec<(u16, Remote)> {
        let mut g = self.sh.lock().unwrap();
        g.remotes.retain(|_, r| r.at.elapsed() < Duration::from_secs(5));
        let mut v: Vec<_> = g.remotes.iter().map(|(k, r)| (*k, *r)).collect();
        v.sort_by_key(|r| r.0);
        v
    }

    pub fn status(&self) -> String {
        self.sh.lock().unwrap().status.clone()
    }

    /// The connection ended (the reason), so evoview is driving on its own.
    pub fn lost(&self) -> Option<String> {
        let s = self.status();
        s.strip_prefix("disconnected: ").map(str::to_string)
    }

    /// Wait (up to `secs`) for our car and for the netcars of the cars heard
    /// so far; then the roster: (number, car folder, driver), by number.
    pub fn wait_roster(&self, secs: f32) -> Vec<(u16, String, String)> {
        let t = Instant::now();
        let mut quiet: Option<((usize, usize), Instant)> = None;
        while t.elapsed().as_secs_f32() < secs {
            {
                let g = self.sh.lock().unwrap();
                if g.status.starts_with("disconnected: ") {
                    break;
                }
                let heard = g.remotes.len() + g.listed.len();
                let known = g.remotes.keys().chain(g.listed.iter()).filter(|k| **k == g.slot || g.roster.contains_key(k)).count();
                if g.spawn.is_some() && heard > 0 && known == heard {
                    break;
                }
                // or 1.5 s without news (an empty server, cars that never answer)
                let key = (heard, known);
                match quiet {
                    Some((k, at)) if k == key => {
                        if g.spawn.is_some() && at.elapsed().as_secs_f32() > 1.5 && t.elapsed().as_secs_f32() > 3.0 {
                            break;
                        }
                    }
                    _ => quiet = Some((key, Instant::now())),
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let g = self.sh.lock().unwrap();
        let mut v: Vec<_> = g.roster.iter().map(|(k, (c, n))| (*k, c.clone(), n.clone())).collect();
        v.sort_by_key(|r| r.0);
        v
    }

    /// A car's CarFinalStateData as its netcar carried it.
    pub fn final_state(&self, n: u16) -> Option<Vec<u8>> {
        self.sh.lock().unwrap().states.get(&n).cloned()
    }

    /// The opponents as loaded: (model, the car number placed there at start).
    pub fn set_ai(&self, ai: Vec<(String, Option<u16>)>, spare: &str) {
        let mut g = self.sh.lock().unwrap();
        g.loaded = ai.iter().filter(|a| !a.0.eq_ignore_ascii_case(spare)).filter_map(|a| a.1).collect();
        g.ai = ai;
        g.spare = spare.to_string();
    }

    /// Players whose car has no slot of its own yet (joined after us, or a
    /// model we had no slot for): (number, car folder, final state). Each is
    /// handed out once; `add_ai` gives it its slot when loaded.
    pub fn new_cars(&self) -> Vec<(u16, String, Option<Vec<u8>>)> {
        let mut g = self.sh.lock().unwrap();
        if crate::envc::var_os("EVOVIEW_MP_LOG").is_some() {
            static LAST: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
            let mut l = LAST.lock().unwrap();
            if l.map_or(true, |t| t.elapsed().as_secs_f32() > 5.0) {
                *l = Some(Instant::now());
                let heard_roster = g.roster.keys().filter(|n| g.remotes.contains_key(n)).count();
                println!("mp new_cars: roster {} heard {} heard-in-roster {} loaded {:?} loading {}", g.roster.len(), g.remotes.len(), heard_roster, g.loaded, g.loading.len());
            }
        }
        let fresh: Vec<u16> = g.roster.keys().copied()
            .filter(|n| !g.loaded.contains(n) && !g.loading.contains(n) && g.remotes.contains_key(n))
            .collect();
        let mut out = Vec::new();
        for n in fresh {
            g.loading.insert(n);
            let car = g.roster.get(&n).map(|r| r.0.clone()).unwrap_or_default();
            if car.is_empty() || car.eq_ignore_ascii_case(&g.spare) {
                continue;
            }
            println!("mp: loading #{n}'s car ({car})");
            out.push((n, car, g.states.get(&n).cloned()));
        }
        if crate::envc::var_os("EVOVIEW_MP_LOG").is_some() && !out.is_empty() {
            println!("mp: {} car(s) to load; roster {}, heard {}, loaded {}", out.len(), g.roster.len(), g.remotes.len(), g.loaded.len());
        }
        out
    }

    /// A late joiner's car is loaded at a new opponent index: it is drawn
    /// there from now on (off any spare it was borrowing).
    pub fn add_ai(&self, model: &str, n: u16) {
        let mut g = self.sh.lock().unwrap();
        for a in g.ai.iter_mut() {
            if a.1 == Some(n) {
                a.1 = None;
            }
        }
        g.ai.push((model.to_string(), Some(n)));
        g.loaded.insert(n);
        g.loading.remove(&n);
    }

    /// Per opponent index: the other car drawn there. A car not placed yet
    /// takes a free index loaded with its model, else any free one; a car
    /// silent for 5 s frees its index.
    pub fn placed(&self) -> Vec<Option<Remote>> {
        let mut g = self.sh.lock().unwrap();
        g.remotes.retain(|_, r| r.at.elapsed() < Duration::from_secs(5));
        let live: std::collections::HashSet<u16> = g.remotes.keys().copied().collect();
        g.hist.retain(|n, _| live.contains(n));
        for a in g.ai.iter_mut() {
            if a.1.map_or(false, |n| !live.contains(&n)) {
                a.1 = None;
            }
        }
        // only cars the server has a netcar for (a public server streams more
        // numbers than it has players: stale entries, which drew phantom cars
        // and took the spare slots); EVOVIEW_MP_ALL=1 draws every number
        let all = crate::envc::var_os("EVOVIEW_MP_ALL").is_some();
        let mut waiting: Vec<u16> = live.iter().copied()
            .filter(|n| (all || g.roster.contains_key(n)) && !g.ai.iter().any(|a| a.1 == Some(*n))).collect();
        waiting.sort();
        for n in waiting {
            // a ghost: a second number at the very spot of a car already drawn
            // (public servers carry some - a rejoin's old entry); drawn once
            let p = g.remotes[&n].state.pos;
            if g.ai.iter().filter_map(|a| a.1).any(|m| g.remotes.get(&m).map_or(false, |r| r.state.pos.distance(p) < 0.5)) {
                continue;
            }
            let model = g.roster.get(&n).map(|r| r.0.to_lowercase());
            // its own model, else a spare (our car: plainly a stand-in, not
            // some other player's model), else anything free
            let spare = g.spare.clone();
            let idx = g.ai.iter().position(|a| a.1.is_none() && model.as_deref() == Some(a.0.to_lowercase().as_str()))
                .or_else(|| g.ai.iter().position(|a| a.1.is_none() && a.0.eq_ignore_ascii_case(&spare)))
                .or_else(|| g.ai.iter().position(|a| a.1.is_none()));
            if let Some(i) = idx {
                g.ai[i].1 = Some(n);
            }
        }
        // played from the jitter buffer (EVOVIEW_MP_DELAY=0: the newest
        // packet carried on by its age, as before)
        let now = g.epoch.map_or(0.0, |e| e.elapsed().as_secs_f64());
        let smooth = interp_delay() > 0.0;
        let g = &mut *g;
        g.ai.iter().map(|a| a.1.and_then(|n| {
            let r = g.remotes.get(&n).copied()?;
            match g.hist.get_mut(&n).filter(|_| smooth).and_then(|h| h.sample(now)) {
                Some(st) => Some(Remote { state: st, at: Instant::now() }),
                None => Some(r),
            }
        })).collect()
    }

    /// The chat, oldest first: (seconds ago, sender, text, kind 0 normal ..
    /// 3 server admin, 4 server message, 5 whisper).
    pub fn chat(&self) -> Vec<(f32, String, String, u32)> {
        let g = self.sh.lock().unwrap();
        g.chat.iter().map(|c| (c.0.elapsed().as_secs_f32(), c.1.clone(), c.2.clone(), c.3)).collect()
    }

    /// Say something in the server's chat.
    pub fn say(&self, text: &str) {
        let t = text.trim();
        if !t.is_empty() {
            self.sh.lock().unwrap().say.push(t.chars().take(200).collect());
        }
    }

    /// The server's track (the session's track folder, e.g. "nurburgring").
    pub fn track(&self) -> Option<String> {
        self.sh.lock().unwrap().track.clone()
    }

    /// The scene containers of the server's session (its layout), as the
    /// handshake listed them: "content\\tracks\\<t>\\containers\\x.scene".
    pub fn containers(&self) -> Vec<String> {
        self.sh.lock().unwrap().containers.clone()
    }

    /// The player drawn at opponent index `k`: (car number, car folder, driver).
    pub fn who(&self, k: usize) -> Option<(u16, String, String)> {
        let g = self.sh.lock().unwrap();
        let n = g.ai.get(k)?.1?;
        let (car, name) = g.roster.get(&n).cloned().unwrap_or_default();
        Some((n, car, name))
    }

    /// The session's time of day now (minutes, 0..1440) and its date, once
    /// the server has sent its clock.
    pub fn game_time(&self) -> Option<(f64, (i32, u32, u32))> {
        let g = self.sh.lock().unwrap();
        let (start, mult, date) = g.clock?;
        let (w, at) = g.world?;
        let el = (w + at.elapsed().as_secs_f64() - g.mode_start).max(0.0);
        Some(((start + el * mult / 60.0).rem_euclid(1440.0), date))
    }
}

/// Whether a socket error means the connection is gone (else it is passing:
/// a timeout, a would-block, Windows' os error 997 on a timed-out read).
fn lost(e: &std::io::Error) -> bool {
    use std::io::ErrorKind::*;
    matches!(e.kind(), ConnectionReset | ConnectionAborted | NotConnected | BrokenPipe | UnexpectedEof)
}

/// Write a whole frame, riding out passing errors (up to 2 s) without
/// resending what already went: write_all gives up on the first.
fn write_frame(tcp: &mut TcpStream, b: &[u8]) -> Result<(), String> {
    let mut off = 0;
    let t = Instant::now();
    while off < b.len() {
        match tcp.write(&b[off..]) {
            Ok(0) => return Err("server closed the connection".into()),
            Ok(n) => off += n,
            Err(e) if !lost(&e) && t.elapsed() < Duration::from_secs(2) => std::thread::sleep(Duration::from_millis(2)),
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(())
}

/// The session's clock from the handshake: 5 gamemode_changed_event { 1 season
/// { 10 event_map { 2 { 4 session_map { 2 { 40 weather { 1 initial_date_time
/// {1 year 2 month 3 day 4 hour 5 minute 6 second}, 2 time_multiplier }}}}}}}
/// - the first event's first session.
fn session_clock(d: &[u8]) -> Option<(f64, f64, (i32, u32, u32))> {
    use crate::proto::{fields, Val};
    let sub = |r: (usize, usize), want: u32| {
        let mut o = None;
        fields(d, r.0, r.1, |n, v| {
            if n == want && o.is_none() {
                o = v.range();
            }
        });
        o
    };
    let mut r = (0, d.len());
    for want in [5u32, 1, 10, 2, 4, 2, 40] {
        r = sub(r, want)?;
    }
    let (mut mult, mut dt) = (1.0f64, None);
    fields(d, r.0, r.1, |n, v| match (n, v) {
        (2, Val::F32(m)) => mult = m as f64,
        (1, Val::Bytes(a, b)) => dt = Some((a, b)),
        _ => {}
    });
    let (a, b) = dt?;
    let mut t = [0i64; 6];
    fields(d, a, b, |n, v| {
        if let (1..=6, Val::Varint(x)) = (n, v) {
            t[n as usize - 1] = x as i64;
        }
    });
    Some(((t[3] * 60 + t[4]) as f64 + t[5] as f64 / 60.0, mult, (t[0] as i32, t[1].max(1) as u32, t[2].max(1) as u32)))
}

/// The session's track: 5 { 1 season { 10 { 2 { 4 { 2 session { 21 scene {
/// 1 track_content_data { 3 folder_path "content\\tracks\\<id>" }}}}}}}}.
fn session_track(d: &[u8]) -> Option<String> {
    use crate::proto::fields;
    let sub = |r: (usize, usize), want: u32| {
        let mut o = None;
        fields(d, r.0, r.1, |n, v| {
            if n == want && o.is_none() {
                o = v.range();
            }
        });
        o
    };
    let mut r = (0, d.len());
    for want in [5u32, 1, 10, 2, 4, 2, 21, 1, 3] {
        r = sub(r, want)?;
    }
    let p = crate::proto::text(d, r).replace(char::from(92), "/");
    p.trim_end_matches('/').rsplit('/').next().map(|s| s.to_lowercase()).filter(|s| !s.is_empty())
}

/// Who is signed in to Steam here (config/loginusers.vdf, the account Steam
/// remembers): (steam id, persona name). What evoview joins as unless
/// EVOVIEW_MP_USER / EVOVIEW_MP_NAME say otherwise.
pub fn steam_user() -> Option<(String, String)> {
    let roots = [
        crate::envc::var("ProgramFiles(x86)").ok().map(|p| format!("{p}/Steam")),
        Some("C:/Program Files (x86)/Steam".to_string()),
    ];
    let txt = roots.iter().flatten().find_map(|r| std::fs::read_to_string(format!("{r}/config/loginusers.vdf")).ok())?;
    // "7656...." { ... "PersonaName" "x" ... "MostRecent" "1" } - the most
    // recent account, else the first
    let mut users: Vec<(String, String, bool)> = Vec::new();
    let mut cur: Option<(String, String, bool)> = None;
    for line in txt.lines() {
        let q: Vec<&str> = line.split('"').collect();
        // a bare quoted key: an account id opening its block
        if q.len() == 3 && q[1].len() == 17 && q[1].chars().all(|c| c.is_ascii_digit()) {
            if let Some(c) = cur.take() {
                users.push(c);
            }
            cur = Some((q[1].to_string(), String::new(), false));
        } else if q.len() >= 5 {
            if let Some(c) = cur.as_mut() {
                match q[1] {
                    "PersonaName" => c.1 = q[3].to_string(),
                    "MostRecent" => c.2 = q[3] == "1",
                    _ => {}
                }
            }
        }
    }
    if let Some(c) = cur {
        users.push(c);
    }
    let u = users.iter().find(|u| u.2).or(users.first())?;
    Some((u.0.clone(), u.1.clone()))
}

/// The session's scene containers: 5 { 1 season { 10 { 2 { 4 { 2 session
/// { 21 scene { 4 containers (repeated) }}}}}}}.
fn session_containers(d: &[u8]) -> Vec<String> {
    use crate::proto::fields;
    let sub = |r: (usize, usize), want: u32| {
        let mut o = None;
        fields(d, r.0, r.1, |n, v| {
            if n == want && o.is_none() {
                o = v.range();
            }
        });
        o
    };
    let mut r = (0, d.len());
    for want in [5u32, 1, 10, 2, 4, 2, 21] {
        match sub(r, want) {
            Some(x) => r = x,
            None => return Vec::new(),
        }
    }
    let mut out = Vec::new();
    fields(d, r.0, r.1, |n, v| {
        if let (4, Some(x)) = (n, v.range()) {
            out.push(crate::proto::text(d, x));
        }
    });
    out
}

/// CreateNetcarCommand {1 car_spawn_data {5 Any {2 CarFinalStateData {1 car_data
/// "content/cars/<folder>/data/x.car"}}}, 2 driver_info {1 first, 2 last,
/// 3 nickname}, 7 car_number}: (number, folder, driver).
fn netcar(d: &[u8]) -> Option<(u16, String, String, Vec<u8>)> {
    use crate::proto::{fields, text, Val};
    let (mut num, mut car, mut nick, mut first, mut last) = (None, String::new(), String::new(), String::new(), String::new());
    let mut fs: Vec<u8> = Vec::new();
    fields(d, 0, d.len(), |n, v| match (n, v) {
        (7, Val::Varint(x)) => num = Some(x as u16),
        (1, Val::Bytes(a, b)) => fields(d, a, b, |n, v| {
            if let (5, Some((a, b))) = (n, v.range()) {
                fields(d, a, b, |n, v| {
                    if let (2, Some((a, b))) = (n, v.range()) {
                        fs = d[a..b].to_vec();
                        fields(d, a, b, |n, v| {
                            if let (1, Some(r)) = (n, v.range()) {
                                let p = text(d, r).replace(char::from(92), "/");
                                let parts: Vec<&str> = p.split('/').collect();
                                if let Some(i) = parts.iter().position(|x| x.eq_ignore_ascii_case("cars")) {
                                    car = parts.get(i + 1).unwrap_or(&"").to_string();
                                }
                            }
                        });
                    }
                });
            }
        }),
        (2, Val::Bytes(a, b)) => fields(d, a, b, |n, v| match (n, v.range()) {
            (1, Some(r)) => first = text(d, r),
            (2, Some(r)) => last = text(d, r),
            (3, Some(r)) => nick = text(d, r),
            _ => {}
        }),
        _ => {}
    });
    let name = if !nick.is_empty() { nick } else { format!("{first} {last}").trim().to_string() };
    // ⚠ proto3 leaves a 0 out: car number 0 (the first car on a server)
    // has no field 7 at all - reading that as 'no number' hid that car
    Some((num.unwrap_or(0), car, name, fs))
}

/// The field of UpdateGameCarState another evoview's car rides in (see run).
const EVOVIEW_TAG: u32 = 99;

/// Another evoview's car from a relayed UpdateGameCarState: (number, car
/// folder, driver, CarFinalStateData) - what CreateNetcarCommand gives for a
/// game client's car.
fn evoview_tag(d: &[u8]) -> Option<(u16, String, String, Vec<u8>)> {
    use crate::proto::{fields, text, Val};
    let mut r = None;
    fields(d, 0, d.len(), |n, v| {
        if let (EVOVIEW_TAG, Some(x)) = (n, v.range()) {
            r = Some(x);
        }
    });
    let (a, b) = r?;
    let (mut ok, mut num, mut fs, mut who) = (false, 0u16, Vec::new(), String::new());
    fields(d, a, b, |n, v| match (n, v) {
        (1, Val::Bytes(x, y)) => ok = &d[x..y] == b"evoview",
        (2, Val::Varint(x)) => num = x as u16,
        (3, Val::Bytes(x, y)) => fs = d[x..y].to_vec(),
        (4, Val::Bytes(x, y)) => who = text(d, (x, y)),
        _ => {}
    });
    if !ok || fs.is_empty() {
        return None;
    }
    // CarFinalStateData {1 car_data "content\\cars\\<car>\\data\\<car>.car"}
    let mut car = String::new();
    fields(&fs, 0, fs.len(), |n, v| {
        if let (1, Some(r)) = (n, v.range()) {
            let p = text(&fs, r).replace(char::from(92), "/");
            let parts: Vec<&str> = p.split('/').collect();
            if let Some(i) = parts.iter().position(|x| x.eq_ignore_ascii_case("cars")) {
                car = parts.get(i + 1).unwrap_or(&"").to_string();
            }
        }
    });
    (!car.is_empty()).then_some((num, car, who, fs))
}

/// Every car number the session's entry data lists: 5 { 1 season { 101
/// event_mutable_data { 2 { 1 cars { 2 assigned_car_number }}}}}.
fn session_cars(d: &[u8]) -> Vec<u16> {
    use crate::proto::{fields, Val};
    let mut out = Vec::new();
    let one = |r: (usize, usize), want: u32| {
        let mut o = None;
        fields(d, r.0, r.1, |n, v| {
            if n == want && o.is_none() {
                o = v.range();
            }
        });
        o
    };
    let Some(r) = one((0, d.len()), 5).and_then(|r| one(r, 1)) else { return out };
    fields(d, r.0, r.1, |n, v| {
        let (101, Some((a, b))) = (n, v.range()) else { return };
        let Some(ev) = one((a, b), 2) else { return };
        fields(d, ev.0, ev.1, |n, v| {
            let (1, Some((a, b))) = (n, v.range()) else { return };
            // a car with no field 2 is number 0 (proto3 leaves 0 out)
            let mut x = 0u16;
            fields(d, a, b, |n, v| {
                if let (2, Val::Varint(v)) = (n, v) {
                    x = v as u16;
                }
            });
            if !out.contains(&x) {
                out.push(x);
            }
        });
    });
    out
}

fn run(addr: &str, user: &str, name: &str, fs: &[u8], mech: &str, sh: &Mutex<Shared>) -> Result<(), String> {
    let log = crate::envc::var_os("EVOVIEW_MP_LOG").is_some();
    let mut tcp = TcpStream::connect(addr).map_err(|e| format!("connect {addr}: {e}"))?;
    tcp.set_nodelay(true).ok();
    let udp = UdpSocket::bind("0.0.0.0:0").map_err(|e| e.to_string())?;
    udp.connect(addr).map_err(|e| e.to_string())?;
    udp.set_nonblocking(true).ok();
    let send = |tcp: &mut TcpStream, name: &str, msg: &[u8]| -> Result<(), String> {
        write_frame(tcp, &tcp_frame(&proto_packet(name, msg)))
    };
    // 1. handshake
    let mut m = Vec::new();
    f_varint(&mut m, 5, 8); // protocol_version
    f_varint(&mut m, 6, 6); // server_version
    f_bytes(&mut m, 7, user.as_bytes());
    f_bytes(&mut m, 9, mech.as_bytes());
    send(&mut tcp, "ClientConnectionRequest", &m)?;
    tcp.set_read_timeout(Some(Duration::from_millis(5))).ok();

    let mut buf: Vec<u8> = Vec::new();
    let mut rd = vec![0u8; 1 << 16];
    let (mut conn, mut car_id) = (0u32, (0u64, 0u64));
    let mut server_time: Option<(f64, Instant)> = None;
    let mut last_state = Instant::now();
    let mut last_send = Instant::now();
    let mut seq = 0u8;
    loop {
        // TCP in
        match tcp.read(&mut rd) {
            Ok(0) => return Err("server closed the connection".into()),
            Ok(n) => buf.extend_from_slice(&rd[..n]),
            Err(e) if !lost(&e) => {
                // ⚠ a timed-out read on Windows can come back as os error 997
                // ("overlapped I/O operation is in progress") rather than
                // TimedOut: it ended a live session on a home server
                if !matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) && log {
                    println!("mp: (passing) read: {e}");
                }
            }
            Err(e) => return Err(e.to_string()),
        }
        while buf.len() >= 2 {
            let (n, h) = match u16::from_le_bytes([buf[0], buf[1]]) {
                0xffff if buf.len() >= 6 => (u32::from_le_bytes([buf[2], buf[3], buf[4], buf[5]]) as usize - 2, 6),
                0xffff => break,
                n => (n as usize, 2),
            };
            if buf.len() < h + n {
                break;
            }
            let p: Vec<u8> = buf[h..h + n].to_vec();
            buf.drain(..h + n);
            if p.len() < 3 || u16::from_le_bytes([p[0], p[1]]) != 2 {
                continue;
            }
            let ln = p[2] as usize;
            let Some(tn) = p.get(3..3 + ln).map(|t| String::from_utf8_lossy(t).to_string()) else { continue };
            let d = &p[3 + ln..];
            match tn.as_str() {
                "ConnectToServerHandshakeResponse" => {
                    let mut ok = 0;
                    crate::proto::fields(d, 0, d.len(), |n, v| match (n, v) {
                        (1, crate::proto::Val::Varint(x)) => ok = x,
                        (4, crate::proto::Val::Varint(x)) => conn = x as u32,
                        _ => {}
                    });
                    if ok != 1 {
                        return Err(format!("handshake refused (result {ok})"));
                    }
                    let nums = session_cars(d);
                    if !nums.is_empty() {
                        sh.lock().unwrap().listed = nums;
                    }
                    if let Some(t) = session_track(d) {
                        sh.lock().unwrap().track = Some(t);
                    }
                    let cs = session_containers(d);
                    if !cs.is_empty() {
                        println!("mp: session layout: {} container(s)", cs.len());
                        sh.lock().unwrap().containers = cs;
                    }
                    if let Some(c) = session_clock(d) {
                        println!("mp: session clock starts {:02}:{:02} x{} ({}-{:02}-{:02})", (c.0 / 60.0) as u32, (c.0 % 60.0) as u32, c.1, c.2 .0, c.2 .1, c.2 .2);
                        sh.lock().unwrap().clock = Some(c);
                    }
                    let mut a = 9u16.to_le_bytes().to_vec();
                    a.extend_from_slice(&conn.to_le_bytes());
                    udp.send(&a).ok();
                    sh.lock().unwrap().status = format!("connected, connection {conn}");
                }
                "AssociateUDPSocketResponse" => {
                    // 3. the car: 1 spawn data { 5 car_final_state Any, 3 controller 2, 8 fuel }
                    let mut any = Vec::new();
                    f_bytes(&mut any, 1, b"type.googleapis.com/CarFinalStateData");
                    f_bytes(&mut any, 2, fs);
                    let mut spawn = Vec::new();
                    f_varint(&mut spawn, 3, 2);
                    f_bytes(&mut spawn, 5, &any);
                    f_f32(&mut spawn, 8, 60.0);
                    let mut drv = Vec::new();
                    let (first, last) = name.split_once(' ').unwrap_or((name, ""));
                    f_bytes(&mut drv, 1, first.as_bytes());
                    f_bytes(&mut drv, 2, last.as_bytes());
                    f_bytes(&mut drv, 3, name.as_bytes());
                    f_bytes(&mut drv, 4, user.as_bytes());
                    let mut rq = Vec::new();
                    f_bytes(&mut rq, 1, &spawn);
                    f_bytes(&mut rq, 2, &drv);
                    send(&mut tcp, "RequestPlayerCar", &rq)?;
                }
                "ChangePlayerCarResponseEvent" => {
                    let (mut pos, mut rot, mut number) = (Vec3::ZERO, Vec3::ZERO, 0u64);
                    let vec3 = |d: &[u8], a: usize, b: usize| {
                        let mut v = Vec3::ZERO;
                        crate::proto::fields(d, a, b, |n, x| match n {
                            1 => v.x = x.f32(),
                            2 => v.y = x.f32(),
                            3 => v.z = x.f32(),
                            _ => {}
                        });
                        v
                    };
                    crate::proto::fields(d, 0, d.len(), |n, v| match (n, v) {
                        (3, crate::proto::Val::Bytes(a, b)) => crate::proto::fields(d, a, b, |n, x| match (n, x) {
                            (1, crate::proto::Val::Varint(x)) => car_id.0 = x,
                            (2, crate::proto::Val::Varint(x)) => car_id.1 = x,
                            _ => {}
                        }),
                        (4, crate::proto::Val::Varint(x)) => number = x,
                        (5, crate::proto::Val::Bytes(a, b)) => crate::proto::fields(d, a, b, |n, x| match (n, x.range()) {
                            (1, Some((a, b))) => pos = vec3(d, a, b),
                            (2, Some((a, b))) => rot = vec3(d, a, b),
                            _ => {}
                        }),
                        _ => {}
                    });
                    // the slot our packets carry is the car's NUMBER (a public server's
                    // cars ran 12..215 with gaps); EVOVIEW_MP_SLOT overrides. Number 0
                    // (omitted, a local test server): 1, as its vAI car sends.
                    let slot = crate::envc::var("EVOVIEW_MP_SLOT").ok().and_then(|v| v.parse().ok())
                        .unwrap_or(number as u16);
                    {
                        let mut g = sh.lock().unwrap();
                        g.spawn = Some((pos, rot));
                        g.slot = slot;
                        g.status = format!("driving: car {} number {number} slot {slot}", car_id.1);
                    }
                    println!("mp: car {:?} number {number}, pit box ({:.1}, {:.1}, {:.1}) heading {:.0} deg", car_id, pos.x, pos.y, pos.z, rot.y);
                    let g = pguid(car_id.0, car_id.1);
                    let mut ev = Vec::new();
                    f_bytes(&mut ev, 1, &g);
                    send(&mut tcp, "CommandFromClient.ClientLoaded", &ev)?;
                    send(&mut tcp, "CommandFromUI.PlayerReady", &ev)?;
                }
                "ServerWorldTime" => {
                    crate::proto::fields(d, 0, d.len(), |n, v| {
                        if let (1, crate::proto::Val::F64(t)) = (n, v) {
                            server_time = Some((t, Instant::now()));
                        }
                    });
                    sh.lock().unwrap().world = server_time;
                }
                "ServerGameModeTime" => {
                    crate::proto::fields(d, 0, d.len(), |n, v| {
                        if let (1, crate::proto::Val::F64(t)) = (n, v) {
                            sh.lock().unwrap().mode_start = t;
                        }
                    });
                }
                // chat {1 kind, 2 timestamp, 3 sender (the server fills it when
                // relaying), 4 text}
                "MultiplayerChatMessage" => {
                    use crate::proto::{fields, text, Val};
                    let (mut kind, mut who, mut msg) = (0u32, String::new(), String::new());
                    fields(d, 0, d.len(), |n, v| match (n, v) {
                        (1, Val::Varint(x)) => kind = x as u32,
                        (3, Val::Bytes(a, b)) => who = text(d, (a, b)),
                        (4, Val::Bytes(a, b)) => msg = text(d, (a, b)),
                        _ => {}
                    });
                    println!("chat: {}{msg}", if who.is_empty() { String::new() } else { format!("{who}: ") });
                    let mut g = sh.lock().unwrap();
                    g.chat.push_back((Instant::now(), who, msg, kind));
                    while g.chat.len() > 12 {
                        g.chat.pop_front();
                    }
                }
                // another evoview's car (see EVOVIEW_TAG): as a netcar
                "UpdateGameCarState" => {
                    if let Some((n, car, who, fs)) = evoview_tag(d) {
                        let mut g = sh.lock().unwrap();
                        if n != g.slot && !g.roster.contains_key(&n) {
                            println!("mp: car {n}: {car} ({who}, evoview)");
                            g.roster.insert(n, (car, who));
                            g.states.insert(n, fs);
                        }
                    }
                }
                "CreateNetcarCommand" => {
                    if log {
                        println!("mp <- CreateNetcarCommand ({} bytes)", d.len());
                    }
                    if let Some((n, car, who, fs)) = netcar(d) {
                        if log {
                            println!("mp: car {n}: {car} ({who})");
                        }
                        let mut g = sh.lock().unwrap();
                        g.roster.insert(n, (car, who));
                        g.states.insert(n, fs);
                    }
                }
                // the server (re)numbers a car: ours -> our packets' slot
                "AssociateCarNumberEvent" => {
                    // no field 2 = number 0 (proto3)
                    let (mut id, mut num) = ((0u64, 0u64), Some(0u16));
                    crate::proto::fields(d, 0, d.len(), |n, v| match (n, v) {
                        (1, crate::proto::Val::Bytes(a, b)) => crate::proto::fields(d, a, b, |n, x| match (n, x) {
                            (1, crate::proto::Val::Varint(x)) => id.0 = x,
                            (2, crate::proto::Val::Varint(x)) => id.1 = x,
                            _ => {}
                        }),
                        (2, crate::proto::Val::Varint(x)) => num = Some(x as u16),
                        _ => {}
                    });
                    if let (true, Some(n)) = (id == car_id, num) {
                        if crate::envc::var_os("EVOVIEW_MP_SLOT").is_none() {
                            sh.lock().unwrap().slot = n;
                            println!("mp: our car number is now {n}");
                        }
                    }
                }
                other => {
                    if log {
                        println!("mp <- {other} ({} bytes)", d.len());
                    }
                }
            }
        }
        // UDP in
        let mut ub = [0u8; 2048];
        while let Ok(n) = udp.recv(&mut ub) {
            // the server's ping (3: u16, f64 server time, ...) wants a pong:
            // `u16 4, f64 that time, f64 ours` (server UDP case 4: its ping is
            // now - the echo) - a client that never answers is dropped after
            // about a minute ("disconnected due to software timeout")
            if n >= 10 && u16::from_le_bytes([ub[0], ub[1]]) == 3 {
                let mut q = 4u16.to_le_bytes().to_vec();
                q.extend_from_slice(&ub[2..10]);
                let ours = server_time.map_or(0.0, |(t, at)| t + at.elapsed().as_secs_f64());
                q.extend_from_slice(&ours.to_le_bytes());
                udp.send(&q).ok();
                continue;
            }
            if let Some((slot, st)) = parse_car(&ub[..n]) {
                let mut g = sh.lock().unwrap();
                if slot != g.slot {
                    g.remotes.insert(slot, Remote { state: st, at: Instant::now() });
                    let ms = i32::from_le_bytes([ub[3], ub[4], ub[5], ub[6]]);
                    let local = g.epoch.get_or_insert_with(Instant::now).elapsed().as_secs_f64();
                    g.hist.entry(slot).or_insert_with(|| Hist { buf: Default::default(), off: 0.0, play: None })
                        .push(local, ms as f64 / 1000.0, st);
                    // a car with no netcar yet: ask (UDP `u16 0xB, u32 number`;
                    // the server answers only clients driving a car, and only for
                    // players' cars - a local server's vAI car never answers)
                    if g.spawn.is_some() && !g.roster.contains_key(&slot)
                        && g.asked.get(&slot).map_or(true, |t| t.elapsed() > Duration::from_secs(2)) {
                        g.asked.insert(slot, Instant::now());
                        let mut q = 0xbu16.to_le_bytes().to_vec();
                        q.extend_from_slice(&(slot as u32).to_le_bytes());
                        udp.send(&q).ok();
                    }
                }
            }
        }
        // the session's cars we have no netcar for: ask (2 s apart each)
        {
            let mut g = sh.lock().unwrap();
            if g.spawn.is_some() {
                let want: Vec<u16> = g.listed.iter().copied()
                    .filter(|n| *n != g.slot && !g.roster.contains_key(n)
                        && g.asked.get(n).map_or(true, |t| t.elapsed() > Duration::from_secs(2)))
                    .collect();
                for n in want {
                    g.asked.insert(n, Instant::now());
                    let mut q = 0xbu16.to_le_bytes().to_vec();
                    q.extend_from_slice(&(n as u32).to_le_bytes());
                    udp.send(&q).ok();
                }
            }
        }
        // chat out: MultiplayerChatMessage {1 ChatNormal, 2 unix ms, 3 our name, 4 text}
        let said: Vec<String> = std::mem::take(&mut sh.lock().unwrap().say);
        for t in said {
            let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
            let mut c = Vec::new();
            f_varint(&mut c, 2, ms);
            f_bytes(&mut c, 3, name.as_bytes());
            f_bytes(&mut c, 4, t.as_bytes());
            send(&mut tcp, "MultiplayerChatMessage", &c)?;
        }
        // TCP out: who we are, for other evoviews. The server never answers a
        // netcar request (0xB) for a car that is not the game's - evoview cars
        // were invisible to each other - but it relays UpdateGameCarState to
        // every client as is, unknown fields included (the game ignores them):
        // {1 car_id, 99 EVOVIEW_TAG {1 "evoview", 2 number, 3 CarFinalStateData,
        // 4 driver name}}, every 3 s so late joiners hear it
        if car_id != (0, 0) && last_state.elapsed() >= Duration::from_secs(3)
            && crate::envc::var_os("EVOVIEW_MP_NOTAG").is_none() {
            last_state = Instant::now();
            let number = sh.lock().unwrap().slot;
            let mut tag = Vec::new();
            f_bytes(&mut tag, 1, b"evoview");
            f_varint(&mut tag, 2, number as u64);
            f_bytes(&mut tag, 3, fs);
            f_bytes(&mut tag, 4, name.as_bytes());
            let mut m = Vec::new();
            f_bytes(&mut m, 1, &pguid(car_id.0, car_id.1));
            f_bytes(&mut m, EVOVIEW_TAG, &tag);
            send(&mut tcp, "UpdateGameCarState", &m)?;
        }
        // UDP out: our car, 20 Hz
        if last_send.elapsed() >= Duration::from_millis(50) {
            last_send = Instant::now();
            let g = sh.lock().unwrap();
            // while evoview is still loading (no state from the frame loop yet):
            // standing in the pit box, or the server drops a silent client
            let me = g.me.or_else(|| g.spawn.map(|(p, r)| CarState {
                pos: p, ypr: Vec3::new(r.y.to_radians(), 0.0, 0.0), rpm: 900, gear: 1, ..Default::default()
            }));
            if let (Some(me), Some((t, at))) = (me, server_time) {
                if g.spawn.is_some() {
                    seq = seq.wrapping_add(1);
                    let ms = ((t + at.elapsed().as_secs_f64()) * 1000.0) as i32;
                    udp.send(&car_packet(seq, ms, g.slot, &me)).ok();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn car_packet_round_trip() {
        let s = CarState { pos: Vec3::new(1.0, 2.0, 3.0), ypr: Vec3::new(0.5, 0.0, 0.0), vel: Vec3::new(0.0, 0.0, 30.0),
                           rpm: 5000, gear: 3, throttle: 1.0, brake: 0.0, steer: 0.25, wheel: 88.0 };
        let p = car_packet(7, 1234, 2, &s);
        assert_eq!(p.len(), 63);
        let (slot, r) = parse_car(&p).unwrap();
        assert_eq!(slot, 2);
        assert_eq!(r.pos, s.pos);
        assert_eq!(r.rpm, 5000);
        assert!((r.steer - 0.25).abs() < 1e-3);
        assert!((r.wheel - 88.0).abs() / 88.0 < 0.15);
    }

    /// A car on a 60 m circle at 30 m/s, a packet every 57 ms (its own clock
    /// 4000 s off ours) arriving 20-80 ms late at random, one in ten lost:
    /// played back at 60 Hz it stays on the circle and never jumps.
    #[test]
    fn jitter_buffer_is_smooth() {
        let (r, v) = (60.0f32, 30.0f32);
        let at = |t: f32| {
            let a = t * v / r;
            (Vec3::new(r * a.cos(), 0.0, r * a.sin()), Vec3::new(-v * a.sin(), 0.0, v * a.cos()))
        };
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        // (arrival, stamp, state)
        let mut pk: Vec<(f64, f64, CarState)> = Vec::new();
        for i in 0..200 {
            let t = i as f32 * 0.057;
            if rnd() < 0.1 {
                continue;
            }
            let (p, vel) = at(t);
            pk.push((t as f64 + 0.02 + 0.06 * rnd() as f64, 4000.0 + t as f64, CarState { pos: p, vel, ..Default::default() }));
        }
        pk.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut h = Hist { buf: Default::default(), off: 0.0, play: None };
        let (mut k, mut prev, mut worst_off, mut worst_step) = (0usize, None::<Vec3>, 0.0f32, 0.0f32);
        let mut local = 1.0f64;
        // the old way for comparison: the newest packet carried on by its age
        let (mut newest, mut old_prev, mut old_step) = (None::<(f64, CarState)>, None::<Vec3>, 0.0f32);
        while local < 10.0 {
            while k < pk.len() && pk[k].0 <= local {
                h.push(pk[k].0, pk[k].1, pk[k].2);
                newest = Some((pk[k].0, pk[k].2));
                k += 1;
            }
            if let Some((t0, n)) = newest {
                let p = n.pos + n.vel * ((local - t0) as f32).min(0.3);
                if let Some(q) = old_prev {
                    old_step = old_step.max(((p - q).length() - 0.5).abs());
                }
                old_prev = Some(p);
            }
            let s = h.sample(local).unwrap();
            worst_off = worst_off.max((s.pos.length() - r).abs());
            if let Some(p) = prev {
                // 30 m/s at 60 Hz = 0.5 m a frame
                worst_step = worst_step.max(((s.pos - p).length() - 0.5).abs());
            }
            prev = Some(s.pos);
            local += 1.0 / 60.0;
        }
        eprintln!("jitter buffer: worst {worst_off:.3} m off the path, worst frame step error {worst_step:.3} m (old way: {old_step:.3} m)");
        assert!(worst_off < 0.05, "off the circle by {worst_off} m");
        assert!(worst_step < 0.1, "a frame's step off 0.5 m by {worst_step} m");
    }
}
