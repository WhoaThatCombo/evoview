//! evoview - an Assetto Corsa EVO car viewer on Vulkan (via wgpu).
//!
//!     evoview <car.evocar>                       interactive
//!     evoview <car.evocar> --shot out.png        render one frame, headless
//!         [--size WxH] [--yaw R] [--pitch R] [--zoom F] [--at X,Y,Z]
//!         [--anim NAME] [--time SECONDS]
//!     evoview <pkg.kspkg> --car <id> [--base content.kspkg]
//!     evoview <pkg.kspkg> --track <folder> [--base content.kspkg]
//!     evoview <track-folder> --track [--freeroam|--circuit]
//!     evoview --replay <file.rpy>                 summarise a replay
//!     evoview --replay <file.rpy> --pkg content.kspkg
//!         [--time SECONDS] [--car N] [--shot out.png]
//!         [--view chase|onboard|side|orbit]       play it back
//!     --fly   start in free-look (tracks do this anyway)
//!     evoview <src> --car <id> --drive | --demo        drive mode on start
//!     evoview <src> --car <id> --demo T --shot out.png [--frames N --fps F]
//!     evoview <src> --car <id> --check report.json [--sheet dir|png]
//!     evoview --replay-car <file.rpy> --pkg content.kspkg [--car N]
//!         [--time T --shot out.png]   one recorded car drives the rig
//!
//! CONTROLS
//!   orbit mode (default)  drag to orbit, wheel to zoom
//!   F                     toggle free-look; press again to re-frame the car
//!   free-look             WASD move, Q/E (or Ctrl/Space) down/up,
//!                         hold either mouse button to look,
//!                         Shift fast, Alt slow, wheel sets fly speed
//!   [ / ]                 step through the car's animations
//!   P                     pause/resume playback, 0 returns to the rest pose
//!   replay                P play/pause, , . scrub, - = speed, R restart,
//!                         C camera (free/chase/onboard/side/orbit),
//!                         [ ] previous/next car, M mute the engine (the
//!                         onboard camera switches to the interior mix).
//!                         The overlay's buttons do the same with the mouse,
//!                         and the timeline can be clicked to seek. The window
//!                         title carries the playhead, car and speed.
//!   G                     toggle the studio floor (it also hides itself
//!                         automatically when the camera drops below it)
//!   L                     track lighting on/off (sun + shade vs flat albedo)
//!   H                     light switch: off / stage 1 (DRL) / 2 (low) / 3 (high)
//!                         of the car's .carlightingsystem (EVOVIEW_LIGHTS=0..3)
//!   B                     show/hide the seated driver (EVOVIEW_DRIVER=0 skips
//!                         loading it; =racing / =street force the outfit)
//!   N                     drive mode: off / keyboard / scripted demo (/ replay
//!                         with --replay-car). Driving: Up/I throttle, Down/K
//!                         brake, Left/J Right/L steer, Space handbrake, X/Z
//!                         gear up/down, Y wipers, T DRS, 1-9 doors / lids
//!                         (carrig.rs binds the clips, drive.rs is the model).
//!                         The effects follow the car state while driving
//!                         (fxstate.rs follow_car): brake lamps, dash, shift
//!                         LEDs, disc heat, backfire on lift-off, wheelspin smoke.
//!   effects (F-keys)      F1 key list, F2 night, F3 bloom, F4 dirt, F5 wet,
//!                         F6 disc temperature, F7 backfire, F8 tyre smoke,
//!                         F9 dust, F10 indicators, F11 rain/fog light,
//!                         F12 hot exhaust
//!   manual (not driving)  PageUp/PageDown rpm, Shift+PageUp/PageDown gear,
//!                         Home/End speed, Insert brake, Delete reverse
//!                         (render.rs KEYS is the full list, printed at start)
//!   Esc                   quit
//!
//! A car folder (content/cars/<id>) is read directly - meshes, textures,
//! materials, liveries, tyres and animations - so no per-car export step is
//! needed. tools/export_car.py still produces .evocar files, which load
//! instantly and are handy for regression shots.

mod asset;
mod race;
mod cluster;
mod motion;
mod dlss;
mod fsr;
mod install;
mod bench;
mod perf;
mod audio;
mod carsound;
mod load;
mod parts;
mod numbers;
mod rain;
mod rainfall;
mod pkg;
mod proto;
mod camera;
mod cull;
mod camfile;
mod lights;
mod livery;
mod contact;
mod material;
mod mipfill;
mod layers;
mod fmodsys;
mod hud;
mod grid;
mod grade;
mod render;
mod replay;
mod replayscene;
mod shot;
mod track;
mod mirrors;
mod probe;
mod driver;
mod carstate;
mod carphys;
mod carrig;
mod drive;
mod fxstate;
mod fxdata;
mod dash;
mod fxgpu;
mod particles;
mod check;
mod check_anim;
mod check_fx;
mod strict;
mod sky;
mod taa;
mod ssr;
mod post;
mod cams;
mod headcam;
mod clouds;
mod trackdrive;
mod aids;
mod pits;
mod laps;
mod lapsim;
mod sctm;
mod tyretherm;
mod ffb;
mod wheel;
mod evophys;
mod rigcar;
mod rigode;
mod rigwish;
mod groundode;
mod collide;
mod mp;
mod ode;

use std::path::PathBuf;

/// Printed for --help. Kept next to main so it cannot drift from the flags.
const HELP: &str = "evoview - an Assetto Corsa EVO car and track viewer (Vulkan, via wgpu)

USAGE
  evoview <car-folder>                    open a car, interactive
  evoview <car-folder> --shot out.png     render one frame, headless
  evoview <content.kspkg> --car <id>      read a car straight out of an archive
  evoview <track-folder> --track          open a track
  evoview <src> --materials               every material as a sphere, on a grid

MATERIAL SHEET
  --materials       replace the scene with one sphere per material, laid out
                    alphabetically and lit by the SAME shader path the real
                    content uses. Prints a legend whose row/column match the
                    picture, flagging fallback (F), alpha (A), cutout (C) and
                    cabin (I) materials. The fastest way to find the one
                    material that is wrong out of ninety.
  --materials-only <a,b,c>
                    narrow the sheet to materials whose name contains any of
                    these - a sheet of just the suspects.
  --grid-radius R   sphere radius in metres (default 1.5). Matters for
                    world-tiled and triplanar materials, which scale by world
                    size - a small ball shows a fraction of a tile.

VIEW
  --size WxH        output size for --shot          (default 1280x720)
  --yaw R --pitch R camera angles, radians
  --zoom F          higher is closer
  --anim NAME --time SECONDS
  --fly             start in free-look

CONTROLS
  drag orbit, wheel zoom, F free-look (WASD, Q/E, Shift fast, Alt slow)
  [ ] step animations, P pause, 0 rest pose, B driver on/off
  V / Shift+V       next / previous livery (title shows name and n/N)

DRIVE (N cycles off / drive / demo; --drive or --demo start in it)
  Up/I throttle, Down/K brake, Left/J Right/L steer, Space handbrake
  X / Z gear up / down, Y wipers, T DRS, 1-9 doors / hood / boot / roof
  H lights (pop-ups follow low beam). The car runs on a treadmill.
  --layout <id>     the track layout (layout_<id>: scenery, racing line and
                    spawn agree; nurburgring: nordschleife, gp_strecke, 24h,
                    sprint ...). Same as EVOVIEW_LAYOUT=<id>.
  --demo T --shot out.png        the scripted demo at T seconds, headless
      [--frames N --fps F]       a numbered frame sequence from T
  --check report.json [--sheet <dir | sheet.png>]
                    one report: animation (steer, travel, pedals, gears, every
                    clip, needles) + effects (every light function, LEDs, dash,
                    glow, exhaust, textures, glass); records {car, check,
                    severity, detail}; the sheet tiles key poses + effect renders

SETUPS (the car loads its stock setup, as the game does)
  --setup <file | name>   drive with one of your saved setups on top of it:
                    a .carsetup path, or a name under Saved Games/ACE/Car
                    Setups (\"lagua\", \"laguna seca/lagua\"). Prints what changed.
                    Same as EVOVIEW_SETUP=<file | name>.
  --list-setups [car]     your saved setups (path, car folder, track, name,
                    the car id it was saved for), tab-separated; then exit
  EVOVIEW_SETUP_FORCE=1   apply a setup saved for another car
  EVOVIEW_SETUP_TABLE=1   print every setup value, not just the changed ones
  EVOVIEW_NO_SETUP=1      neither: the bare car data

LIVERIES
  --list-liveries   print  stem<TAB>display name<TAB>race number  per visual
                    preset (default first) and exit
  stdin             a running viewer reads commands, one per line:
                      livery <stem | n | next | prev>     quit

SHARED ASSETS
  Tyres, paints and the studio environment come from YOUR install of the game.
  evoview ships no game content. It finds content.kspkg via Steam; override with
    EVOVIEW_KSPKG=<path to content.kspkg>

ENVIRONMENT
  EVOVIEW_KSPKG     path to content.kspkg (file or install folder)
  EVOVIEW_ASSETS    extra asset roots, ';' separated
  EVOVIEW_VISUAL    which visual preset (trim / livery) to wear
  EVOVIEW_PAINT     override the body colour
  EVOVIEW_DRIVER=0  no driver in the seat (=racing / =street force the outfit)
  EVOVIEW_MAX_FPS=N frame cap (default 60; 0 = uncapped, which --bench needs
                    for real numbers - tools/bench.py sets it)
  EVOVIEW_NOVSYNC=1 no vsync (still capped by EVOVIEW_MAX_FPS)
  EVOVIEW_TEX_LOD=N track texture detail: 0 full, 1 half (default: auto, full
                    resolution below 4K and half above)
  EVOVIEW_MAT_LOG=1 dump every material: albedo, resolved base colour, params
  EVOVIEW_MAT_HL=<s> paint materials whose name contains <s> magenta

EFFECTS (F1 lists every key; env vars for --shot, see fxstate.rs)
  F2 night  F3 bloom  F4 dirt  F5 wet  F6 disc temp  F7 backfire  F8 tyre smoke
  F9 dust  F10 indicators  F11 rain/fog light  F12 hot exhaust
  not driving: PageUp/Down rpm (Shift: gear), Home/End speed, Insert brake,
  Delete reverse. Driving (N): the car state drives all of these.
  EVOVIEW_LIGHTS EVOVIEW_BRAKE EVOVIEW_INDICATOR EVOVIEW_FOG EVOVIEW_GEAR
  EVOVIEW_RPM EVOVIEW_SPEED EVOVIEW_THROTTLE EVOVIEW_DISC_TEMP EVOVIEW_EXHAUST_TEMP
  EVOVIEW_DIRT EVOVIEW_WET EVOVIEW_NIGHT EVOVIEW_BACKFIRE EVOVIEW_BACKFIRE_TYPE
  EVOVIEW_SMOKE EVOVIEW_DUST EVOVIEW_BLOOM EVOVIEW_BLINK EVOVIEW_DASH_DUMP=<dir>

  --version         print the version
";


/// Process start, for the headless first-frame timing (EVOVIEW_BENCH).
pub static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

fn main() {
    T0.get_or_init(std::time::Instant::now);
    perf::mark_start();
    // EVOVIEW_UPSCALER=dlss | fsr: its quality mode sets the render scale
    dlss::apply_render_scale();
    fsr::apply_render_scale();
    let args: Vec<String> = std::env::args().collect();
    // --wheel-setup: map the wheel, pedals, paddles and H shifter (wheel.rs)
    if args.iter().any(|a| a == "--wheel-setup") {
        let i = args.iter().position(|a| a == "--wheel-setup").unwrap_or(0);
        let only: Vec<String> = args[i + 1..].iter().take_while(|a| !a.starts_with("--")).cloned().collect();
        return wheel::setup_wizard(&only);
    }
    // --tex-audit <content.kspkg | mod .kspkg | folder of loose files>:
    // every texture through the renderer's decoder (load::tex_audit)
    if let Some(i) = args.iter().position(|a| a == "--mat-audit") {
        let target = args.get(i + 1).filter(|a| !a.starts_with("--")).or_else(|| args.get(1)).cloned().unwrap_or_default();
        return load::mat_audit(&PathBuf::from(target), track::KNOWN_SHADERS, material::CAR_SHADERS);
    }
    if let Some(i) = args.iter().position(|a| a == "--tex-audit") {
        let target = args.get(i + 1).filter(|a| !a.starts_with("--")).or_else(|| args.get(1)).cloned().unwrap_or_default();
        return load::tex_audit(&PathBuf::from(target));
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("evoview {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print!("{}", HELP);
        return;
    }
    let flag = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    // --list-setups [car]: the player's saved setups (carphys.rs)
    if let Some(i) = args.iter().position(|a| a == "--list-setups") {
        return carphys::list_setups(args.get(i + 1).filter(|a| !a.starts_with("--")).map(|s| s.as_str()));
    }
    // --setup <file | name>: drive with one of the player's setups; the same
    // as EVOVIEW_SETUP, which every Physics::read sees (set before threads)
    // --layout <id>: the track layout (scenery, racing line, spawn); the
    // same as EVOVIEW_LAYOUT, which the loader and trackdrive read
    if let Some(l) = flag("--layout") {
        std::env::set_var("EVOVIEW_LAYOUT", l);
    }
    if let Some(s) = flag("--setup") {
        std::env::set_var("EVOVIEW_SETUP", s);
    }
    // --strict: draw what had to be guessed (strict.rs)
    if args.iter().any(|a| a == "--strict") {
        strict::set_on(true);
    }

    let path: PathBuf = args
        .get(1)
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "supra.evocar".to_string())
        .into();

    // Probe the game's FMOD: load the banks for one car and report what events
    // and parameters they actually expose.
    // `--fmodprobe-all <model> --pkg content.kspkg`: every event of the car
    // bank and the shared banks, with its declared parameters
    // `--sound-test <model> --pkg content.kspkg [--wav out.wav]`: a scripted
    // drive through the car's sound, rendered offline (carsound.rs)
    if let Some(model) = flag("--sound-test") {
        let pkg = PathBuf::from(flag("--pkg").expect("--sound-test needs --pkg content.kspkg"));
        let out = PathBuf::from(flag("--wav").unwrap_or_else(|| format!("{model}_sound.wav")));
        return carsound::sound_test(&pkg, &model, &out);
    }
    if let Some(model) = flag("--fmodprobe-all") {
        let pkg = PathBuf::from(flag("--pkg").expect("--fmodprobe-all needs --pkg content.kspkg"));
        let dir = pkg.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
        let mut p = pkg::Package::open(&pkg).expect("package");
        let mut grab = |name: &str| -> Vec<u8> {
            p.by_path.get(&format!("content\\sfx\\{name}")).copied().map(|e| p.read(e)).unwrap_or_default()
        };
        let master = grab("master_bank.bank");
        let strings = grab("master_bank.strings.bank");
        let car = grab(&format!("{}.bank", model.to_ascii_lowercase()));
        let mut f = fmodsys::Fmod::new(&dir).expect("FMOD");
        let mut all: Vec<(String, Vec<String>)> = vec![(model.clone(), f.load_banks(&master, &strings, &car).expect("banks"))];
        for b in ["wheel", "wheel_hub", "turbo", "transmission", "gearbox", "brakes", "backfire", "bodywork",
                  "damage", "environment", "horn", "indicators", "effect_chain"] {
            let d = grab(&format!("{b}.bank"));
            match f.load_extra(&d) {
                Ok(ev) => all.push((b.to_string(), ev)),
                Err(e) => println!("{b}: {e}"),
            }
        }
        for (b, evs) in all {
            println!("== {b}: {} event(s)", evs.len());
            for e in evs {
                match f.event_info(&e) {
                    Some((ps, d3)) => println!("   {e}{}  {}", if d3 { " [3D]" } else { "" },
                        ps.iter().map(|(n, lo, hi, d)| format!("{n}({lo}..{hi} ={d})")).collect::<Vec<_>>().join(" ")),
                    None => println!("   {e}  (no description)"),
                }
            }
        }
        return;
    }
    if let Some(model) = flag("--fmodprobe") {
        let pkg = flag("--pkg").map(PathBuf::from)
            .expect("--fmodprobe needs --pkg content.kspkg");
        return fmod_probe(&pkg, &model);
    }

    // `--replay-car <file.rpy> --pkg content.kspkg [--car N]`: one recorded car
    // in the studio, its steering, pedals, gears, wheel spin and suspension
    // driven by the recording (drive.rs ReplayFeed)
    if let Some(rp) = flag("--replay-car") {
        let pkg = PathBuf::from(flag("--pkg").expect("--replay-car needs --pkg content.kspkg"));
        let base = flag("--base").map(PathBuf::from);
        return replay_car(&PathBuf::from(rp), &pkg, base.as_deref(), &flag);
    }
    // `--replay <file>` alone summarises a .rpy; add `--pkg content.kspkg` to
    // load the track and cars it names and actually play it back.
    if let Some(rp) = flag("--replay") {
        let rp = PathBuf::from(rp);
        return match flag("--pkg") {
            Some(pkg) => play_replay(&rp, &PathBuf::from(pkg), flag("--base").map(PathBuf::from)),
            None => dump_replay(&rp),
        };
    }

    // Shared assets (tyres, paints) that live outside a car folder. Defaults
    // to ./assets next to the viewer; override with EVOVIEW_ASSETS.
    let asset_roots: Vec<PathBuf> = std::env::var("EVOVIEW_ASSETS")
        .map(|v| v.split(';').map(PathBuf::from).collect())
        .unwrap_or_else(|_| Vec::new());

    // A car folder is loaded directly - no per-car export step. The .evocar
    // path is kept because it loads instantly and is useful for regression
    // shots.
    let t0 = std::time::Instant::now();
    // A .kspkg is read IN PLACE: `evoview content.kspkg --car ks_toyota_ae86...`
    // extracts nothing to disk. Without --car it just lists what is inside.
    if path.extension().map(|e| e == "kspkg").unwrap_or(false) {
        let base = flag("--base").map(PathBuf::from);
        // `--track <id> --drive-car <car>`: drive a car on the track
        // (trackdrive.rs). --drive-test SECONDS runs a scripted drive headless
        // and, with --shot, saves a chase-camera still at its end.
        // `--donut <track> [--car-id <car>] [--cars N]`: N cars drifting one
        // big shared donut, played back like a replay (EVOVIEW_DONUT="x,z,r"
        // moves it; default: the start of the racing line)
        if let Some(track_id) = flag("--donut") {
            return play_donut(&path, base, &track_id, &flag);
        }
        // `--convoy <track> --layout <l> --models a,b,c --cars N --at x,z
        // [--lead m] [--speed m/s] [--gap m]`: a pack of cars driving the
        // racing line two abreast, reaching the point --at after --lead m
        // `--lapsim <track> --car <id> [--layout <l>]`: an ideal lap from
        // the car's physics data (lapsim.rs)
        // --pkg-cat <entry> <out file>: one archive entry, decoded, to a file
        if let Some(i) = args.iter().position(|a| a == "--pkg-cat") {
            let key = args[i + 1].to_lowercase().replace('/', "\\");
            if let Ok(p) = pkg::Package::open_filtered(&path, &[key.as_str()]) {
                if let Some(e) = p.by_path.get(&key) {
                    let _ = std::fs::write(&args[i + 2], p.read(*e));
                    println!("wrote {}", args[i + 2]);
                } else {
                    println!("{key}: not in the archive");
                }
            }
            return;
        }
        // --pkg-grep <entry prefix> <text>: which archive entries contain a string
        if let Some(i) = args.iter().position(|a| a == "--pkg-grep") {
            let (pre, needle) = (args[i + 1].to_lowercase(), args[i + 2].as_bytes().to_vec());
            if let Ok(p) = pkg::Package::open_filtered(&path, &[pre.as_str()]) {
                let mut keys: Vec<_> = p.by_path.keys().cloned().collect();
                keys.sort();
                for k in keys {
                    let d = p.read(p.by_path[&k]);
                    let hits = d.windows(needle.len()).filter(|w| *w == &needle[..]).count();
                    if hits > 0 {
                        println!("PKGGREP {k} {hits}");
                    }
                }
            }
            return;
        }
        if args.iter().any(|a| a == "--sctm-dump") {
            return sctm::dump(&path);
        }
        if args.iter().any(|a| a == "--gearbox-table") {
            return rigcar::fleet_table(&path, base.as_deref());
        }
        if let (Some(track_id), Some(car_id)) = (flag("--lapsim"), flag("--car")) {
            lapsim::run(&path, base.as_deref(), &track_id, flag("--layout").as_deref(), &car_id);
            return;
        }
        if let Some(track_id) = flag("--convoy") {
            return play_convoy(&path, base, &track_id, &flag);
        }
        if let (Some(track_id), Some(car_id)) = (flag("--track"), flag("--drive-car")) {
            return drive_on_track(&path, base.as_deref(), &track_id, &car_id, &flag);
        }
        if let Some(track_id) = flag("--track") {
            let world = if args.iter().any(|a| a == "--circuit") {
                Some(false)
            } else if args.iter().any(|a| a == "--freeroam") {
                Some(true)
            } else {
                None
            };
            let l = match load::load_track_from_package_opts(
                &path, &track_id, base.as_deref(), world,
            ) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("could not load track {track_id}: {e}");
                    std::process::exit(1);
                }
            };
            println!(
                "{track_id}: {} mesh(es), {} texture(s), {} material(s) in {:.2}s",
                l.car.meshes.len(), l.car.textures.len(), l.car.materials.len(),
                t0.elapsed().as_secs_f32());
            return run_or_shot(
                l.car, Some(l.skeleton), l.animations, &flag,
                render::ViewOpts {
                    start_fly: true,
                    ground: false,
                    title: format!("evoview - {track_id}"),
                    ..Default::default()
                },
            );
        }
        if let Some(car_id) = flag("--car") {
            let mut sw = match livery::Switcher::from_package(&path, &car_id, base.as_deref()) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("could not load {car_id}: {e}");
                    std::process::exit(1);
                }
            };
            if args.iter().any(|a| a == "--list-liveries") {
                return livery::print_list(&sw.list);
            }
            let interactive = flag("--shot").is_none() && flag("--bench").is_none() && flag("--check").is_none();
            if interactive {
                sw.keep_textures();
            }
            let l = match sw.load() {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("could not load {car_id}: {e}");
                    std::process::exit(1);
                }
            };
            println!(
                "{car_id}: {} mesh(es), {} texture(s), {} material(s) in {:.2}s (no extraction)",
                l.car.meshes.len(), l.car.textures.len(), l.car.materials.len(),
                t0.elapsed().as_secs_f32());
            if args.iter().any(|a| a == "--load-only") {
                return;
            }
            let opts = render::ViewOpts {
                title: format!("evoview - {car_id}"),
                livery: interactive.then_some(sw),
                ..Default::default()
            };
            return run_or_shot(l.car, Some(l.skeleton), l.animations, &flag, opts);
        }
        match (load::list_package(&path), load::list_tracks(&path)) {
            (Ok(cars), Ok(tracks)) => {
                println!("{} car(s), {} track(s) in {}:", cars.len(), tracks.len(), path.display());
                for c in &cars { println!("  car    {c}"); }
                for t in &tracks { println!("  track  {t}"); }
            }
            (Err(e), _) => eprintln!("{e}"),
            _ => {}
        }
        return;
    }

    if path.is_dir() && args.iter().any(|a| a == "--track") {
        let base = flag("--base").map(PathBuf::from);
        let world = if args.iter().any(|a| a == "--circuit") {
            Some(false)
        } else if args.iter().any(|a| a == "--freeroam") {
            Some(true)
        } else {
            None
        };
        let l = match load::load_track_dir_opts(&path, base.as_deref(), world) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("could not load track folder {}: {e}", path.display());
                std::process::exit(1);
            }
        };
        println!(
            "{}: {} mesh(es), {} texture(s), {} material(s) in {:.2}s",
            path.display(), l.car.meshes.len(), l.car.textures.len(),
            l.car.materials.len(), t0.elapsed().as_secs_f32());
        return run_or_shot(
            l.car, Some(l.skeleton), l.animations, &flag,
            render::ViewOpts {
                start_fly: true,
                ground: false,
                title: format!("evoview - {}", path.file_name().and_then(|s| s.to_str()).unwrap_or("track")),
                ..Default::default()
            },
        );
    }

    let interactive = flag("--shot").is_none() && flag("--bench").is_none() && flag("--check").is_none();
    let mut switcher = None;
    let (car, skeleton, animations) = if path.is_dir() {
        let loaded = livery::Switcher::from_dir(&path, &asset_roots).and_then(|mut sw| {
            if args.iter().any(|a| a == "--list-liveries") {
                livery::print_list(&sw.list);
                std::process::exit(0);
            }
            if interactive {
                sw.keep_textures();
            }
            let l = sw.load()?;
            switcher = interactive.then_some(sw);
            Ok(l)
        });
        match loaded {
            Ok(l) => (l.car, Some(l.skeleton), l.animations),
            Err(e) => {
                eprintln!("could not load car folder {}: {e}", path.display());
                std::process::exit(1);
            }
        }
    } else {
        match asset::Car::load(&path) {
            Ok(c) => (c, None, Vec::new()),
            Err(e) => {
                eprintln!("could not load {}: {e}", path.display());
                eprintln!("pass a car folder (content/cars/<id>) or an .evocar");
                std::process::exit(1);
            }
        }
    };
    println!(
        "{}: {} mesh(es), {} texture(s), {} material(s) in {:.2}s",
        path.display(),
        car.meshes.len(),
        car.textures.len(),
        car.materials.len(),
        t0.elapsed().as_secs_f32()
    );
    if !animations.is_empty() {
        println!(
            "animations: {}",
            animations
                .iter()
                .map(|a| format!("{} ({}f)", a.name, a.frames))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let title = format!("evoview - {}",
        path.file_name().and_then(|s| s.to_str()).unwrap_or("car"));
    run_or_shot(car, skeleton, animations, &flag,
                render::ViewOpts { title, livery: switcher, ..Default::default() });
}

fn run_or_shot(
    car: asset::Car,
    skeleton: Option<load::Skeleton>,
    animations: Vec<load::Animation>,
    flag: &dyn Fn(&str) -> Option<String>,
    mut opts: render::ViewOpts,
) {
    // everything before the renderer starts: archive, decode, assembly
    perf::add("content load (wall)", perf::since_start());
    if args_has_fly() {
        opts.start_fly = true;
    }
    println!(
        "content: tyres={} rims={} anims={} bones={}",
        car.meshes.iter().filter(|m| m.name.starts_with("tyre_")).count(),
        car.meshes.iter().filter(|m| m.name.starts_with("rim_")).count(),
        animations.len(),
        skeleton.as_ref().map(|s| s.bones.len()).unwrap_or(0)
    );
    // --bench <seconds>: fixed orbit through the real renderer, then stats.
    if let Some(secs) = flag("--bench").and_then(|v| v.parse::<f32>().ok()) {
        let (width, height) = flag("--size")
            .and_then(|s| {
                let (a, b) = s.split_once('x')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .unwrap_or((1600, 900));
        let name = flag("--car")
            .or_else(|| flag("--track"))
            .or_else(|| std::env::args().nth(1))
            .unwrap_or_default();
        let shot = flag("--bench-shot").map(PathBuf::from);
        opts.bench = Some(bench::Opts { seconds: secs, width, height, name, shot });
    }
    // --check report.json: sweep every input and report what looks broken
    // a track's --check: one headless frame builds every material, then the
    // guesses are the report (strict.rs)
    let track_check = flag("--check").filter(|_| car.track_env.is_some());
    if let Some(out) = flag("--check").filter(|_| car.track_env.is_none()) {
        let name = flag("--car").or_else(|| std::env::args().nth(1)).unwrap_or_default();
        // --sheet <dir | file.png> (look-anim's spelling) or --check-sheet
        let sheet = flag("--sheet").or_else(|| flag("--check-sheet")).map(PathBuf::from);
        check::run(car, skeleton, &name, &PathBuf::from(out), sheet.as_deref());
        return;
    }
    let shot_out = flag("--shot").or_else(|| track_check.as_ref()
        .map(|_| std::env::temp_dir().join("evoview_track_check.png").to_string_lossy().into_owned()));
    if let Some(out) = shot_out {
        let (w, h) = flag("--size")
            .and_then(|s| {
                let (a, b) = s.split_once('x')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .unwrap_or((1600, 1000));
        // A contact sheet wants to be seen square on; an orbit angle wastes
        // most of the frame on foreshortening and hides the far columns.
        let (dy, dp) = if crate::grid::wanted() {
            (std::f32::consts::FRAC_PI_2, 0.0f32)
        } else {
            (2.5f32, 0.35f32)
        };
        let yaw = flag("--yaw").and_then(|v| v.parse().ok()).unwrap_or(dy);
        let pitch = flag("--pitch").and_then(|v| v.parse().ok()).unwrap_or(dp);
        // Zoom and a look-at point, so a detail can be inspected without
        // cropping a wide render down to a handful of pixels.
        // The framing margin is sized for a car with a wing and a shadow; a
        // flat sheet just wants to fill the frame.
        let dz = if crate::grid::wanted() { 1.7f32 } else { 1.0f32 };
        let zoom = flag("--zoom").and_then(|v| v.parse().ok()).unwrap_or(dz);
        let at = flag("--at").and_then(|s| {
            let v: Vec<f32> = s.split(',').filter_map(|x| x.parse().ok()).collect();
            (v.len() == 3).then(|| glam::Vec3::new(v[0], v[1], v[2]))
        });
        // --anim <substring> --time <seconds> renders a posed frame
        let pose = flag("--anim").and_then(|want| {
            let skel = skeleton.as_ref()?;
            let a = animations
                .iter()
                .find(|a| a.name.to_lowercase().contains(&want.to_lowercase()))?;
            let t = flag("--time").and_then(|v| v.parse().ok()).unwrap_or(0.5f32);
            println!("posing {} at {t}s ({} frames @ {} fps)", a.name, a.frames, a.fps);
            Some(skel.pose(Some(a), t))
        });
        // --cam: an absolute car-space camera (see camfile.rs) instead of the orbit
        let view = flag("--cam").map(|spec| camfile::view(&spec, w as f32 / h as f32)
            .unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(2) }));
        // --demo T: the scripted drive at T seconds; --frames N --fps F turns
        // it into a numbered sequence (out-0000.png ...) from one upload
        let rig = skeleton.as_ref().and_then(|s| s.rig.as_deref());
        if let (Some(t), Some(rig)) = (flag("--demo").and_then(|v| v.parse::<f32>().ok()), rig) {
            let n: usize = flag("--frames").and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
            let fps: f32 = flag("--fps").and_then(|v| v.parse().ok()).unwrap_or(30.0);
            let mut d = drive::Drive::new(rig);
            // the effects follow the car (fxstate.rs follow_car): pre-roll
            // in small steps so disc heat and blinkers have their history
            let mut fx = fxstate::FxState::from_env();
            let mut fxs = Vec::with_capacity(n);
            let mut now = 0.0f32;
            while now < t - 1e-4 {
                let step = (1.0 / 30.0f32).min(t - now);
                drive::demo_run(&mut d, rig, now, now + step);
                now += step;
                fx.time = now;
                fx.follow_car(&d.cs, step);
            }
            let outp = PathBuf::from(&out);
            let mut frames = Vec::with_capacity(n);
            for i in 0..n {
                if i > 0 {
                    let now = t + (i - 1) as f32 / fps;
                    drive::demo_run(&mut d, rig, now, now + 1.0 / fps);
                    fx.time = now + 1.0 / fps;
                    fx.follow_car(&d.cs, 1.0 / fps);
                }
                fxs.push(Some(fx));
                let inp = d.inputs(rig);
                let path = if n == 1 {
                    outp.clone()
                } else {
                    let stem = outp.file_stem().and_then(|s| s.to_str()).unwrap_or("frame");
                    outp.with_file_name(format!("{stem}-{i:04}.png"))
                };
                if i == 0 {
                    let (pd, rd) = d.attitude_deg();
                    println!("demo t={t:.2}s: {:.0} km/h, gear {}, {:.0} rpm, steer {:+.2}, thr {:.2} brk {:.2}, pitch {pd:+.2} roll {rd:+.2} deg",
                             d.cs.speed_mps * 3.6, d.cs.gear, d.cs.rpm, d.cs.steer, d.cs.throttle, d.cs.brake);
                }
                frames.push((Some(rig.pose(&inp)), path));
            }
            pollster::block_on(shot::shot_frames_fx(
                car, frames, fxs, Some(1.0 / fps), w, h, yaw, pitch, zoom, at, opts.start_fly, view));
            return;
        }
        pollster::block_on(shot::shot(
            car, &PathBuf::from(out), w, h, yaw, pitch, zoom, at, pose,
            opts.start_fly, view));
        if let Some(rep) = &track_check {
            let name = flag("--track").or_else(|| std::env::args().nth(1)).unwrap_or_default();
            strict::write_report(std::path::Path::new(rep), &name);
        }
    } else {
        opts.drive = if flag("--demo").is_some() || std::env::args().any(|a| a == "--demo") {
            2
        } else if std::env::args().any(|a| a == "--drive") {
            1
        } else {
            0
        };
        pollster::block_on(render::run(car, skeleton, animations, opts));
    }
}

fn replay_car(rp: &PathBuf, pkg: &PathBuf, base: Option<&std::path::Path>,
              flag: &dyn Fn(&str) -> Option<String>) {
    let rep = match replay::Replay::load(rp) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("could not read replay {}: {e}", rp.display());
            std::process::exit(1);
        }
    };
    let car = flag("--car").and_then(|v| v.parse::<usize>().ok()).unwrap_or_else(|| {
        (0..rep.cars.len()).find(|&i| !rep.is_placeholder(i) && rep.cars[i].model.is_some()).unwrap_or(0)
    });
    let model = rep.cars.get(car).and_then(|c| c.model.clone()).unwrap_or_else(|| rep.model.clone());
    println!("replay car {car}: {model}, {:.0}s", rep.cars.get(car).map_or(0.0, |c| c.n_matrix as f32 / rep.rate()));
    let l = match load::load_from_package(pkg, &model, base) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("could not load {model}: {e}");
            std::process::exit(1);
        }
    };
    if std::env::var_os("EVOVIEW_REPLAY_DUMP").is_some() {
        let span = rep.cars.get(car).map_or(0.0, |c| c.n_matrix as f32 / rep.rate());
        let mut t = 0.0;
        while t < span {
            if let Some(tel) = rep.telemetry(car, t) {
                println!("t {t:6.1} v {:6.1} km/h wheel {:6.1} km/h stamp {:7.2} rpm {:6.0} gear {} steer {:+7.1} thr {:.2} brk {:.2}",
                         rep.speed(car, t) * 3.6, tel.wheel.iter().sum::<f32>() / 4.0 * 0.33 * 3.6, tel.time,
                         tel.rpm, tel.gear, tel.steer_deg, tel.throttle, tel.brake);
            }
            t += 2.0;
        }
        return;
    }
    let mut feed = drive::ReplayFeed::new(rep, car);
    let Some(rig) = l.skeleton.rig.as_deref() else {
        eprintln!("{model}: no rig");
        std::process::exit(1);
    };
    if let Some(out) = flag("--shot") {
        let t: f32 = flag("--time").and_then(|v| v.parse().ok()).unwrap_or(10.0);
        let n: usize = flag("--frames").and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
        let fps: f32 = flag("--fps").and_then(|v| v.parse().ok()).unwrap_or(30.0);
        let (w, h) = flag("--size")
            .and_then(|s| {
                let (a, b) = s.split_once('x')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .unwrap_or((1280, 800));
        let yaw = flag("--yaw").and_then(|v| v.parse().ok()).unwrap_or(2.5f32);
        let pitch = flag("--pitch").and_then(|v| v.parse().ok()).unwrap_or(0.35f32);
        let zoom = flag("--zoom").and_then(|v| v.parse().ok()).unwrap_or(1.0f32);
        let view = flag("--cam").map(|spec| camfile::view(&spec, w as f32 / h as f32)
            .unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(2) }));
        let mut d = drive::Drive::new(rig);
        // settle the springs on the seconds before
        let mut tt = (t - 3.0).max(0.0);
        while tt < t {
            feed.apply(&mut d, rig, tt, 1.0 / 60.0);
            tt += 1.0 / 60.0;
        }
        let outp = PathBuf::from(&out);
        let mut frames = Vec::new();
        for i in 0..n {
            if i > 0 {
                for _ in 0..2 {
                    tt += 0.5 / fps;
                    feed.apply(&mut d, rig, tt, 0.5 / fps);
                }
            }
            if i == 0 {
                println!("replay t={t:.2}s: {:.0} km/h, gear {}, {:.0} rpm, steer {:+.2}, thr {:.2} brk {:.2}",
                         d.cs.speed_mps * 3.6, d.cs.gear, d.cs.rpm, d.cs.steer, d.cs.throttle, d.cs.brake);
            }
            let path = if n == 1 {
                outp.clone()
            } else {
                let stem = outp.file_stem().and_then(|s| s.to_str()).unwrap_or("frame");
                outp.with_file_name(format!("{stem}-{i:04}.png"))
            };
            frames.push((Some(rig.pose(&d.inputs(rig))), path));
        }
        pollster::block_on(shot::shot_frames(l.car, frames, w, h, yaw, pitch, zoom, None, false, view));
        return;
    }
    let opts = render::ViewOpts {
        title: format!("evoview - {model} (replay car {car})"),
        drive: 3,
        replay_feed: Some(feed),
        ..Default::default()
    };
    pollster::block_on(render::run(l.car, Some(l.skeleton), l.animations, opts));
}

fn args_has_fly() -> bool {
    std::env::args().any(|a| a == "--fly")
}

/// Load one car's banks through the game's FMOD and print what they contain.
fn fmod_probe(pkg: &PathBuf, model: &str) {
    let dir = pkg.parent().unwrap_or(std::path::Path::new("."));
    let mut p = match pkg::Package::open(pkg) {
        Ok(p) => p,
        Err(e) => { eprintln!("{e}"); std::process::exit(1); }
    };
    let mut grab = |name: &str| -> Vec<u8> {
        match p.by_path.get(&format!("content\\sfx\\{name}")).copied() {
            Some(e) => p.read(e),
            None => { eprintln!("missing content\\sfx\\{name}"); Vec::new() }
        }
    };
    let master = grab("master_bank.bank");
    let strings = grab("master_bank.strings.bank");
    let car = grab(&format!("{}.bank", model.to_ascii_lowercase()));
    println!("banks: master {}B, strings {}B, {model} {}B",
             master.len(), strings.len(), car.len());
    let mut f = match fmodsys::Fmod::new(dir) {
        Ok(f) => f,
        Err(e) => { eprintln!("FMOD: {e}"); std::process::exit(1); }
    };
    match f.load_banks(&master, &strings, &car) {
        Ok(events) => {
            println!("{} event(s) in {model}:", events.len());
            for e in &events {
                println!("   {e}");
            }
            // Try the engine event and see which parameter names it takes.
            if let Some(ev) = events.iter().find(|e| e.contains("engine_ext"))
                .or_else(|| events.iter().find(|e| e.contains("engine")))
            {
                let cands = ["rpm", "RPM", "Rpm", "engine_rpm", "load", "Load",
                             "throttle", "Throttle", "gas", "gear", "Gear",
                             "speed", "Speed", "tyre_load", "limiter", "clutch",
                             "distance", "Distance", "boost", "turbo"];
                match f.play(ev, &cands) {
                    Ok(()) => {
                        println!("playing {ev}");
                        for name in f.params.clone() {
                            match f.param_range(&name) {
                                Some((lo, hi, def)) => println!(
                                    "  param {name:10} {lo:9.1} .. {hi:9.1}  default {def:.1}"),
                                None => println!("  param {name:10} (no range)"),
                            }
                        }
                        let g = f.global_params();
                        println!("{} global parameter(s):", g.len());
                        for (n, lo, hi) in g {
                            println!("  global {n:22} {lo:9.1} .. {hi:9.1}");
                        }
                    }
                    Err(e) => eprintln!("\ncould not play {ev}: {e}"),
                }
            }
        }
        Err(e) => eprintln!("bank load: {e}"),
    }
}

/// Load a replay's track and cars and play it back.
fn play_donut(pkg: &PathBuf, base: Option<PathBuf>, track: &str, flag: &dyn Fn(&str) -> Option<String>) {
    let t0 = std::time::Instant::now();
    let model = flag("--car-id").unwrap_or_else(|| "ks_toyota_supra_mkiv".into());
    let n: usize = flag("--cars").and_then(|v| v.parse().ok()).unwrap_or(5);
    let env3 = std::env::var("EVOVIEW_DONUT").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse::<f32>().ok()).collect::<Vec<_>>());
    let (centre, radius) = match env3.as_deref() {
        Some([x, z, r]) => (glam::Vec3::new(*x, 0.0, *z), *r),
        _ => (trackdrive::spawn_point(pkg, track).map_or(glam::Vec3::ZERO, |p| p.0), 9.0),
    };
    let mut d = replay::Donut {
        centre, radius, speed: flag("--speed").and_then(|v| v.parse().ok()).unwrap_or(9.0), angle: 48.0, seconds: 30.0,
        wheelbase: 2.55, track_width: 1.52, tyre_radius: 0.33,
        variation: flag("--variation").and_then(|v| v.parse().ok()).unwrap_or(1.0),
    };
    // first flat, to load the track and cars; then again on the ground
    let flat = replay::Replay::donut(track, &model, n, &d, &|_, _| None);
    let mut scene = match replayscene::Scene::build(&flat, pkg, base.as_deref()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("could not build the donut scene: {e}");
            std::process::exit(1);
        }
    };
    let rep = match scene.car.track_env.as_deref().and_then(|e| e.collision.as_ref().map(|c| (c, e.surface_defs.clone()))) {
        Some((c, defs)) => {
            let g = trackdrive::Ground::new(c, &defs);
            let top = 1.0e4;
            d.centre.y = g.probe(centre.x, centre.z, top).map_or(0.0, |h| h.y);
            replay::Replay::donut(track, &model, n, &d, &|x, z| g.probe(x, z, top).map(|h| h.y))
        }
        None => flat,
    };
    println!("donut: {n} x {model} on {track} round ({:.1}, {:.1}, {:.1}) r {radius} m, {:.1}s",
             d.centre.x, d.centre.y, d.centre.z, t0.elapsed().as_secs_f32());
    play_loaded(rep, scene, format!("donut on {track}"), pkg, base, flag);
}

fn play_convoy(pkg: &PathBuf, base: Option<PathBuf>, track: &str, flag: &dyn Fn(&str) -> Option<String>) {
    let t0 = std::time::Instant::now();
    let models: Vec<String> = flag("--models").unwrap_or_else(|| "ks_bmw_m4_gt3".into())
        .split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let n: usize = flag("--cars").and_then(|v| v.parse().ok()).unwrap_or(models.len());
    let line = trackdrive::racing_line_layout(pkg, track, flag("--layout").as_deref());
    if line.len() < 8 {
        eprintln!("no racing line for {track} (layout {:?})", flag("--layout"));
        std::process::exit(1);
    }
    let f2 = |name: &str, d: f32| flag(name).and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    // start --lead metres before the line point nearest --at
    let mut acc = 0.0f32;
    let mut best = (f32::MAX, 0.0f32);
    let atp = flag("--at").map(|s| s.split(',').filter_map(|x| x.trim().parse::<f32>().ok()).collect::<Vec<_>>());
    for i in 0..line.len() {
        if let Some([x, z]) = atp.as_deref() {
            let d = (line[i].x - x).powi(2) + (line[i].z - z).powi(2);
            if d < best.0 {
                best = (d, acc);
            }
        }
        acc += (line[(i + 1) % line.len()] - line[i]).length();
    }
    let mut c = replay::Convoy {
        line, start_s: best.1 - f2("--lead", 60.0), speed: f2("--speed", 22.0), gap: f2("--gap", 11.0),
        seconds: f2("--seconds", 20.0), wheelbase: 2.65, track_width: 1.6, tyre_radius: 0.34,
    };
    let flat = replay::Replay::convoy(track, &models, n, &c, &|_, _, _| None);
    let scene = match replayscene::Scene::build(&flat, pkg, base.as_deref()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("could not build the convoy scene: {e}");
            std::process::exit(1);
        }
    };
    // the cars' own footprint (the first model's wheels): an F2004 is
    // 3.1 m between the axles, not a GT3's 2.65
    if let Some(s) = scene.slots.first() {
        let w = s.centres;
        let wb = ((w[0].z + w[1].z) - (w[2].z + w[3].z)).abs() * 0.5;
        let tw = ((w[0].x - w[1].x).abs() + (w[2].x - w[3].x).abs()) * 0.5;
        if (1.5..4.5).contains(&wb) && (1.0..2.5).contains(&tw) {
            (c.wheelbase, c.track_width, c.tyre_radius) = (wb, tw, s.radius.clamp(0.25, 0.45));
            println!("convoy: footprint {wb:.2} m x {tw:.2} m, tyre r {:.2} ({})", c.tyre_radius, s.model);
        }
    }
    let rep = match scene.car.track_env.as_deref().and_then(|e| e.collision.as_ref().map(|col| (col, e.surface_defs.clone()))) {
        Some((col, defs)) => {
            let g = trackdrive::Ground::new(col, &defs);
            // the line itself on the ground too (it is recorded a little high)
            for p in c.line.iter_mut() {
                if let Some(h) = g.probe(p.x, p.z, p.y + 3.0) {
                    p.y = h.y;
                }
            }
            replay::Replay::convoy(track, &models, n, &c, &|x, z, top| g.probe(x, z, top).map(|h| h.y))
        }
        None => flat,
    };
    // EVOVIEW_LINE_DUMP: the grounded line every 10 m around --at (camera placing)
    if std::env::var_os("EVOVIEW_LINE_DUMP").is_some() {
        let mut acc = 0.0f32;
        let mut next = best.1 - 150.0;
        for i in 0..c.line.len() {
            if acc >= next && acc <= best.1 + 250.0 {
                let p = c.line[i];
                println!("LINE s {:+.0} ({:.1}, {:.1}, {:.1})", acc - best.1, p.x, p.y, p.z);
                next = acc + 10.0;
            }
            acc += (c.line[(i + 1) % c.line.len()] - c.line[i]).length();
        }
    }
    println!("convoy: {n} car(s) of {} model(s) on {track}, from {:.0} m, {:.1}s", models.len(), c.start_s, t0.elapsed().as_secs_f32());
    play_loaded(rep, scene, format!("convoy on {track}"), pkg, base, flag);
}

fn play_replay(path: &PathBuf, pkg: &PathBuf, base: Option<PathBuf>) {
    let rep = match replay::Replay::load(path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("could not read replay {}: {e}", path.display());
            std::process::exit(1);
        }
    };
    println!(
        "{}: {} car(s) on {}, {:.0}s captured",
        path.file_name().and_then(|s| s.to_str()).unwrap_or("replay"),
        rep.cars.len(), rep.track.as_deref().unwrap_or("?"), rep.captured());
    let scene = match replayscene::Scene::build(&rep, pkg, base.as_deref()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("could not build the replay scene: {e}");
            std::process::exit(1);
        }
    };
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("replay").to_string();
    let flag = |name: &str| -> Option<String> {
        let a: Vec<String> = std::env::args().collect();
        a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned()
    };
    play_loaded(rep, scene, name, pkg, base, &flag);
}

fn play_loaded(rep: replay::Replay, mut scene: replayscene::Scene, name: String, pkg: &PathBuf,
               _base: Option<PathBuf>, flag: &dyn Fn(&str) -> Option<String>) {
    let t0 = std::time::Instant::now();
    for s in &scene.slots {
        println!("  {} x{}", s.model, s.cars.len());
    }
    let at = flag("--time").and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0);
    let wheels = replayscene::Wheels::build(&rep);
    scene.pose_at(&rep, &wheels, at);
    let focus = rep
        .cars
        .iter()
        .enumerate()
        .filter(|(i, _)| !rep.is_placeholder(*i))
        .filter_map(|(i, _)| rep.sample(i, at).map(|p| p.pos))
        // ⚠ A car can be sitting exactly at the origin at this instant even
        // though it drives later (spawning, or in the pits). Averaging those
        // in drags the camera off the pack and below the track.
        .filter(|p| p.length_squared() > 1.0)
        .fold((glam::Vec3::ZERO, 0.0f32), |(s, n), p| (s + p, n + 1.0));
    let mut focus = (focus.1 > 0.0).then(|| focus.0 / focus.1).or_else(|| scene.start_focus(&rep));
    // --car N centres on one driver instead of the pack, for looking at a
    // single car closely.
    if let Some(n) = flag("--car").and_then(|v| v.parse::<usize>().ok()) {
        if let Some(p) = rep.sample(n, at) {
            focus = Some(p.pos);
        }
    }
    if let Some(f) = focus {
        println!("cars centre on ({:.1}, {:.1}, {:.1}) at t={at}s", f.x, f.y, f.z);
    }
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // A still of the replay at `--time`, for checking playback without a human
    // watching the window.
    if let Some(out) = flag("--shot") {
        let (w, h) = flag("--size")
            .and_then(|s| {
                let (a, b) = s.split_once('x')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .unwrap_or((1600, 1000));
        let eye = focus.unwrap_or(glam::Vec3::ZERO);
        // --view puts a follow camera on the car chosen by --car, so each
        // replay camera can be checked as a still.
        let view = flag("--view").and_then(|name| {
            let mode = match name.as_str() {
                "chase" => camera::Follow::Chase,
                "onboard" => camera::Follow::Onboard,
                "side" => camera::Follow::Side,
                "orbit" => camera::Follow::Orbit,
                _ => return None,
            };
            let ci = flag("--car").and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
            let p = rep.sample(ci, at)?;
            Some(mode.view_proj(w as f32 / h as f32, p.pos, p.forward, p.up, p.right, at))
        });
        pollster::block_on(shot::shot(
            scene.car, &PathBuf::from(out), w, h,
            flag("--yaw").and_then(|v| v.parse().ok()).unwrap_or(2.5f32),
            flag("--pitch").and_then(|v| v.parse().ok()).unwrap_or(0.35f32),
            flag("--zoom").and_then(|v| v.parse().ok()).unwrap_or(1.0f32),
            Some(eye), None, true, view));
        return;
    }
    let title = format!("evoview - {name}");
    let bench = flag("--bench").and_then(|v| v.parse::<f32>().ok()).map(|secs| {
        let (width, height) = flag("--size").and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        }).unwrap_or((1280, 720));
        bench::Opts { seconds: secs, width, height, name: name.clone(), shot: flag("--bench-shot").map(PathBuf::from) }
    });
    pollster::block_on(render::run(
        scene.car, None, Vec::new(),
        render::ViewOpts {
            start_fly: true,
            ground: false,
            title,
            playback: Some(render::Playback {
                replay: rep, slots: scene.slots, wheels,
            }),
            drive: 0,
            replay_feed: None,
            pkg: Some(pkg.clone()),
            focus,
            bench,
            livery: None,
            track_drive: None,
            tv: None,
            line_driver: None,
            car_id: None,
            pits: None,
            laps: None,
            race: None,
            ai_slots: Vec::new(),
            mp: None,
        },
    ));
}

/// A car driven on a track: the track and the car merged into one scene, the
/// car's physics from its own data, the ground from the track's colliders.
fn drive_on_track(pkg: &PathBuf, base: Option<&std::path::Path>, track: &str, car_id: &str,
                  flag: &dyn Fn(&str) -> Option<String>) {
    let t0 = std::time::Instant::now();
    // --ai N / EVOVIEW_AI=N: opponents (race.rs), one model slot each
    let n_ai = flag("--ai").and_then(|v| v.parse::<usize>().ok()).map_or_else(race::env_count, |n| n.min(race::NAMES.len()));
    // EVOVIEW_MP=host:port: online (mp.rs); the other players are drawn in
    // EVOVIEW_MP_CARS render slots (4) of EVOVIEW_AI_CARS or this car's model
    let mp_addr = std::env::var("EVOVIEW_MP").ok().filter(|v| !v.is_empty());
    // online: join first - the other players' cars decide what is loaded
    let mp = mp_addr.and_then(|addr| {
        let Some((fs, mech)) = mp::car_final_state(pkg, car_id) else {
            return Some(mp::Mp::failed(format!("could not join: no spawn preset (.carfinalstate) found for {car_id}")));
        };
        // EVOVIEW_MP_NAME (the evoview tab's Driver name), else the Steam
        // account signed in here; its id is who we join as
        let steam = mp::steam_user();
        let user = std::env::var("EVOVIEW_MP_USER").ok().filter(|v| !v.is_empty())
            .or_else(|| steam.as_ref().map(|s| s.0.clone()))
            .unwrap_or_else(|| "76561190000000000".into());
        let name = std::env::var("EVOVIEW_MP_NAME").ok().map(|v| v.trim().chars().take(32).collect::<String>())
            .filter(|v| !v.is_empty())
            .or_else(|| steam.as_ref().map(|s| s.1.clone()).filter(|n| !n.is_empty()))
            .unwrap_or_else(|| "evoview driver".into());
        println!("mp: joining {addr} as {name:?}");
        Some(mp::Mp::start(addr, user, name, fs, mech))
    });
    let (ai_models, ai_own) = match mp.as_ref() {
        Some(m) => mp_models(pkg, car_id, m),
        // EVOVIEW_AI_OWN=<file>: every opponent wears that CarFinalStateData
        // (a player's look captured online) - checks of the online liveries
        None => (race::models(car_id, n_ai), std::env::var("EVOVIEW_AI_OWN").ok()
            .and_then(|f| std::fs::read(f).ok()).map_or(Vec::new(), |b| vec![Some(b); n_ai])),
    };
    // online: the server's track, whatever the evoview tab had picked
    let server_track = mp.as_ref().and_then(|m| m.track());
    let track: &str = match server_track.as_deref() {
        Some(t) if !t.eq_ignore_ascii_case(track) => {
            println!("mp: the server runs {t}; loading that instead of {track}");
            t
        }
        _ => track,
    };
    // online: the server's layout - its containers, and the layout id its
    // layout_<id>.scene names (the racing line and the name rules follow it)
    if let Some(m) = mp.as_ref() {
        let cs = m.containers();
        let lay = cs.iter().filter_map(|c| {
            let b = c.rsplit(['\\', '/']).next()?.to_lowercase();
            b.strip_prefix("layout_")?.strip_suffix(".scene").map(str::to_string)
        }).next();
        if let Some(l) = lay {
            println!("mp: the server runs layout {l}");
            std::env::set_var("EVOVIEW_LAYOUT", l);
        }
        *load::SESSION_CONTAINERS.write().unwrap() = cs;
    }
    let (mut scene, mut car_vfs, ai_vfs, ai_place) = match replayscene::Scene::build_race(pkg, base, track, car_id, &ai_models, &ai_own) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("could not load: {e}");
            std::process::exit(1);
        }
    };
    let phys = carphys::Physics::read(&mut car_vfs);
    let phys_pits = phys.clone();
    let aero = phys.as_ref().map(|p| lapsim::aero(&mut car_vfs, p)).unwrap_or_default();
    let env = scene.car.track_env.as_deref();
    let Some(coll) = env.and_then(|e| e.collision.as_ref()) else {
        eprintln!("{track}: no collision surfaces found");
        std::process::exit(1);
    };
    let ground = std::sync::Arc::new(trackdrive::Ground::new(coll, &env.map(|e| e.surface_defs.clone()).unwrap_or_default()));
    let slot = scene.slots[0].clone();
    // the opponents: each its own physics on its model's data
    let ai_rig = std::env::var("EVOVIEW_AI_RIG").map_or(std::env::var_os("EVOVIEW_RIGCAR").is_some(), |v| v != "0");
    let phys_ai: Vec<_> = ai_vfs.into_iter().map(|(_, mut vfs)| {
        let ph = carphys::Physics::read(&mut vfs);
        let aero = ph.as_ref().map(|p| lapsim::aero(&mut vfs, p)).unwrap_or_default();
        (ph, aero)
    }).collect();
    let mut ai_cars = Vec::new();
    for &(mi, si, inst) in &ai_place {
        let (ph, aero) = &phys_ai[mi];
        let mut t = trackdrive::TrackDrive::with_model(ground.clone(), scene.slots[si].clone(), ph.clone(), ai_rig, false);
        if let Some((rc, _)) = t.rig.as_mut() {
            rc.aero = *aero;
            rc.water_flow = false;
        }
        ai_cars.push((t, si, inst));
    }
    let mut td = trackdrive::TrackDrive::with_model(ground, slot, phys, std::env::var_os("EVOVIEW_RIGCAR").is_some(), true);
    td.set_aero(aero);
    let (at, yaw) = trackdrive::spawn_point(pkg, track).unwrap_or((glam::Vec3::ZERO, 0.0));
    if !td.place(at, yaw) {
        eprintln!("no drivable surface at the spawn point ({:.0}, {:.0}, {:.0})", at.x, at.y, at.z);
    }
    // online: the pit box the server handed out
    let mp_spawn = mp.as_ref().and_then(|m| mp_spawn(pkg, track, m));
    if let Some(m) = mp.as_ref() {
        match mp_spawn {
            Some((at, yaw)) => {
                if !td.place_from(at, yaw, 3.0) {
                    eprintln!("mp: no surface under the pit box ({:.1}, {:.1}, {:.1})", at.x, at.y, at.z);
                }
            }
            None => eprintln!("mp: no car from the server yet ({})", m.status()),
        }
    }
    println!("{car_id} on {track}: spawned at ({:.1}, {:.1}, {:.1}) heading {:.0} deg, {:.1}s",
             td.pos.x, td.pos.y, td.pos.z, yaw.to_degrees(), t0.elapsed().as_secs_f32());
    // the pit lane, its rules and the car's pit data (pits.rs)
    let mut pits = phys_pits.as_ref().map(|p| {
        let mut car = pits::PitCar::load(&mut car_vfs, p);
        // the garage's wing / ride-height rows rebuild the aero from these
        car.aero_src = Some((car_vfs, p.clone()));
        let arb = td.rig.as_ref().map_or([0.0; 2], |(rc, _)| rc.arb());
        pits::Pits::new(pits::PitLane::load(pkg, track), pits::Rules::load(pkg), car, arb)
    });
    if let (Some(secs), Some(p)) = (flag("--pit-test").and_then(|v| v.parse::<f32>().ok()), pits.as_mut()) {
        pits::test(pkg, track, &mut td, p, secs, flag("--pit-script").as_deref().unwrap_or("stop"));
        return;
    }
    if let Some(p) = pits.as_mut() {
        pits::start(&mut td, p);
    }
    // lap timing (laps.rs); scripted drivers' laps are not saved as PBs
    let scripted = flag("--drive-test").is_some() || flag("--bench").is_some() || std::env::var_os("EVOVIEW_DRIVE_LINE").is_some();
    let mut laps = laps::Session::load(pkg, track, car_id, !scripted);
    let mut race = (!ai_cars.is_empty()).then(|| {
        let grid = pits::PitLane::load(pkg, track).map(|l| l.grid).unwrap_or_default();
        let mut r = race::Race::new(trackdrive::racing_line(pkg, track), ai_cars, &grid, &mut td);
        r.net = mp.is_some();
        // online the server runs the session: no lights, no hold, no banner
        if r.net {
            r.go = Some(-10.0);
        }
        r
    });
    // Race::new put the player on the grid: online, the server's pit box
    if let Some((at, yaw)) = mp_spawn {
        td.place_from(at, yaw, 3.0);
    }
    // EVOVIEW_AI_WARP=s: the race run on headless for s seconds first (the
    // player's car on the racing-line driver) - mid-race captures
    if let (Some(r), Some(w)) = (race.as_mut(), std::env::var("EVOVIEW_AI_WARP").ok().and_then(|v| v.parse::<f32>().ok())) {
        let mut d = trackdrive::LineDriver::new(trackdrive::racing_line(pkg, track));
        d.snap(td.pos);
        d.offset = race::lateral(&r.line, td.pos);
        d.pace = 0.95;
        let dt = 1.0 / 120.0;
        let mut t = 0.0;
        while t < w {
            if r.held() {
                d.rest();
            }
            let mut c = d.controls(&td, dt);
            r.filter_player(&mut c, td.speed_kmh());
            td.step(&c, dt);
            r.step(&mut td, dt);
            t += dt;
        }
        println!("race: warped {w:.0} s; order {:?}", r.race_order().iter().map(|(k, v)| format!("{} {v:.2}", r.name(*k))).collect::<Vec<_>>());
    }
    if let Some(secs) = flag("--drive-test").and_then(|v| v.parse::<f32>().ok()) {
        // follow the racing line unless --drive-script weave is asked for
        // (a track without a racing line - interns - gets the weave)
        let line = trackdrive::racing_line(pkg, track);
        if line.len() < 3 {
            println!("drive-test: {track} has no racing line; the scripted weave instead");
        }
        let driver = (flag("--drive-script").as_deref() != Some("weave") && line.len() >= 3)
            .then(|| trackdrive::LineDriver::new(line));
        // --wav out.wav: the racing-line drive through the car's sound,
        // rendered offline with a chase-camera listener
        if let Some(wav) = flag("--wav") {
            let mut driver = driver.expect("--wav needs the racing-line driver");
            let evs = scene.car.fx.as_ref().map_or(Vec::new(), |f| f.actor.audio_events.clone());
            let mut s = carsound::CarSound::new(pkg, car_id, &evs, Some(std::path::Path::new(&wav))).expect("sound");
            let dt = s.fmod.block_seconds();
            let mut t = 0.0f32;
            // with --ai the race runs too and its nearest engines play
            if let Some(r) = race.as_ref() {
                driver.snap(td.pos);
                driver.offset = race::lateral(&r.line, td.pos);
            }
            while t < secs {
                if race.as_ref().map_or(false, |r| r.held()) {
                    driver.rest();
                }
                let mut c = driver.controls(&td, dt);
                if let Some(r) = race.as_ref() {
                    r.filter_player(&mut c, td.speed_kmh());
                }
                td.step(&c, dt);
                if let Some(r) = race.as_mut() {
                    r.step(&mut td, dt);
                }
                let mut f = td.sound_feed(c.handbrake);
                let p = td.pose();
                f.car_pos = (p.pos + p.up * 0.5).to_array();
                f.car_vel = (p.forward * td.vx).to_array();
                f.car_fwd = p.forward.to_array();
                f.car_up = p.up.to_array();
                f.cam_pos = (p.pos - p.forward * 6.0 + p.up * 2.0).to_array();
                f.cam_fwd = p.forward.to_array();
                f.cam_up = p.up.to_array();
                if let Some(r) = race.as_ref() {
                    s.update_ai(&r.sound_cars(), f.cam_pos, dt);
                }
                s.update(&f, dt);
                t += dt;
            }
            s.stop_all();
            s.fmod.release();
            println!("drive sound: {wav} ({secs:.0} s)");
            return;
        }
        match (race.as_mut(), driver) {
            (Some(r), Some(d)) => race::test_run(r, &mut td, d, secs),
            (_, driver) => trackdrive::test_run(&mut td, secs, driver, laps.as_mut()),
        }
        if let Some(out) = flag("--shot") {
            let (w, h) = flag("--size").and_then(|s| {
                let (a, b) = s.split_once('x')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            }).unwrap_or((1280, 720));
            // bake the car where the drive ended, then a chase camera on it
            let (b, wm) = td.matrices();
            for &i in &td.slot.meshes {
                scene.car.meshes[i].xforms = vec![b.to_cols_array()];
            }
            for k in 0..4 {
                for &i in &td.slot.wheels[k] {
                    scene.car.meshes[i].xforms = vec![wm[k].to_cols_array()];
                }
            }
            let p = td.pose();
            let view = Some(camera::Follow::Chase.view_proj(w as f32 / h as f32, p.pos + p.up * 0.45,
                                                           p.forward, p.up, p.right, 0.0));
            pollster::block_on(shot::shot(scene.car, &PathBuf::from(out), w, h, 2.5, 0.35, 1.0,
                                          Some(p.pos), None, true, view));
        }
        return;
    }
    if scene.skeleton.is_none() {
        println!("  {car_id}: no rig; the body and wheels move, nothing else animates");
    }
    // EVOVIEW_RIG_LOG: where the rig's rest pose puts the hubs and the body
    // against their authored (bind) positions
    if let (Some(sk), true) = (scene.skeleton.as_ref(), std::env::var_os("EVOVIEW_RIG_LOG").is_some()) {
        if let Some(rig) = sk.rig.as_deref() {
            let mut d = drive::Drive::new(rig);
            d.follow(rig, &td.rig_feed(), false, 0.02);
            let pose = rig.pose(&d.inputs(rig));
            for (i, b) in sk.bones.iter().enumerate() {
                let n = b.name.to_uppercase();
                if n.starts_with("HUB_") || n.starts_with("TYRE_") || n.starts_with("WHEEL_") || n == "BODY" || n.starts_with("SUSP_") {
                    let w = load::mat_mul(&pose[i], &sk.bind[i]);
                    println!("RIG {:14} bind ({:6.3} {:6.3} {:6.3})  posed ({:6.3} {:6.3} {:6.3})", b.name,
                             sk.bind[i][12], sk.bind[i][13], sk.bind[i][14], w[12], w[13], w[14]);
                }
            }
        }
    }
    // --bench <s> [--bench-shot out.png]: the live loop offscreen, for
    // looking at what only shows while running (TAA, animation)
    let bench = flag("--bench").and_then(|v| v.parse::<f32>().ok()).map(|secs| {
        let (width, height) = flag("--size").and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        }).unwrap_or((1280, 720));
        bench::Opts { seconds: secs, width, height, name: track.to_string(), shot: flag("--bench-shot").map(PathBuf::from) }
    });
    pollster::block_on(render::run(
        scene.car, scene.skeleton, scene.animations,
        render::ViewOpts {
            start_fly: true,
            ground: false,
            title: format!("evoview - {car_id} on {track}"),
            focus: Some(td.pos),
            pkg: Some(pkg.clone()),
            track_drive: Some(td),
            line_driver: std::env::var_os("EVOVIEW_DRIVE_LINE").map(|_| trackdrive::racing_line(pkg, track)).filter(|l| l.len() >= 3).map(trackdrive::LineDriver::new),
            tv: Some(cams::TvDirector::new(cams::load_tv(pkg, track))),
            car_id: Some(car_id.to_string()),
            pits,
            laps,
            race,
            ai_slots: scene.slots.iter().skip(1).cloned().collect(),
            mp,
            bench,
            ..Default::default()
        },
    ));
}

/// Summarise a replay: what cars are in it, where they start, and how far each
/// one travels. Distance is the useful sanity check - a stream that decodes
/// with the wrong stride produces a wild, jittering total rather than a lap.
fn dump_replay(path: &PathBuf) {
    let t0 = std::time::Instant::now();
    let r = match replay::Replay::load(path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("could not read replay {}: {e}", path.display());
            std::process::exit(1);
        }
    };
    let wrapped = if r.captured() < r.duration - 5.0 { " (buffer wrapped)" } else { "" };
    println!(
        "v{} {} - {} car(s), {:.0}s session, {:.0}s captured{} @ {:.2} Hz, read in {:.2}s",
        r.version, r.model, r.cars.len(), r.duration, r.captured(), wrapped, r.rate(),
        t0.elapsed().as_secs_f32());
    for (i, c) in r.cars.iter().enumerate() {
        if r.is_placeholder(i) {
            println!("  car{i:2} {:32} (empty slot)", c.model.as_deref().unwrap_or("?"));
            continue;
        }
        let p = r.pose(i, 0);
        let mut dist = 0.0f32;
        let mut prev = p.pos;
        for f in 1..c.n_matrix as usize {
            let q = r.pose(i, f).pos;
            dist += prev.distance(q);
            prev = q;
        }
        let tel = r.telemetry(i, r.captured() * 0.15);
        println!(
            "  car{i:2} {:30} {:5}f {:6.0}m  start=({:7.1},{:6.1},{:7.1})  \
             @15%: {:5.0}rpm g{} {:3.0}km/h thr{:3.0}% brk{:3.0}% steer{:+5.0}deg",
            c.model.as_deref().unwrap_or("?"), c.n_matrix, dist,
            p.pos.x, p.pos.y, p.pos.z,
            tel.map_or(0.0, |t| t.rpm), tel.map_or(0, |t| t.gear),
            r.speed(i, r.captured() * 0.15) * 3.6,
            tel.map_or(0.0, |t| t.throttle * 100.0),
            tel.map_or(0.0, |t| t.brake * 100.0),
            tel.map_or(0.0, |t| t.steer_deg));
    }
}

/// Online (mp.rs): one opponent slot per other player, loaded with that
/// player's car in their own design and paints, then EVOVIEW_MP_SPARE (24)
/// spares of our car for players who join later. Every model on the server
/// (EVOVIEW_MP_MODELS caps it to the most common; 0 = all) and at most
/// EVOVIEW_MP_CARS (64) cars in all.
/// The pit box the server handed out, facing the way the track's own box
/// there does: the spawn transform's rotation has its own sign convention
/// (taken as is, the car stood backwards in most boxes).
fn mp_spawn(pkg: &PathBuf, track: &str, m: &mp::Mp) -> Option<(glam::Vec3, f32)> {
    let (at, yaw) = m.spawn()?;
    let lane = pits::PitLane::load(pkg, track);
    let near = lane.as_ref().and_then(|l| {
        l.boxes.iter().chain(&l.pits).chain(&l.grid)
            .map(|s| (s.pos.distance(at), s.yaw))
            .filter(|(d, _)| *d < 4.0)
            .min_by(|a, b| a.0.total_cmp(&b.0))
    });
    match near {
        Some((d, y)) => {
            println!("mp: pit box heading {:.0} deg from the track's box {d:.1} m away (server's said {:.0})", y.to_degrees(), yaw.to_degrees());
            Some((at, y))
        }
        None => Some((at, yaw)),
    }
}

fn mp_models(pkg: &PathBuf, car_id: &str, m: &mp::Mp) -> (Vec<String>, Vec<Option<Vec<u8>>>) {
    let env_n = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(d);
    let roster = m.wait_roster(12.0);
    // only cars this install has (a server's mod car would fail the load)
    let have = |c: &str| {
        let dir = format!("content\\cars\\{}\\", c.to_lowercase());
        pkg::Package::open_filtered(pkg, &[&dir]).map_or(false, |p| !p.by_path.is_empty())
    };
    let mut count: std::collections::HashMap<String, usize> = Default::default();
    for r in &roster {
        *count.entry(r.1.to_lowercase()).or_default() += 1;
    }
    let mut common: Vec<(String, usize)> = count.into_iter().collect();
    common.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut allowed: std::collections::HashSet<String> = Default::default();
    for (c, _) in common {
        let cap = env_n("EVOVIEW_MP_MODELS", 0);
        if cap > 0 && allowed.len() >= cap {
            break;
        }
        if !c.is_empty() && have(&c) {
            allowed.insert(c);
        }
    }
    let mut ai: Vec<(String, Option<u16>)> = roster.iter()
        .map(|r| (if allowed.contains(&r.1.to_lowercase()) { r.1.to_lowercase() } else { car_id.to_string() }, Some(r.0)))
        .collect();
    ai.truncate(env_n("EVOVIEW_MP_CARS", 64));
    for _ in 0..env_n("EVOVIEW_MP_SPARE", 24) {
        ai.push((car_id.to_string(), None));
    }
    println!("mp: {} other car(s) on the server, {} model(s) loaded, {} spare slot(s)",
             roster.len(), allowed.len(), env_n("EVOVIEW_MP_SPARE", 24));
    for r in &roster {
        println!("  #{:<3} {:28} {}", r.0, r.1, r.2);
    }
    let models = ai.iter().map(|a| a.0.clone()).collect();
    // their own design and paints (only on their own model)
    let own = ai.iter().map(|a| a.1.filter(|_| a.0 != car_id || roster.iter().any(|r| Some(r.0) == a.1 && r.1.eq_ignore_ascii_case(car_id)))
        .and_then(|n| m.final_state(n))).collect();
    m.set_ai(ai, car_id);
    (models, own)
}
