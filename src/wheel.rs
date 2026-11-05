//! A force-feedback wheel through DirectInput 8, the way the game drives
//! one (DirectInputDevice::setFF 0x141f52030): one constant-force effect on
//! the X axis, magnitude = clamp(force x 10000, +-10000), plus a damper
//! effect. dinput8.dll is loaded at runtime and its COM vtables are called
//! directly, so there is no extra crate or import library.
//!
//! Every attached game controller is opened - a Moza base, its pedals and an
//! HGP shifter are separate USB devices - the force-feedback one exclusively
//! (the game's cooperative level), the rest shared. `--wheel-setup` asks for
//! each control in turn (steering, pedals, paddles, H gates) and saves the
//! mapping to %APPDATA%/evoview/wheel.cfg; without it the first FF device
//! steers on X with Y / Rz pedals. EVOVIEW_WHEEL_LOG=1 prints raw state.

use std::ffi::c_void;

type HRESULT = i32;

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq)]
pub struct Guid(u32, u16, u16, [u8; 8]);

const fn g(a: u32, b: u16, c: u16, d: [u8; 8]) -> Guid {
    Guid(a, b, c, d)
}
const IID_IDIRECTINPUT8W: Guid = g(0xBF798031, 0x483A, 0x4DA2, [0xAA, 0x99, 0x5D, 0x64, 0xED, 0x36, 0x97, 0x00]);
const GUID_CONSTANT_FORCE: Guid = g(0x13541C20, 0x8E33, 0x11D0, [0x9A, 0xD0, 0x00, 0xA0, 0xC9, 0xA0, 0x6E, 0x35]);
const GUID_DAMPER: Guid = g(0x13541C28, 0x8E33, 0x11D0, [0x9A, 0xD0, 0x00, 0xA0, 0xC9, 0xA0, 0x6E, 0x35]);
const fn axis(a: u32) -> Guid {
    g(a, 0xC9F3, 0x11CF, [0xBF, 0xC7, 0x44, 0x45, 0x53, 0x54, 0x00, 0x00])
}
const GUID_X: Guid = axis(0xA36D02E0);
const GUID_Y: Guid = axis(0xA36D02E1);
const GUID_Z: Guid = axis(0xA36D02E2);
const GUID_RX: Guid = axis(0xA36D02F4);
const GUID_RY: Guid = axis(0xA36D02F5);
const GUID_RZ: Guid = axis(0xA36D02E3);
const GUID_SLIDER: Guid = axis(0xA36D02E4);

// dinput.h
const DI8DEVCLASS_GAMECTRL: u32 = 4;
const DIEDFL_ATTACHEDONLY: u32 = 0x1;
const DIEDFL_FORCEFEEDBACK: u32 = 0x100;
const DISCL_EXCLUSIVE: u32 = 0x1;
const DISCL_BACKGROUND: u32 = 0x8;
const DIDFT_AXIS: u32 = 0x3;
const DIDFT_ANYINSTANCE: u32 = 0x00FF_FF00;
const DIDFT_OPTIONAL: u32 = 0x8000_0000;
const DIDOI_ASPECTPOSITION: u32 = 0x100;
const DIDF_ABSAXIS: u32 = 1;
const DIEFF_OBJECTOFFSETS: u32 = 0x2;
const DIEFF_CARTESIAN: u32 = 0x10;
const DIEP_AXES: u32 = 0x20;
const DIEP_DIRECTION: u32 = 0x40;
const DIEP_TYPESPECIFICPARAMS: u32 = 0x100;
const DIEP_START: u32 = 0x2000_0000;
const DIEB_NOTRIGGER: u32 = 0xFFFF_FFFF;
const INFINITE: u32 = 0xFFFF_FFFF;
const DIPH_DEVICE: u32 = 0;
const DIPROP_RANGE: usize = 4;
const DIPROP_AUTOCENTER: usize = 9;

/// 8 axes then 128 buttons (DIJOYSTATE2's button count: Moza rims report
/// far more than 32)
#[repr(C)]
#[derive(Clone, Copy)]
struct JoyState {
    axes: [i32; 8], // x y z rx ry rz slider0 slider1
    buttons: [u8; 128],
}

impl Default for JoyState {
    fn default() -> Self {
        JoyState { axes: [0; 8], buttons: [0; 128] }
    }
}

#[repr(C)]
struct ObjFormat {
    guid: *const Guid,
    ofs: u32,
    ty: u32,
    flags: u32,
}

#[repr(C)]
struct DataFormat {
    size: u32,
    obj_size: u32,
    flags: u32,
    data_size: u32,
    num_objs: u32,
    objs: *const ObjFormat,
}

#[repr(C)]
struct Effect {
    size: u32,
    flags: u32,
    duration: u32,
    sample_period: u32,
    gain: u32,
    trigger_button: u32,
    trigger_repeat: u32,
    axes_n: u32,
    axes: *const u32,
    direction: *const i32,
    envelope: *const c_void,
    type_size: u32,
    type_params: *const c_void,
    start_delay: u32,
}

#[repr(C)]
struct Condition {
    offset: i32,
    pos_coef: i32,
    neg_coef: i32,
    pos_sat: u32,
    neg_sat: u32,
    dead_band: i32,
}

#[repr(C)]
struct PropDword {
    size: u32,
    header_size: u32,
    obj: u32,
    how: u32,
    data: u32,
}

#[repr(C)]
struct PropRange {
    size: u32,
    header_size: u32,
    obj: u32,
    how: u32,
    min: i32,
    max: i32,
}

/// slot `n` of a COM object's vtable
unsafe fn vt(obj: *mut c_void, n: usize) -> *const c_void {
    *(*(obj as *const *const *const c_void)).add(n)
}

macro_rules! com {
    ($obj:expr, $slot:expr, fn($($t:ty),*) $(, $a:expr)*) => {{
        let f: unsafe extern "system" fn(*mut c_void $(, $t)*) -> HRESULT = std::mem::transmute(vt($obj, $slot));
        f($obj $(, $a)*)
    }};
}

#[link(name = "kernel32")]
extern "system" {
    fn GetModuleHandleW(name: *const u16) -> *mut c_void;
}

#[link(name = "user32")]
extern "system" {
    fn GetDesktopWindow() -> *mut c_void;
}

const DISCL_NONEXCLUSIVE: u32 = 0x2;
const DIDFT_BUTTON: u32 = 0xC;


#[derive(Clone, Copy, Debug, Default)]
pub struct WheelInput {
    /// -1 right .. +1 left (trackdrive's convention)
    pub steer: f32,
    pub throttle: f32,
    pub brake: f32,
    /// clutch pedal / paddle, 0 released .. 1 pressed (None: not mapped)
    pub clutch: Option<f32>,
    /// paddles, true on the frame they are pressed
    pub shift_up: bool,
    pub shift_down: bool,
    /// H-pattern: Some(1..=7) in a gate, Some(-1) reverse, Some(0) between
    /// gates (neutral); None without a shifter mapped
    pub h_gear: Option<i32>,
    pub handbrake: Option<f32>,
    /// starter button, true on the frame it is pressed
    pub starter: bool,
    /// cockpit aids pressed this frame, -1 down / +1 up: brake bias, TC,
    /// ABS (aids::Knob::ALL order)
    pub aids: [i32; 3],
    /// the pit buttons pressed this frame, PIT_ROLES order (pits.rs)
    pub pit: [bool; 8],
}

/// The pit roles: limiter, menu, up, down, left, right, ok, back to the pits.
pub const PIT_ROLES: [&str; 8] = ["pit_limiter", "pit_menu", "pit_up", "pit_down", "pit_left", "pit_right", "pit_ok", "pit_return"];

/// The roles that default to the game's own buttons, and the game action
/// each mirrors (payload None = any). Cockpit aids: InputAction 148
/// Brake_Bias_Cycle, 140 TractionControl_Cycle, 144 ABS_Cycle; payload 1
/// steps down, 2 up (handler fn 0x1417901a0). Pits: 138
/// Car_Pit_Limiter_Toggle is the limiter; the menu borrows the TC (140) and
/// ABS (144) cycle buttons as up / down and right / left - the cockpit aids
/// skip those presses while the pit menu is up (render.rs).
const GAME_ROLES: [(&str, u64, Option<u64>); 11] = [
    ("bias_down", 148, Some(1)), ("bias_up", 148, Some(2)),
    ("tc_down", 140, Some(1)), ("tc_up", 140, Some(2)),
    ("abs_down", 144, Some(1)), ("abs_up", 144, Some(2)),
    ("pit_limiter", 138, None), ("pit_up", 140, Some(2)), ("pit_down", 140, Some(1)),
    ("pit_right", 144, Some(2)), ("pit_left", 144, Some(1)),
];

/// The buttons the game's own controls file gives GAME_ROLES (read
/// only): %USERPROFILE%/Saved Games/ACE/input_devices.inputdeviceconfiguration,
/// InputDeviceConfiguration {1 InputDeviceCommandList {1 controller {1
/// product_name}, 2 commands {1 ActionBase {1 input, 3 payload}, 2 index
/// (the DirectInput button), 3 require_shift}}}. Shifted ones are skipped.
fn game_binds() -> Vec<(String, Bind)> {
    use crate::proto::{fields, text, Val};
    let Some(home) = crate::envc::var_os("USERPROFILE") else { return Vec::new() };
    let p = std::path::PathBuf::from(home).join("Saved Games").join("ACE").join("input_devices.inputdeviceconfiguration");
    let Ok(d) = std::fs::read(p) else { return Vec::new() };
    let mut out = Vec::new();
    fields(&d, 0, d.len(), |f, v| {
        let (1, Some((a, b))) = (f, v.range()) else { return };
        let mut name = String::new();
        let mut cmds = Vec::new();
        fields(&d, a, b, |g, w| match (g, w.range()) {
            (1, Some((x, y))) => fields(&d, x, y, |h, u| {
                if let (1, Some(r)) = (h, u.range()) {
                    name = text(&d, r);
                }
            }),
            (2, Some((x, y))) => {
                let (mut act, mut pay, mut idx, mut shift) = (0u64, 0u64, None, false);
                fields(&d, x, y, |h, u| match (h, u) {
                    (1, Val::Bytes(i, j)) => fields(&d, i, j, |k, z| match (k, z) {
                        (1, Val::Varint(n)) => act = n,
                        (3, Val::Varint(n)) => pay = n,
                        _ => {}
                    }),
                    (2, Val::Varint(n)) => idx = Some(n as usize),
                    (3, Val::Varint(n)) => shift = n != 0,
                    _ => {}
                });
                if let (Some(i), false) = (idx, shift) {
                    cmds.push((act, pay, i));
                }
            }
            _ => {}
        });
        for (role, act, pay) in GAME_ROLES {
            if let Some(&(_, _, i)) = cmds.iter().find(|c| c.0 == act && pay.map_or(true, |p| c.1 == p)) {
                if i < 128 {
                    out.push((role.to_string(), Bind::Button { dev: name.clone(), button: i }));
                }
            }
        }
    });
    out
}

/// One control: an axis between two raw readings, or a button.
#[derive(Clone, Debug)]
enum Bind {
    Axis { dev: String, axis: usize, rest: i32, full: i32 },
    Button { dev: String, button: usize },
}

/// role -> control; the H gates are "h1".."h7" and "hr"
#[derive(Clone, Debug, Default)]
pub struct Mapping {
    binds: Vec<(String, Bind)>,
}

impl Mapping {
    fn path() -> std::path::PathBuf {
        let base = crate::envc::var_os("APPDATA").map(std::path::PathBuf::from).unwrap_or_else(|| ".".into());
        base.join("evoview").join("wheel.cfg")
    }

    /// `role axis <axis> <rest> <full> <device>` / `role button <n> <device>`
    pub fn load() -> Option<Mapping> {
        let text = std::fs::read_to_string(Self::path()).ok()?;
        let mut m = Mapping::default();
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            let mut it = line.splitn(6, ' ');
            let (Some(role), Some(kind)) = (it.next(), it.next()) else { continue };
            let b = match kind {
                "axis" => {
                    let (Some(a), Some(r), Some(f), Some(d)) = (it.next(), it.next(), it.next(), it.next()) else { continue };
                    let rest = it.next().map(|x| format!("{d} {x}")).unwrap_or_else(|| d.to_string());
                    Bind::Axis { dev: rest, axis: a.parse().ok()?, rest: r.parse().ok()?, full: f.parse().ok()? }
                }
                "button" => {
                    let (Some(n), Some(d)) = (it.next(), it.next()) else { continue };
                    let rest: Vec<&str> = std::iter::once(d).chain(it).collect();
                    Bind::Button { dev: rest.join(" "), button: n.parse().ok()? }
                }
                _ => continue,
            };
            m.binds.push((role.to_string(), b));
        }
        Some(m)
    }

    fn save(&self) -> std::io::Result<std::path::PathBuf> {
        let p = Self::path();
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        let mut out = String::from("# evoview wheel mapping (--wheel-setup writes this)\n");
        for (role, b) in &self.binds {
            match b {
                Bind::Axis { dev, axis, rest, full } => out += &format!("{role} axis {axis} {rest} {full} {dev}\n"),
                Bind::Button { dev, button } => out += &format!("{role} button {button} {dev}\n"),
            }
        }
        std::fs::write(&p, out)?;
        Ok(p)
    }

    fn get(&self, role: &str) -> Option<&Bind> {
        self.binds.iter().find(|(r, _)| r == role).map(|(_, b)| b)
    }
}

struct Device {
    name: String,
    dev: *mut c_void,
    state: JoyState,
    force: *mut c_void,
    damper: *mut c_void,
}

pub struct Wheel {
    _lib: libloading::Library,
    di: *mut c_void,
    devs: Vec<Device>,
    /// the device that gets force feedback
    ffb: Option<usize>,
    pub name: String,
    map: Mapping,
    prev_buttons: Vec<[u8; 128]>,
    last_log: std::time::Instant,
    damper_on: bool,
}

extern "system" fn enum_all(inst: *const u8, ctx: *mut c_void) -> i32 {
    unsafe {
        let out = &mut *(ctx as *mut Vec<(Guid, String)>);
        let guid = std::ptr::read_unaligned(inst.add(4) as *const Guid);
        // DIDEVICEINSTANCEW: tszProductName at 560, 260 wide chars
        let w = std::slice::from_raw_parts(inst.add(560) as *const u16, 260);
        let n = w.iter().position(|&c| c == 0).unwrap_or(260);
        out.push((guid, String::from_utf16_lossy(&w[..n])));
    }
    1 // DIENUM_CONTINUE
}

unsafe fn data_format() -> (Vec<ObjFormat>, DataFormat) {
    let guids = [&GUID_X, &GUID_Y, &GUID_Z, &GUID_RX, &GUID_RY, &GUID_RZ, &GUID_SLIDER, &GUID_SLIDER];
    let mut objs: Vec<ObjFormat> = guids
        .iter()
        .enumerate()
        .map(|(i, g)| ObjFormat { guid: *g as *const Guid, ofs: (i * 4) as u32, ty: DIDFT_AXIS | DIDFT_ANYINSTANCE | DIDFT_OPTIONAL, flags: DIDOI_ASPECTPOSITION })
        .collect();
    for b in 0..128u32 {
        objs.push(ObjFormat { guid: std::ptr::null(), ofs: 32 + b, ty: DIDFT_BUTTON | DIDFT_ANYINSTANCE | DIDFT_OPTIONAL, flags: 0 });
    }
    let fmt = DataFormat {
        size: std::mem::size_of::<DataFormat>() as u32,
        obj_size: std::mem::size_of::<ObjFormat>() as u32,
        flags: DIDF_ABSAXIS,
        data_size: std::mem::size_of::<JoyState>() as u32,
        num_objs: objs.len() as u32,
        objs: std::ptr::null(),
    };
    (objs, fmt)
}

fn dev_match(name: &str, want: &str) -> bool {
    name.to_lowercase().contains(&want.to_lowercase())
}

impl Wheel {
    /// Every attached controller; force feedback on the first FF device.
    /// `hwnd` = the viewer window (FF needs an exclusive foreground-capable
    /// window), null for a console tool (no FF).
    pub fn open(hwnd: *mut c_void) -> Result<Wheel, String> {
        unsafe {
            let lib = libloading::Library::new("dinput8.dll").map_err(|e| e.to_string())?;
            let create: libloading::Symbol<unsafe extern "system" fn(*mut c_void, u32, *const Guid, *mut *mut c_void, *mut c_void) -> HRESULT> =
                lib.get(b"DirectInput8Create").map_err(|e| e.to_string())?;
            let mut di = std::ptr::null_mut();
            let hr = create(GetModuleHandleW(std::ptr::null()), 0x0800, &IID_IDIRECTINPUT8W, &mut di, std::ptr::null_mut());
            if hr < 0 || di.is_null() {
                return Err(format!("DirectInput8Create {hr:#x}"));
            }
            let mut all: Vec<(Guid, String)> = Vec::new();
            com!(di, 4, fn(u32, extern "system" fn(*const u8, *mut c_void) -> i32, *mut c_void, u32),
                DI8DEVCLASS_GAMECTRL, enum_all, &mut all as *mut _ as *mut c_void, DIEDFL_ATTACHEDONLY);
            let mut ff: Vec<(Guid, String)> = Vec::new();
            com!(di, 4, fn(u32, extern "system" fn(*const u8, *mut c_void) -> i32, *mut c_void, u32),
                DI8DEVCLASS_GAMECTRL, enum_all, &mut ff as *mut _ as *mut c_void, DIEDFL_ATTACHEDONLY | DIEDFL_FORCEFEEDBACK);
            if all.is_empty() {
                com!(di, 2, fn());
                return Err("no game controller attached".into());
            }
            let (objs, mut fmt) = data_format();
            fmt.objs = objs.as_ptr();
            let mut w = Wheel {
                _lib: lib,
                di,
                devs: Vec::new(),
                ffb: None,
                name: String::new(),
                map: Mapping::load().unwrap_or_default(),
                prev_buttons: Vec::new(),
                last_log: std::time::Instant::now(),
                damper_on: false,
            };
            let window = if hwnd.is_null() { GetDesktopWindow() } else { hwnd };
            for (guid, name) in all {
                let mut dev = std::ptr::null_mut();
                if com!(di, 3, fn(*const Guid, *mut *mut c_void, *mut c_void), &guid, &mut dev, std::ptr::null_mut()) < 0 || dev.is_null() {
                    continue;
                }
                if com!(dev, 11, fn(*const DataFormat), &fmt) < 0 {
                    com!(dev, 2, fn());
                    continue;
                }
                let is_ff = !hwnd.is_null() && w.ffb.is_none() && ff.iter().any(|f| f.0 == guid);
                let coop = if is_ff { DISCL_EXCLUSIVE | DISCL_BACKGROUND } else { DISCL_NONEXCLUSIVE | DISCL_BACKGROUND };
                if com!(dev, 13, fn(*mut c_void, u32), window, coop) < 0 {
                    com!(dev, 2, fn());
                    continue;
                }
                let range = PropRange { size: 24, header_size: 16, obj: 0, how: DIPH_DEVICE, min: -10000, max: 10000 };
                com!(dev, 6, fn(usize, *const c_void), DIPROP_RANGE, &range as *const _ as *const c_void);
                let mut d = Device { name, dev, state: JoyState::default(), force: std::ptr::null_mut(), damper: std::ptr::null_mut() };
                if is_ff {
                    let ac = PropDword { size: 20, header_size: 16, obj: 0, how: DIPH_DEVICE, data: 0 };
                    com!(dev, 6, fn(usize, *const c_void), DIPROP_AUTOCENTER, &ac as *const _ as *const c_void);
                }
                com!(dev, 7, fn());
                if is_ff && Self::effects(&mut d).is_ok() {
                    w.ffb = Some(w.devs.len());
                    w.name = d.name.clone();
                }
                w.devs.push(d);
            }
            if w.name.is_empty() {
                w.name = w.devs.first().map(|d| d.name.clone()).unwrap_or_default();
            }
            // the cockpit aids and pit roles default to the game's own buttons (wheel.cfg wins)
            for (role, b) in game_binds() {
                if let Bind::Button { dev, button } = &b {
                    if w.map.get(&role).is_none() && w.find(dev).is_some() {
                        println!("wheel: {role} = {dev} button {button} (from the game's controls)");
                        w.map.binds.push((role, b));
                    }
                }
            }
            w.prev_buttons = vec![[0; 128]; w.devs.len()];
            Ok(w)
        }
    }

    unsafe fn effects(d: &mut Device) -> Result<(), String> {
        let axes = [0u32]; // DIJOFS_X
        let dir = [0i32];
        let magnitude = 0i32;
        let eff = Effect {
            size: std::mem::size_of::<Effect>() as u32,
            flags: DIEFF_CARTESIAN | DIEFF_OBJECTOFFSETS,
            duration: INFINITE,
            sample_period: 0,
            gain: 10000,
            trigger_button: DIEB_NOTRIGGER,
            trigger_repeat: 0,
            axes_n: 1,
            axes: axes.as_ptr(),
            direction: dir.as_ptr(),
            envelope: std::ptr::null(),
            type_size: 4,
            type_params: &magnitude as *const i32 as *const c_void,
            start_delay: 0,
        };
        let hr = com!(d.dev, 18, fn(*const Guid, *const Effect, *mut *mut c_void, *mut c_void), &GUID_CONSTANT_FORCE, &eff, &mut d.force, std::ptr::null_mut());
        if hr < 0 || d.force.is_null() {
            return Err(format!("CreateEffect constant force {hr:#x}"));
        }
        com!(d.force, 7, fn(u32, u32), 1, 0);
        let cond = Condition { offset: 0, pos_coef: 0, neg_coef: 0, pos_sat: 10000, neg_sat: 10000, dead_band: 0 };
        let deff = Effect { type_size: std::mem::size_of::<Condition>() as u32, type_params: &cond as *const _ as *const c_void, ..eff };
        com!(d.dev, 18, fn(*const Guid, *const Effect, *mut *mut c_void, *mut c_void), &GUID_DAMPER, &deff, &mut d.damper, std::ptr::null_mut());
        Ok(())
    }

    pub fn devices(&self) -> Vec<String> {
        self.devs.iter().map(|d| d.name.clone()).collect()
    }

    fn poll(&mut self) {
        for (i, d) in self.devs.iter_mut().enumerate() {
            self.prev_buttons[i] = d.state.buttons;
            unsafe {
                com!(d.dev, 25, fn());
                if com!(d.dev, 9, fn(u32, *mut JoyState), std::mem::size_of::<JoyState>() as u32, &mut d.state) < 0 {
                    com!(d.dev, 7, fn());
                }
            }
        }
    }

    fn find(&self, dev: &str) -> Option<usize> {
        self.devs.iter().position(|d| dev_match(&d.name, dev))
    }

    fn axis(&self, role: &str) -> Option<f32> {
        let Some(Bind::Axis { dev, axis, rest, full }) = self.map.get(role) else { return None };
        let d = &self.devs[self.find(dev)?];
        let v = d.state.axes[*axis] as f32;
        Some(((v - *rest as f32) / (*full as f32 - *rest as f32)).clamp(0.0, 1.0))
    }

    fn button(&self, role: &str) -> Option<(bool, bool)> {
        let Some(Bind::Button { dev, button }) = self.map.get(role) else { return None };
        let i = self.find(dev)?;
        let now = self.devs[i].state.buttons[*button] & 0x80 != 0;
        let was = self.prev_buttons[i][*button] & 0x80 != 0;
        Some((now, now && !was))
    }

    /// Poll every device and turn the mapping into driving inputs.
    pub fn read(&mut self) -> Option<WheelInput> {
        self.poll();
        if self.devs.is_empty() {
            return None;
        }
        if crate::envc::var_os("EVOVIEW_WHEEL_LOG").is_some() && self.last_log.elapsed().as_secs_f32() > 1.0 {
            self.last_log = std::time::Instant::now();
            for d in &self.devs {
                let pressed: Vec<usize> = (0..128).filter(|b| d.state.buttons[*b] & 0x80 != 0).collect();
                println!("WHEEL {:24} axes {:?} buttons {:?}", d.name, d.state.axes, pressed);
            }
        }
        let mut wi = WheelInput::default();
        // steering: the mapped axis (rest = full left, full = full right), or
        // X on the FF base
        wi.steer = match self.map.get("steer") {
            Some(Bind::Axis { .. }) => -(self.axis("steer").unwrap_or(0.5) * 2.0 - 1.0),
            _ => {
                let d = &self.devs[self.ffb.unwrap_or(0)];
                -(d.state.axes[0] as f32 / 10000.0).clamp(-1.0, 1.0)
            }
        };
        let pedal_default = |w: &Wheel, axis: usize| {
            let d = &w.devs[w.ffb.unwrap_or(0)];
            ((1.0 - d.state.axes[axis] as f32 / 10000.0) * 0.5).clamp(0.0, 1.0)
        };
        wi.throttle = self.axis("throttle").unwrap_or_else(|| pedal_default(self, 1));
        wi.brake = self.axis("brake").unwrap_or_else(|| pedal_default(self, 5));
        wi.clutch = self.axis("clutch");
        wi.handbrake = self.axis("handbrake").or_else(|| self.button("handbrake").map(|b| if b.0 { 1.0 } else { 0.0 }));
        wi.shift_up = self.button("shift_up").map_or(false, |b| b.1);
        wi.shift_down = self.button("shift_down").map_or(false, |b| b.1);
        wi.starter = self.button("starter").map_or(false, |b| b.1);
        for (i, k) in ["bias", "tc", "abs"].iter().enumerate() {
            let hit = |r: String| self.button(&r).map_or(false, |b| b.1);
            wi.aids[i] = hit(format!("{k}_up")) as i32 - hit(format!("{k}_down")) as i32;
        }
        for (i, r) in PIT_ROLES.iter().enumerate() {
            wi.pit[i] = self.button(r).map_or(false, |b| b.1);
        }
        let gates = ["h1", "h2", "h3", "h4", "h5", "h6", "h7"];
        if gates.iter().any(|g| self.map.get(g).is_some()) || self.map.get("hr").is_some() {
            let mut g = 0;
            for (i, r) in gates.iter().enumerate() {
                if self.button(r).map_or(false, |b| b.0) {
                    g = i as i32 + 1;
                }
            }
            if self.button("hr").map_or(false, |b| b.0) {
                g = -1;
            }
            wi.h_gear = Some(g);
        }
        Some(wi)
    }

    /// Send one step's force (-1..1) and damper (0..1) to the FF device.
    pub fn send(&mut self, force: f32, damper: f32) {
        let Some(i) = self.ffb else { return };
        let d = &self.devs[i];
        unsafe {
            let magnitude = (force * 10000.0).clamp(-10000.0, 10000.0) as i32;
            let axes = [0u32];
            let dir = [0i32];
            let mut eff = Effect {
                size: std::mem::size_of::<Effect>() as u32,
                flags: DIEFF_CARTESIAN | DIEFF_OBJECTOFFSETS,
                duration: INFINITE,
                sample_period: 0,
                gain: 10000,
                trigger_button: DIEB_NOTRIGGER,
                trigger_repeat: 0,
                axes_n: 1,
                axes: axes.as_ptr(),
                direction: dir.as_ptr(),
                envelope: std::ptr::null(),
                type_size: 4,
                type_params: &magnitude as *const i32 as *const c_void,
                start_delay: 0,
            };
            com!(d.force, 6, fn(*const Effect, u32), &eff, DIEP_TYPESPECIFICPARAMS | DIEP_START);
            if !d.damper.is_null() {
                let c = (damper.clamp(0.0, 1.0) * 10000.0) as i32;
                let cond = Condition { offset: 0, pos_coef: c, neg_coef: c, pos_sat: 10000, neg_sat: 10000, dead_band: 0 };
                eff.type_size = std::mem::size_of::<Condition>() as u32;
                eff.type_params = &cond as *const _ as *const c_void;
                let flags = if self.damper_on { DIEP_TYPESPECIFICPARAMS } else { DIEP_TYPESPECIFICPARAMS | DIEP_AXES | DIEP_DIRECTION | DIEP_START };
                com!(d.damper, 6, fn(*const Effect, u32), &eff, flags);
                self.damper_on = true;
            }
        }
    }
}

impl Drop for Wheel {
    fn drop(&mut self) {
        unsafe {
            for d in &self.devs {
                for e in [d.force, d.damper] {
                    if !e.is_null() {
                        com!(e, 8, fn());
                        com!(e, 2, fn());
                    }
                }
                com!(d.dev, 8, fn());
                com!(d.dev, 2, fn());
            }
            com!(self.di, 2, fn());
        }
    }
}

// ---- `--wheel-setup`: ask for each control, detect it, save the mapping

fn wait_enter(prompt: &str) {
    use std::io::Write;
    print!("{prompt}");
    let _ = std::io::stdout().flush();
    // a terminal can deliver one Enter as two line breaks: a line that
    // arrives within 300 ms of the prompt is that echo, not the answer
    let shown = std::time::Instant::now();
    loop {
        let mut s = String::new();
        match std::io::stdin().read_line(&mut s) {
            Ok(0) | Err(_) => return,
            _ if shown.elapsed().as_millis() < 300 => continue,
            _ => return,
        }
    }
}

/// Snapshot every device after a short poll.
fn snapshot(w: &mut Wheel) -> Vec<JoyState> {
    for _ in 0..3 {
        w.poll();
        std::thread::sleep(std::time::Duration::from_millis(30));
    }
    w.devs.iter().map(|d| d.state).collect()
}

/// The axis that moved most between two snapshots: (device, axis, from, to).
fn moved_axis(a: &[JoyState], b: &[JoyState]) -> Option<(usize, usize, i32, i32)> {
    let mut best: Option<(usize, usize, i32, i32)> = None;
    for d in 0..a.len().min(b.len()) {
        for ax in 0..8 {
            let (x, y) = (a[d].axes[ax], b[d].axes[ax]);
            if (y - x).abs() > 4000 && best.map_or(true, |bb| (y - x).abs() > (bb.3 - bb.2).abs()) {
                best = Some((d, ax, x, y));
            }
        }
    }
    best
}

/// A button that is down in `b` and was not in `a`: (device, button).
fn new_button(a: &[JoyState], b: &[JoyState]) -> Option<(usize, usize)> {
    for d in 0..a.len().min(b.len()) {
        for k in 0..128 {
            if b[d].buttons[k] & 0x80 != 0 && a[d].buttons[k] & 0x80 == 0 {
                return Some((d, k));
            }
        }
    }
    None
}

/// `only`: redo just these roles (e.g. ["throttle"]) and keep the rest of
/// the saved mapping; empty = the full walk-through.
pub fn setup_wizard(only: &[String]) {
    let mut w = match Wheel::open(std::ptr::null_mut()) {
        Ok(w) => w,
        Err(e) => {
            println!("wheel setup: {e}");
            return;
        }
    };
    println!("evoview wheel setup. Devices found:");
    for d in w.devices() {
        println!("  - {d}");
    }
    println!("Each step: put the control at rest, press Enter, then move it and press Enter again.");
    println!("To skip a control (e.g. no handbrake), leave it at rest for both Enters.\n");
    let mut m = if only.is_empty() { Mapping::default() } else { Mapping::load().unwrap_or_default() };
    let want = |role: &str| only.is_empty() || only.iter().any(|o| o == role);
    let name = |w: &Wheel, d: usize| w.devs[d].name.clone();
    for (role, rest_msg, move_msg) in [
        ("steer", "Turn the wheel FULLY LEFT and hold it", "Now turn it FULLY RIGHT and hold it"),
        ("throttle", "Throttle released", "Throttle pressed FULLY"),
        ("brake", "Brake released", "Brake pressed FULLY"),
        ("clutch", "Clutch released", "Clutch pressed FULLY"),
        ("handbrake", "Handbrake released (axis handbrake; a button one comes later)", "Handbrake pulled FULLY"),
    ] {
        if !want(role) {
            continue;
        }
        m.binds.retain(|(r, _)| r != role);
        wait_enter(&format!("[{role}] {rest_msg}, then Enter... "));
        let a = snapshot(&mut w);
        wait_enter(&format!("[{role}] {move_msg}, then Enter... "));
        let b = snapshot(&mut w);
        match moved_axis(&a, &b) {
            Some((d, ax, from, to)) => {
                println!("    -> {} axis {ax} ({from} .. {to})", name(&w, d));
                m.binds.push((role.into(), Bind::Axis { dev: name(&w, d), axis: ax, rest: from, full: to }));
            }
            None => {
                println!("    -> nothing moved past the threshold, skipped. Every axis, rest -> pressed:");
                for d in 0..a.len().min(b.len()) {
                    let moves: Vec<String> = (0..8).map(|k| format!("{k}: {} -> {}", a[d].axes[k], b[d].axes[k])).collect();
                    println!("       {}: {}", name(&w, d), moves.join(", "));
                }
            }
        }
    }
    for (role, msg) in [
        ("shift_up", "Pull and HOLD the UPSHIFT paddle (right)"),
        ("shift_down", "Pull and HOLD the DOWNSHIFT paddle (left)"),
        ("h1", "H shifter into 1st and hold"),
        ("h2", "H shifter into 2nd"),
        ("h3", "H shifter into 3rd"),
        ("h4", "H shifter into 4th"),
        ("h5", "H shifter into 5th"),
        ("h6", "H shifter into 6th"),
        ("h7", "H shifter into 7th (skip if it has none)"),
        ("hr", "H shifter into REVERSE"),
        ("starter", "Press and hold the button for the STARTER (ignition)"),
        ("handbrake", "Press and hold a HANDBRAKE button (skip if done above)"),
        ("bias_up", "Press and hold the button for BRAKE BIAS forward (skip: none)"),
        ("bias_down", "Press and hold the button for BRAKE BIAS rearward"),
        ("tc_up", "Press and hold the button for TC up"),
        ("tc_down", "Press and hold the button for TC down"),
        ("abs_up", "Press and hold the button for ABS up"),
        ("abs_down", "Press and hold the button for ABS down"),
        ("pit_limiter", "Press and hold the PIT LIMITER button (skip: the game's own)"),
        ("pit_menu", "Press and hold a PIT MENU button (open / close; skip for none)"),
        ("pit_up", "Press and hold PIT MENU UP (skip: the game's TC+)"),
        ("pit_down", "Press and hold PIT MENU DOWN (skip: the game's TC-)"),
        ("pit_left", "Press and hold PIT MENU LEFT / less (skip: the game's ABS-)"),
        ("pit_right", "Press and hold PIT MENU RIGHT / more (skip: the game's ABS+)"),
        ("pit_ok", "Press and hold PIT MENU OK (skip: the limiter button, when stopped)"),
    ] {
        if !want(role) || (role == "handbrake" && matches!(m.get("handbrake"), Some(Bind::Axis { .. }))) {
            continue;
        }
        m.binds.retain(|(r, _)| r != role);
        wait_enter(&format!("[{role}] release everything (shifter in neutral), Enter... "));
        let a = snapshot(&mut w);
        wait_enter(&format!("[{role}] {msg}, Enter... "));
        let b = snapshot(&mut w);
        match new_button(&a, &b) {
            Some((d, k)) => {
                println!("    -> {} button {k}", name(&w, d));
                m.binds.push((role.into(), Bind::Button { dev: name(&w, d), button: k }));
            }
            None => println!("    -> no new button, skipped"),
        }
    }
    match m.save() {
        Ok(p) => println!("\nsaved {}", p.display()),
        Err(e) => println!("\ncould not save: {e}"),
    }
}
