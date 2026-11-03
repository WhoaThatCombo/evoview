//! Playing car audio through the game's own FMOD.
//!
//! Reimplementing the mix by hand does not work: the banks are only the raw
//! material, and what makes them sound like the game is FMOD Studio's event
//! graph - separate interior and exterior engine layers, filters, curves,
//! distance attenuation and spatialisation, with turbo, transmission, backfire
//! and wheel banks layered on top. The game's `AudioData` schema lists those as
//! separate systems.
//!
//! So this does not reimplement anything. It loads `fmod.dll` and
//! `fmodstudio.dll` out of the user's own game folder, hands FMOD the same
//! banks the game uses, and drives the same event parameters from the replay.
//! Nothing of FMOD's is shipped with evoview - it is loaded from the install
//! that already has it.
//!
//! ⚠ Bank load ORDER matters: the master bank and its strings bank first, then
//! the car. FMOD cannot resolve an event path without the strings bank.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::path::Path;

use libloading::{Library, Symbol};

/// 2.3.13, as the digits appear: 0x0002_03_13. `System_Create` rejects a
/// header version that does not match the library.
const FMOD_VERSION: c_uint = 0x0002_0313;

type Sys = *mut c_void;
type Bank = *mut c_void;
type Desc = *mut c_void;
pub type Inst = *mut c_void;

/// FMOD_3D_ATTRIBUTES: position, velocity, forward, up. FMOD is left-handed
/// (+X right, +Y up, +Z forward); callers pass the viewer's right-handed
/// vectors through `lh`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Attr3 {
    pos: [f32; 3],
    vel: [f32; 3],
    fwd: [f32; 3],
    up: [f32; 3],
}

fn lh(v: [f32; 3]) -> [f32; 3] {
    [v[0], v[1], -v[2]]
}

pub struct Fmod {
    // Kept alive: unloading the libraries would invalidate every pointer.
    _core: Library,
    studio: Library,
    system: Sys,
    inst: Option<Inst>,
    desc: Option<Desc>,
    /// Banks stay loaded for the life of the process; the car bank is swapped.
    car_bank: Option<Bank>,
    pub event: String,
    pub params: Vec<String>,
}

macro_rules! sym {
    ($lib:expr, $name:literal, $t:ty) => {
        unsafe {
            let s: Symbol<$t> = $lib
                .get($name)
                .map_err(|e| format!("{}: {e}", String::from_utf8_lossy($name)))?;
            *s
        }
    };
}

/// FMOD_STUDIO_PARAMETER_DESCRIPTION. Confirmed correct for 2.3.13 by reading
/// a known event's ranges back and getting sensible numbers (gear 0-8,
/// speed 0-500) rather than garbage.
#[repr(C)]
struct ParamDesc {
    name: *const c_char,
    id1: c_uint,
    id2: c_uint,
    minimum: f32,
    maximum: f32,
    defaultvalue: f32,
    kind: c_int,
    flags: c_uint,
    guid: [u8; 16],
}

fn ck(r: c_int, what: &str) -> Result<(), String> {
    if r == 0 { Ok(()) } else { Err(format!("{what} failed: FMOD_RESULT {r}")) }
}

impl Fmod {
    /// `dir` is the game folder holding fmod.dll / fmodstudio.dll.
    pub fn new(dir: &Path) -> Result<Fmod, String> {
        Self::new_opts(dir, None)
    }

    /// `wav`: render offline into that file instead of the sound card
    /// (FMOD_OUTPUTTYPE_WAVWRITER_NRT; every `update` mixes one block), for
    /// checking a mix without speakers.
    pub fn new_opts(dir: &Path, wav: Option<&Path>) -> Result<Fmod, String> {
        // Load the core by absolute path FIRST: fmodstudio.dll depends on it,
        // and this way the loader finds the copy we mean rather than searching.
        let core = unsafe { Library::new(dir.join("fmod.dll")) }
            .map_err(|e| format!("fmod.dll: {e}"))?;
        let studio = unsafe { Library::new(dir.join("fmodstudio.dll")) }
            .map_err(|e| format!("fmodstudio.dll: {e}"))?;

        let create = sym!(studio, b"FMOD_Studio_System_Create",
                          unsafe extern "C" fn(*mut Sys, c_uint) -> c_int);
        let init = sym!(studio, b"FMOD_Studio_System_Initialize",
                        unsafe extern "C" fn(Sys, c_int, c_uint, c_uint, *mut c_void) -> c_int);
        let mut system: Sys = std::ptr::null_mut();
        ck(unsafe { create(&mut system, FMOD_VERSION) }, "System_Create")?;
        let wav_c = wav.and_then(|p| CString::new(p.to_string_lossy().as_ref()).ok());
        if wav_c.is_some() {
            let core_of = sym!(studio, b"FMOD_Studio_System_GetCoreSystem",
                               unsafe extern "C" fn(Sys, *mut Sys) -> c_int);
            let set_out = sym!(core, b"FMOD_System_SetOutput", unsafe extern "C" fn(Sys, c_int) -> c_int);
            let mut cs: Sys = std::ptr::null_mut();
            ck(unsafe { core_of(system, &mut cs) }, "GetCoreSystem")?;
            // 5 = FMOD_OUTPUTTYPE_WAVWRITER_NRT
            ck(unsafe { set_out(cs, 5) }, "SetOutput")?;
        }
        // 512 virtual channels; offline: synchronous studio updates (1) and
        // the core mixing from update (FMOD_INIT_MIX_FROM_UPDATE 2)
        let (sf, cf, extra) = match wav_c.as_ref() {
            Some(c) => (1, 2, c.as_ptr() as *mut c_void),
            None => (0, 0, std::ptr::null_mut()),
        };
        ck(unsafe { init(system, 512, sf, cf, extra) }, "System_Initialize")?;

        // ⚠ The banks reference the game's own DSP plugins, and loading one
        // without them fails with FMOD_ERR_PLUGIN_MISSING (54). They have to be
        // registered on the CORE system before any bank is loaded.
        let core_of = sym!(studio, b"FMOD_Studio_System_GetCoreSystem",
                           unsafe extern "C" fn(Sys, *mut Sys) -> c_int);
        let mut coresys: Sys = std::ptr::null_mut();
        ck(unsafe { core_of(system, &mut coresys) }, "GetCoreSystem")?;
        let load_plugin = sym!(core, b"FMOD_System_LoadPlugin",
                               unsafe extern "C" fn(Sys, *const c_char, *mut c_uint, c_uint) -> c_int);
        for name in ["fmod_distance_filter.dll", "fmod_gain.dll", "resonanceaudio.dll"] {
            let p = dir.join(name);
            if !p.is_file() {
                continue;
            }
            if let Ok(c) = CString::new(p.to_string_lossy().as_ref()) {
                let mut h: c_uint = 0;
                let r = unsafe { load_plugin(coresys, c.as_ptr(), &mut h, 0) };
                if r != 0 {
                    eprintln!("FMOD: plugin {name} did not load (result {r})");
                }
            }
        }

        Ok(Fmod {
            _core: core,
            studio,
            system,
            inst: None,
            desc: None,
            car_bank: None,
            event: String::new(),
            params: Vec::new(),
        })
    }

    fn load_bank(&self, data: &[u8]) -> Result<Bank, String> {
        let f = sym!(self.studio, b"FMOD_Studio_System_LoadBankMemory",
                     unsafe extern "C" fn(Sys, *const u8, c_int, c_int, c_uint, *mut Bank) -> c_int);
        let mut b: Bank = std::ptr::null_mut();
        // mode 0 = FMOD_STUDIO_LOAD_MEMORY (FMOD copies, so no alignment rule),
        // flags 0 = FMOD_STUDIO_LOAD_BANK_NORMAL (blocking).
        ck(unsafe { f(self.system, data.as_ptr(), data.len() as c_int, 0, 0, &mut b) },
           "LoadBankMemory")?;
        Ok(b)
    }

    /// Load one more bank (the shared ones: wheel, turbo, environment ...);
    /// returns its event paths. Banks stay loaded.
    pub fn load_extra(&mut self, data: &[u8]) -> Result<Vec<String>, String> {
        let b = self.load_bank(data)?;
        self.events(b)
    }

    /// Every parameter an event declares: (name, min, max, default), and
    /// whether the event is 3D. Uses the 2.3.13 description layout
    /// (verified: ranges read back sane, see ParamDesc).
    pub fn event_info(&self, event: &str) -> Option<(Vec<(String, f32, f32, f32)>, bool)> {
        let get: Symbol<unsafe extern "C" fn(Sys, *const c_char, *mut Desc) -> c_int> =
            unsafe { self.studio.get(b"FMOD_Studio_System_GetEvent") }.ok()?;
        let count: Symbol<unsafe extern "C" fn(Desc, *mut c_int) -> c_int> =
            unsafe { self.studio.get(b"FMOD_Studio_EventDescription_GetParameterDescriptionCount") }.ok()?;
        let byidx: Symbol<unsafe extern "C" fn(Desc, c_int, *mut ParamDesc) -> c_int> =
            unsafe { self.studio.get(b"FMOD_Studio_EventDescription_GetParameterDescriptionByIndex") }.ok()?;
        let is3d: Symbol<unsafe extern "C" fn(Desc, *mut c_int) -> c_int> =
            unsafe { self.studio.get(b"FMOD_Studio_EventDescription_Is3D") }.ok()?;
        let c = CString::new(event).ok()?;
        let mut d: Desc = std::ptr::null_mut();
        if unsafe { get(self.system, c.as_ptr(), &mut d) } != 0 {
            return None;
        }
        let mut n: c_int = 0;
        unsafe { count(d, &mut n) };
        let mut out = Vec::new();
        for i in 0..n {
            let mut pd: ParamDesc = unsafe { std::mem::zeroed() };
            if unsafe { byidx(d, i, &mut pd) } == 0 && !pd.name.is_null() {
                let nm = unsafe { CStr::from_ptr(pd.name) }.to_string_lossy().into_owned();
                out.push((nm, pd.minimum, pd.maximum, pd.defaultvalue));
            }
        }
        let mut t: c_int = 0;
        unsafe { is3d(d, &mut t) };
        Some((out, t != 0))
    }

    /// Load master + strings once, then a car bank. Returns the event paths the
    /// car bank provides.
    pub fn load_banks(&mut self, master: &[u8], strings: &[u8], car: &[u8])
                      -> Result<Vec<String>, String> {
        if self.car_bank.is_none() {
            self.load_bank(master)?;
            self.load_bank(strings)?;
        }
        let b = self.load_bank(car)?;
        self.car_bank = Some(b);
        self.events(b)
    }

    fn events(&self, bank: Bank) -> Result<Vec<String>, String> {
        let count = sym!(self.studio, b"FMOD_Studio_Bank_GetEventCount",
                         unsafe extern "C" fn(Bank, *mut c_int) -> c_int);
        let list = sym!(self.studio, b"FMOD_Studio_Bank_GetEventList",
                        unsafe extern "C" fn(Bank, *mut Desc, c_int, *mut c_int) -> c_int);
        let path = sym!(self.studio, b"FMOD_Studio_EventDescription_GetPath",
                        unsafe extern "C" fn(Desc, *mut c_char, c_int, *mut c_int) -> c_int);
        let mut n: c_int = 0;
        ck(unsafe { count(bank, &mut n) }, "GetEventCount")?;
        let mut v: Vec<Desc> = vec![std::ptr::null_mut(); n.max(0) as usize];
        let mut got: c_int = 0;
        if n > 0 {
            ck(unsafe { list(bank, v.as_mut_ptr(), n, &mut got) }, "GetEventList")?;
        }
        let mut out = Vec::new();
        let mut buf = vec![0i8; 512];
        for d in v.iter().take(got.max(0) as usize) {
            let mut r: c_int = 0;
            if unsafe { path(*d, buf.as_mut_ptr() as *mut c_char, 512, &mut r) } == 0 {
                let s = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
                out.push(s.to_string_lossy().into_owned());
            }
        }
        Ok(out)
    }

    /// Start one event by path and remember which parameters it accepts.
    ///
    /// Parameter names are discovered by TRYING them rather than by reading
    /// FMOD's parameter-description struct: that struct's layout changes
    /// between FMOD versions, and a wrong guess there is a silent memory bug,
    /// whereas a wrong name here is just a non-zero result code.
    pub fn play(&mut self, event: &str, candidates: &[&str]) -> Result<(), String> {
        let get = sym!(self.studio, b"FMOD_Studio_System_GetEvent",
                       unsafe extern "C" fn(Sys, *const c_char, *mut Desc) -> c_int);
        let mk = sym!(self.studio, b"FMOD_Studio_EventDescription_CreateInstance",
                      unsafe extern "C" fn(Desc, *mut Inst) -> c_int);
        let start = sym!(self.studio, b"FMOD_Studio_EventInstance_Start",
                         unsafe extern "C" fn(Inst) -> c_int);
        let setp = sym!(self.studio, b"FMOD_Studio_EventInstance_SetParameterByName",
                        unsafe extern "C" fn(Inst, *const c_char, f32, c_int) -> c_int);
        let c = CString::new(event).map_err(|e| e.to_string())?;
        let mut d: Desc = std::ptr::null_mut();
        ck(unsafe { get(self.system, c.as_ptr(), &mut d) }, "GetEvent")?;
        let mut i: Inst = std::ptr::null_mut();
        ck(unsafe { mk(d, &mut i) }, "CreateInstance")?;
        ck(unsafe { start(i) }, "EventInstance_Start")?;
        self.params.clear();
        for name in candidates {
            if let Ok(cn) = CString::new(*name) {
                if unsafe { setp(i, cn.as_ptr(), 0.0, 1) } == 0 {
                    self.params.push((*name).to_string());
                }
            }
        }
        self.inst = Some(i);
        self.desc = Some(d);
        self.event = event.to_string();
        Ok(())
    }

    /// Seconds of audio one `update` mixes in offline (WAV writer) mode:
    /// the core's DSP block length over its sample rate.
    pub fn block_seconds(&self) -> f32 {
        let (Ok(core_of), Ok(bs), Ok(fmt)) = (
            unsafe { self.studio.get::<unsafe extern "C" fn(Sys, *mut Sys) -> c_int>(b"FMOD_Studio_System_GetCoreSystem") },
            unsafe { self._core.get::<unsafe extern "C" fn(Sys, *mut c_uint, *mut c_int) -> c_int>(b"FMOD_System_GetDSPBufferSize") },
            unsafe { self._core.get::<unsafe extern "C" fn(Sys, *mut c_int, *mut c_int, *mut c_int) -> c_int>(b"FMOD_System_GetSoftwareFormat") },
        ) else { return 1024.0 / 48000.0 };
        let mut cs: Sys = std::ptr::null_mut();
        let (mut len, mut num, mut rate, mut mode, mut raw) = (0u32, 0i32, 0i32, 0i32, 0i32);
        unsafe {
            core_of(self.system, &mut cs);
            bs(cs, &mut len, &mut num);
            fmt(cs, &mut rate, &mut mode, &mut raw);
        }
        if len == 0 || rate == 0 { 1024.0 / 48000.0 } else { len as f32 / rate as f32 }
    }

    /// Shut FMOD down (finalises a WAV writer's header). The Fmod must not
    /// be used afterwards.
    pub fn release(&self) {
        if let Ok(f) = unsafe { self.studio.get::<unsafe extern "C" fn(Sys) -> c_int>(b"FMOD_Studio_System_Release") } {
            unsafe { f(self.system) };
        }
    }

    /// Start an event and hand its instance back (several can play at once:
    /// engine, tyres, wind, ambience ...).
    pub fn spawn(&self, event: &str) -> Option<Inst> {
        let get: Symbol<unsafe extern "C" fn(Sys, *const c_char, *mut Desc) -> c_int> =
            unsafe { self.studio.get(b"FMOD_Studio_System_GetEvent") }.ok()?;
        let mk: Symbol<unsafe extern "C" fn(Desc, *mut Inst) -> c_int> =
            unsafe { self.studio.get(b"FMOD_Studio_EventDescription_CreateInstance") }.ok()?;
        let start: Symbol<unsafe extern "C" fn(Inst) -> c_int> =
            unsafe { self.studio.get(b"FMOD_Studio_EventInstance_Start") }.ok()?;
        let c = CString::new(event).ok()?;
        let mut d: Desc = std::ptr::null_mut();
        if unsafe { get(self.system, c.as_ptr(), &mut d) } != 0 {
            return None;
        }
        let mut i: Inst = std::ptr::null_mut();
        if unsafe { mk(d, &mut i) } != 0 || unsafe { start(i) } != 0 {
            return None;
        }
        Some(i)
    }

    /// Set a parameter on one instance; false if the event has no such name.
    pub fn pset(&self, i: Inst, name: &str, value: f32) -> bool {
        let (Ok(f), Ok(c)) = (unsafe { self.studio.get::<unsafe extern "C" fn(Inst, *const c_char, f32, c_int) -> c_int>(
            b"FMOD_Studio_EventInstance_SetParameterByName") }, CString::new(name)) else { return false };
        unsafe { f(i, c.as_ptr(), value, 0) == 0 }
    }

    pub fn pvol(&self, i: Inst, v: f32) {
        if let Ok(f) = unsafe { self.studio.get::<unsafe extern "C" fn(Inst, f32) -> c_int>(b"FMOD_Studio_EventInstance_SetVolume") } {
            unsafe { f(i, v) };
        }
    }

    /// Place an instance in the world (right-handed viewer space, metres).
    pub fn p3d(&self, i: Inst, pos: [f32; 3], vel: [f32; 3], fwd: [f32; 3], up: [f32; 3]) {
        if let Ok(f) = unsafe { self.studio.get::<unsafe extern "C" fn(Inst, *const Attr3) -> c_int>(b"FMOD_Studio_EventInstance_Set3DAttributes") } {
            let a = Attr3 { pos: lh(pos), vel: lh(vel), fwd: lh(fwd), up: lh(up) };
            unsafe { f(i, &a) };
        }
    }

    /// The listener (the camera).
    pub fn listener(&self, pos: [f32; 3], vel: [f32; 3], fwd: [f32; 3], up: [f32; 3]) {
        if let Ok(f) = unsafe { self.studio.get::<unsafe extern "C" fn(Sys, c_int, *const Attr3, *const [f32; 3]) -> c_int>(
            b"FMOD_Studio_System_SetListenerAttributes") } {
            let a = Attr3 { pos: lh(pos), vel: lh(vel), fwd: lh(fwd), up: lh(up) };
            unsafe { f(self.system, 0, &a, std::ptr::null()) };
        }
    }

    pub fn stop(&self, i: Inst) {
        if let Ok(f) = unsafe { self.studio.get::<unsafe extern "C" fn(Inst, c_int) -> c_int>(b"FMOD_Studio_EventInstance_Stop") } {
            // 1 = FMOD_STUDIO_STOP_IMMEDIATE
            unsafe { f(i, 1) };
        }
    }

    /// Stop letting the event's own release play out (an AI car leaving
    /// earshot), then release it: FMOD frees it once it has stopped.
    pub fn stop_release(&self, i: Inst) {
        if let Ok(f) = unsafe { self.studio.get::<unsafe extern "C" fn(Inst, c_int) -> c_int>(b"FMOD_Studio_EventInstance_Stop") } {
            // 0 = FMOD_STUDIO_STOP_ALLOWFADEOUT
            unsafe { f(i, 0) };
        }
        if let Ok(f) = unsafe { self.studio.get::<unsafe extern "C" fn(Inst) -> c_int>(b"FMOD_Studio_EventInstance_Release") } {
            unsafe { f(i) };
        }
    }

    /// (min, max, default) for one parameter of the playing event.
    ///
    /// ⚠ This reads FMOD's parameter-description struct, whose layout is
    /// version-specific. It is only used to REPORT ranges, never to drive
    /// anything, so a layout that drifts shows up as silly numbers in a probe
    /// rather than as corrupted audio.
    pub fn param_range(&self, name: &str) -> Option<(f32, f32, f32)> {
        let d = self.desc?;
        let f: Symbol<unsafe extern "C" fn(Desc, *const c_char, *mut ParamDesc) -> c_int> =
            unsafe { self.studio.get(b"FMOD_Studio_EventDescription_GetParameterDescriptionByName") }
                .ok()?;
        let c = CString::new(name).ok()?;
        let mut pd: ParamDesc = unsafe { std::mem::zeroed() };
        if unsafe { f(d, c.as_ptr(), &mut pd) } != 0 {
            return None;
        }
        Some((pd.minimum, pd.maximum, pd.defaultvalue))
    }

    /// Every GLOBAL parameter the loaded banks define, with its range.
    ///
    /// The engine event itself only exposes gear/speed/limiter/boost - rpm is
    /// not among them - so the things that actually drive the note live at
    /// system scope. The game's AudioData schema calls these
    /// `GlobalParametersEnum`.
    pub fn global_params(&self) -> Vec<(String, f32, f32)> {
        let mut out = Vec::new();
        let (Ok(count), Ok(byidx)) = (
            unsafe { self.studio.get::<unsafe extern "C" fn(Sys, *mut c_int) -> c_int>(
                b"FMOD_Studio_System_GetParameterDescriptionCount") },
            unsafe { self.studio.get::<unsafe extern "C" fn(Sys, c_int, *mut ParamDesc) -> c_int>(
                b"FMOD_Studio_System_GetParameterDescriptionByIndex") },
        ) else { return out };
        let mut n: c_int = 0;
        if unsafe { count(self.system, &mut n) } != 0 {
            return out;
        }
        for i in 0..n {
            let mut pd: ParamDesc = unsafe { std::mem::zeroed() };
            if unsafe { byidx(self.system, i, &mut pd) } == 0 && !pd.name.is_null() {
                let nm = unsafe { CStr::from_ptr(pd.name) }.to_string_lossy().into_owned();
                out.push((nm, pd.minimum, pd.maximum));
            }
        }
        out
    }

    /// Set a global (system-scope) parameter.
    pub fn set_global(&self, name: &str, value: f32) {
        if let (Ok(f), Ok(c)) = (
            unsafe { self.studio.get::<unsafe extern "C" fn(Sys, *const c_char, f32, c_int) -> c_int>(
                b"FMOD_Studio_System_SetParameterByName") },
            CString::new(name),
        ) {
            unsafe { f(self.system, c.as_ptr(), value, 0) };
        }
    }

    pub fn set(&self, name: &str, value: f32) {
        let (Some(i), Ok(f)) = (self.inst, std::panic::catch_unwind(|| {
            let s: Symbol<unsafe extern "C" fn(Inst, *const c_char, f32, c_int) -> c_int> =
                unsafe { self.studio.get(b"FMOD_Studio_EventInstance_SetParameterByName") }
                    .expect("SetParameterByName");
            unsafe { *s }
        })) else { return };
        if let Ok(c) = CString::new(name) {
            unsafe { f(i, c.as_ptr(), value, 0) };
        }
    }

    pub fn set_volume(&self, v: f32) {
        if let Some(i) = self.inst {
            if let Ok(s) = unsafe {
                self.studio.get::<unsafe extern "C" fn(Inst, f32) -> c_int>(
                    b"FMOD_Studio_EventInstance_SetVolume")
            } {
                unsafe { s(i, v) };
            }
        }
    }

    /// FMOD does its mixing on its own thread but needs this called regularly
    /// to process commands and parameter changes.
    pub fn update(&self) {
        if let Ok(s) = unsafe {
            self.studio.get::<unsafe extern "C" fn(Sys) -> c_int>(b"FMOD_Studio_System_Update")
        } {
            unsafe { s(self.system) };
        }
    }
}
