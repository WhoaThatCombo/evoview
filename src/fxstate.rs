//! What the car is DOING, as far as the visual effects care: pedals, engine,
//! speed, light switch, indicators, and the weather on the bodywork.
//!
//! This is deliberately a small, flat struct. The drive / demo mode (look-anim,
//! `carstate.rs`) owns the real car state; when it lands, the adapter is one
//! function that copies its fields in here (see `FxState::from_parts`). Until
//! then the keys and the env vars below drive it.
//!
//! Headless shots read everything from env vars (`FxState::from_env`):
//!
//! | var                 | meaning                                         |
//! |---------------------|-------------------------------------------------|
//! | EVOVIEW_LIGHTS=0..3 | light switch: off / stage 1 (DRL) / 2 / 3       |
//! | EVOVIEW_BRAKE=0..1  | brake pedal                                     |
//! | EVOVIEW_THROTTLE    | throttle pedal 0..1                             |
//! | EVOVIEW_RPM         | engine rpm                                      |
//! | EVOVIEW_SPEED       | road speed, km/h                                |
//! | EVOVIEW_GEAR        | -1 reverse, 0 neutral, 1.. forward              |
//! | EVOVIEW_INDICATOR   | off / left / right / hazard                     |
//! | EVOVIEW_BLINK=0/1   | which half of the blink cycle a still shows     |
//! | EVOVIEW_FOG=1       | rain / fog light                                |
//! | EVOVIEW_DISC_TEMP   | brake disc temperature, deg C (glow above ~600) |
//! | EVOVIEW_EXHAUST_TEMP| exhaust heat 0..1 (mechanics glow)              |
//! | EVOVIEW_DIRT=0..1   | dirt level                                      |
//! | EVOVIEW_WET=0..1    | wetness                                         |
//! | EVOVIEW_NIGHT=1     | night: ambient down, emissive multipliers up    |
//! | EVOVIEW_BACKFIRE=t  | a backfire that started t seconds before the shot |
//! | EVOVIEW_BACKFIRE_TYPE=1..5 | which of the car's backfire types fires     |
//! | EVOVIEW_SMOKE=0..1  | tyre smoke strength (rear tyres)                |
//! | EVOVIEW_DUST=0..1   | dust kicked up (all tyres)                      |
//! | EVOVIEW_BLOOM=0..   | glow strength (0 = off, the default)            |

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Indicator {
    Off,
    Left,
    Right,
    Hazard,
}

/// The lap timing the dash pages bind (laps.rs): ModelCurrentCar
/// current_lap_time_ms, delta_time_ms, predicted_lap_time_ms and
/// low_frequency.{last,best}_laptime_ms / total_lap_count. 0 = no time yet.
#[derive(Clone, Copy, Debug, Default)]
pub struct LapDash {
    pub current_ms: f64,
    pub last_ms: f64,
    pub best_ms: f64,
    pub delta_ms: f64,
    pub predicted_ms: f64,
    /// laps finished
    pub laps: u32,
    /// seconds since the last lap ended (the pages' show-laptime popup)
    pub since_lap: f32,
    pub last_is_best: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct FxState {
    /// lap timing for the dash; None = no timing (the page's zeros)
    pub lap: Option<LapDash>,
    /// the dash's TC / ABS levels (0 = off); None = the page's default
    pub tc_level: Option<f32>,
    pub abs_level: Option<f32>,
    /// front brake share 0..1 for the dash; None = the page's default
    pub brake_bias: Option<f32>,
    /// the four tyres' pressures, bar (FL FR RL RR), when known
    pub tyre_bar: Option<[f32; 4]>,
    /// the pit limiter is on (pits.rs): dash `.pitlimiter`, lights
    /// category 40, LED group 2
    pub pit_limiter: bool,
    pub brake: f32,
    pub throttle: f32,
    pub rpm: f32,
    pub speed_mps: f32,
    pub gear: i32,
    /// 0 = lights off, 1..=3 = the car's main-light stages
    pub lights: u32,
    pub indicator: Indicator,
    pub fog: bool,
    pub dirt: f32,
    pub wet: f32,
    /// falling rain 0..1 (EVOVIEW_RAIN, F5): drops on the glass, wipers,
    /// and everything wet
    pub rain: f32,
    pub night: bool,
    pub disc_temp_c: f32,
    pub exhaust_heat: f32,
    /// seconds since the last backfire started (< 0 = none pending)
    pub backfire_age: f32,
    pub smoke: f32,
    pub dust: f32,
    pub bloom: f32,
    /// wall-clock seconds, drives blinkers and particles
    pub time: f32,
    /// Headless only: force the blink phase (Some(true) = lamps lit).
    pub blink_phase: Option<bool>,
    /// --check: light ONLY this lighting function (FN_*), at the strongest
    /// intensity the car's data gives it.
    pub force_fn: Option<usize>,
    /// which backfire type (1..5, the game's BackfireType1..5) a trigger
    /// fires; None = pick by the car's visual probabilities
    pub backfire_type: Option<u32>,
}

impl Default for FxState {
    fn default() -> Self {
        Self {
            lap: None,
            tc_level: None,
            abs_level: None,
            brake_bias: None,
            tyre_bar: None,
            pit_limiter: false,
            brake: 0.0,
            throttle: 0.0,
            rpm: 0.0,
            speed_mps: 0.0,
            gear: 0,
            lights: 0,
            indicator: Indicator::Off,
            fog: false,
            dirt: 0.0,
            wet: 0.0,
            rain: 0.0,
            night: false,
            disc_temp_c: 20.0,
            exhaust_heat: 0.0,
            backfire_age: -1.0,
            smoke: 0.0,
            dust: 0.0,
            bloom: 0.0,
            time: 0.0,
            blink_phase: None,
            force_fn: None,
            backfire_type: None,
        }
    }
}

fn env_f(name: &str) -> Option<f32> {
    crate::envc::var(name).ok().and_then(|v| v.trim().parse::<f32>().ok())
}

impl FxState {
    /// Everything at rest, then whatever the EVOVIEW_* vars say.
    pub fn from_env() -> Self {
        let mut s = Self::default();
        if let Some(v) = env_f("EVOVIEW_LIGHTS") {
            s.lights = (v.max(0.0) as u32).min(3);
        }
        if let Some(v) = env_f("EVOVIEW_BRAKE") { s.brake = v.clamp(0.0, 1.0); }
        if let Some(v) = env_f("EVOVIEW_THROTTLE") { s.throttle = v.clamp(0.0, 1.0); }
        if let Some(v) = env_f("EVOVIEW_RPM") { s.rpm = v.max(0.0); }
        if let Some(v) = env_f("EVOVIEW_SPEED") { s.speed_mps = v / 3.6; }
        if let Some(v) = env_f("EVOVIEW_GEAR") { s.gear = v as i32; }
        s.indicator = match crate::envc::var("EVOVIEW_INDICATOR").unwrap_or_default().to_lowercase().as_str() {
            "left" | "l" => Indicator::Left,
            "right" | "r" => Indicator::Right,
            "hazard" | "h" | "both" => Indicator::Hazard,
            _ => Indicator::Off,
        };
        s.fog = env_f("EVOVIEW_FOG").map_or(false, |v| v > 0.5);
        if let Some(v) = env_f("EVOVIEW_DIRT") { s.dirt = v.clamp(0.0, 1.0); }
        if let Some(v) = env_f("EVOVIEW_WET") { s.wet = v.clamp(0.0, 1.0); }
        if let Some(v) = env_f("EVOVIEW_RAIN") {
            s.rain = v.clamp(0.0, 1.0);
            if s.rain > 0.0 {
                s.wet = s.wet.max(1.0);
            }
        }
        s.night = env_f("EVOVIEW_NIGHT").map_or(false, |v| v > 0.5);
        if let Some(v) = env_f("EVOVIEW_DISC_TEMP") { s.disc_temp_c = v; }
        if let Some(v) = env_f("EVOVIEW_EXHAUST_TEMP") { s.exhaust_heat = v.clamp(0.0, 1.0); }
        if let Some(v) = env_f("EVOVIEW_BACKFIRE") { s.backfire_age = v; }
        if let Some(v) = env_f("EVOVIEW_BACKFIRE_TYPE") { s.backfire_type = Some((v as u32).clamp(1, 5)); }
        if let Some(v) = env_f("EVOVIEW_SMOKE") { s.smoke = v.clamp(0.0, 1.0); }
        if let Some(v) = env_f("EVOVIEW_DUST") { s.dust = v.clamp(0.0, 1.0); }
        if let Some(v) = env_f("EVOVIEW_BLOOM") { s.bloom = v.max(0.0); }
        s.blink_phase = env_f("EVOVIEW_BLINK").map(|v| v > 0.5);
        // debug: light one lighting function alone (fxdata FN_*)
        s.force_fn = env_f("EVOVIEW_FORCE_FN").map(|v| v as usize);
        // a still with an indicator set shows it lit unless told otherwise
        if s.blink_phase.is_none() && s.indicator != Indicator::Off {
            s.blink_phase = Some(true);
        }
        s
    }

    /// Copy the driving inputs (see `follow_car` for the whole car state).
    pub fn from_parts(&mut self, throttle: f32, brake: f32, rpm: f32, speed_mps: f32, gear: i32) {
        self.throttle = throttle;
        self.brake = brake;
        self.rpm = rpm;
        self.speed_mps = speed_mps;
        self.gear = gear;
    }

    /// Drive / demo / replay mode: take the car state (carstate.rs) and
    /// derive what the effects need from it. Brake lamps follow the pedal,
    /// the dash and shift LEDs follow rpm / speed / gear, the backfire sim
    /// sees the lift-off itself (particles.rs), the brake discs heat with
    /// braking energy and cool with airflow, and the rear tyres smoke on a
    /// launch wheelspin or a handbrake slide - the drive model has no tyre
    /// slip, so that last one is a heuristic.
    pub fn follow_car(&mut self, cs: &crate::carstate::CarState, dt: f32) {
        self.from_parts(cs.throttle, cs.brake, cs.rpm, cs.speed_mps.abs(), cs.gear);
        self.lights = (cs.lights as u32).min(3);
        self.indicator = match cs.indicator {
            -1 => Indicator::Left,
            1 => Indicator::Right,
            2 => Indicator::Hazard,
            _ => Indicator::Off,
        };
        // live blinking, not the still's forced phase
        self.blink_phase = None;
        // disc temperature: heat ~ braking power, cooling ~ (T - ambient)
        // with airflow. ~3 s of full braking from 180 km/h reaches ~900 C.
        let v = cs.speed_mps.abs();
        let heat = cs.brake * v * 6.5;
        let cool = (self.disc_temp_c - 20.0) * (0.05 + v * 0.004);
        self.disc_temp_c = (self.disc_temp_c + (heat - cool) * dt).clamp(20.0, 1200.0);
        // exhaust heat follows load and rpm
        let limit = if cs.rpm_limit > 0.0 { cs.rpm_limit } else { 7000.0 };
        let target = (cs.throttle * (cs.rpm / limit)).clamp(0.0, 1.0);
        self.exhaust_heat += (target - self.exhaust_heat) * (dt * 0.5).min(1.0);
        // wheelspin: full throttle, low gear, low speed, high rpm; or a slide
        let launch = cs.throttle > 0.9 && (1..=2).contains(&cs.gear) && v < 18.0 && cs.rpm > limit * 0.75;
        let slide = cs.handbrake > 0.5 && v > 5.0;
        let want = if launch || slide { 1.0 } else { 0.0 };
        self.smoke += (want - self.smoke) * (dt * 4.0).min(1.0);
    }

    pub fn indicator_name(&self) -> &'static str {
        match self.indicator {
            Indicator::Off => "off",
            Indicator::Left => "left",
            Indicator::Right => "right",
            Indicator::Hazard => "hazard",
        }
    }

    pub fn next_indicator(&mut self) {
        self.indicator = match self.indicator {
            Indicator::Off => Indicator::Left,
            Indicator::Left => Indicator::Right,
            Indicator::Right => Indicator::Hazard,
            Indicator::Hazard => Indicator::Off,
        };
    }

    /// One line for the window title / console.
    pub fn summary(&self) -> String {
        format!(
            "lights {} | brake {:.0}% | ind {} | fog {} | gear {} | {:.0} rpm | {:.0} km/h | disc {:.0}C | dirt {:.1} | wet {:.1}{}{}",
            self.lights, self.brake * 100.0, self.indicator_name(),
            if self.fog { "on" } else { "off" }, self.gear, self.rpm,
            self.speed_mps * 3.6, self.disc_temp_c, self.dirt, self.wet,
            if self.night { " | night" } else { "" },
            if self.bloom > 0.0 { " | bloom" } else { "" })
    }
}
