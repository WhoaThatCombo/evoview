//! The car's live inputs and outputs, shared by everything that animates or
//! lights the car: drive.rs writes it every frame (keyboard drive model,
//! scripted demo, or a replay), anim/lights/fx read it.
//!
//! Kept deliberately small and plain-old-data so a reader never needs to know
//! where the numbers came from. Units are SI unless the name says otherwise.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarState {
    /// Road speed, metres per second (signed: negative when reversing).
    pub speed_mps: f32,
    /// Engine speed.
    pub rpm: f32,
    /// Rev limiter from the car's .carengine (0 = unknown).
    pub rpm_limit: f32,
    /// -1 reverse, 0 neutral, 1.. forward gears.
    pub gear: i32,
    /// Pedals, 0..1.
    pub throttle: f32,
    pub brake: f32,
    pub clutch: f32,
    /// Steering WHEEL position, -1 full left lock .. +1 full right lock.
    pub steer: f32,
    /// 0..1.
    pub handbrake: f32,
    /// Light switch: 0 off, 1 DRL/side, 2 low beam, 3 high beam.
    pub lights: u8,
    /// Indicator: -1 left, 0 off, 1 right, 2 hazards.
    pub indicator: i8,
    /// Driver aids currently intervening.
    pub abs_active: bool,
    pub tc_active: bool,
}

impl Default for CarState {
    fn default() -> Self {
        CarState {
            speed_mps: 0.0,
            rpm: 0.0,
            rpm_limit: 0.0,
            gear: 0,
            throttle: 0.0,
            brake: 0.0,
            clutch: 0.0,
            steer: 0.0,
            handbrake: 0.0,
            lights: 0,
            indicator: 0,
            abs_active: false,
            tc_active: false,
        }
    }
}
