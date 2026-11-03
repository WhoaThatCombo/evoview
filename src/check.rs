//! `--check report.json [--sheet <dir | sheet.png>]`: exercise a car without
//! the game and say what is broken. One report, two halves:
//!
//! * check_anim.rs (look-anim): steer lock to lock, every corner droop to
//!   bump, pedals, gears, every bound clip, every needle, tyres vs body -
//!   CPU-side on the real rig, thresholds calibrated on the stock fleet.
//! * check_fx.rs (look-fx): every light function on its own, LEDs / shift
//!   lights, the dash, brake glow, exhaust flames, dirt, textures, materials,
//!   glass - rendered headless and diffed.
//!
//! The report is `{car, records: [{car, check, severity, detail}], counts,
//! anim: {physics, slots, corners, steering, clearance, needles, ...}}`;
//! animation records are named `anim_<code>`. The sheet tiles the
//! animation key poses with the effect renders.

use crate::asset::Car;
use crate::load::Skeleton;
use std::path::Path;

pub fn run(car: Car, skeleton: Option<Skeleton>, name: &str, out: &Path, sheet: Option<&Path>) {
    let anim = crate::check_anim::analyse(&car, skeleton.as_ref(), name);
    pollster::block_on(crate::check_fx::run(car, name, out, sheet, Some(anim)));
}
