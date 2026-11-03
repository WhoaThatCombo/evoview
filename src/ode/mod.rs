//! Open Dynamics Engine 0.15.2, ported to Rust (BSD-3-Clause upstream; see
//! THIRD-PARTY-NOTICES.md). AC EVO's physics runs on a Kunos build of ODE
//! 0.15.x: double precision, OPCODE trimeshes, QuickStep.
//!
//! The port follows ODE's code expression by expression so the results
//! match the C library bit for bit; tools/ode_oracle checks that against
//! the real 0.15.2 built from source.
#![allow(dead_code)]

pub mod api;
pub mod collision;
pub mod joints;
pub mod math;
pub mod opcode;
pub mod quickstep;
pub mod volume;
pub mod world;
#[cfg(test)]
mod oracle;

#[allow(unused_imports)]
pub use joints::{Contact, Feedback, JointType};
#[allow(unused_imports)]
pub use math::{Mass, Real, M3, V3};
#[allow(unused_imports)]
pub use world::{BodyId, GroupId, JointId, World};
