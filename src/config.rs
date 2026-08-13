//! Simulation config (feature `config`): the Rust replacement for Fiberfox's `.ffp` XML.
//!
//! A single `serde`-deserializable struct (TOML). Because the whole scheme is known up-front, the
//! `.ffp` load-order bug (motionvolumes parsed against zero gradients → all volumes move) simply
//! can't happen here. Sketch of the surface is in `docs/FEATURES.md`.
//!
//! Fill in `#[derive(Deserialize)]` structs mirroring [`crate::kspace::Acquisition`],
//! [`crate::motion::MotionMode`], and the compartment/signal-model parameters, then
//! `toml::from_str` in the CLI.

// #[derive(serde::Deserialize)]
// pub struct SimConfig { pub acquisition: ..., pub motion: ..., pub compartments: ... }

/// Load a TOML config into a `SimConfig`.
pub fn load(_path: &std::path::Path) -> std::io::Result<()> {
    todo!("toml::from_str into SimConfig; see docs/FEATURES.md config sketch")
}
