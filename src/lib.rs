//! MAXIM on the lightgpu toolkit: the model plan, a CPU executor and a GPU one.
//!
//! The engine is organised around one artifact: a [`model::Plan`], the
//! straight-line sequence of ops a *fixed input shape* runs. Building the plan
//! resolves every parameter name, transposes each kernel into the layout a
//! lightgpu kernel wants, and records every buffer's size and live range. Both
//! backends then walk that same list, so `--device cpu` and `--device gpu` cannot
//! drift apart by more than floating point, and a new op only has to be added
//! once - in three places that the compiler checks against each other (the op
//! enum, `exec_cpu` and `exec_gpu`).

pub mod config;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod exec_cpu;
#[cfg(feature = "cuda")]
pub mod exec_gpu;
pub mod host;
pub mod image;
pub mod memguard;
pub mod model;
pub mod weights;

use std::fmt;

/// The engine's error type. Everything that can go wrong here is a bad file, a
/// missing parameter or an unsupported shape, so a string is the honest
/// representation.
#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(s: String) -> Error {
        Error(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Error {
        Error(s.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error(e.to_string())
    }
}
