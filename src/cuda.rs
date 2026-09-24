//! Device buffers and the two kernel modules.
//!
//! The toolkit's `vm` layer gives a `Module` (one loaded fatbin), a `DevBuf` (a
//! device allocation) and `Args`/`Launch` for calling into them; this file adds
//! what MAXIM needs on top: a buffer type that knows its element count, a cache
//! of weights already uploaded, and the two fatfins - the toolkit's kernel set
//! and this engine's - loaded as separate modules so neither can shadow a name
//! in the other.
//!
//! The fatbin paths come from the build script as `LIGHTGPU_FATBIN_*` (one per
//! `lightgpu-build::Source`), which is why the names here have to match
//! `build.rs` exactly. That coupling is deliberate: a mistyped module name is a
//! compile error rather than a launch failure.

use lightgpu::vm::{Args, DevBuf, Launch, Module};
use std::collections::HashMap;

use crate::Error;

/// This engine's own kernels.
pub const PROJECT_FATBIN: &[u8] = include_bytes!(env!("LIGHTGPU_FATBIN_MAXIM_PROJECT"));

/// The toolkit kernels this engine calls - the same list as `build.rs`, which is
/// what compiles them in.
pub const TOOLKIT_FATBIN: &[u8] = include_bytes!(env!("LIGHTGPU_FATBIN_MAXIM_TOOLKIT"));

/// The names in each module, for the startup completeness check.
pub const PROJECT_KERNELS: &[&str] = &[
    "mx_conv1x1_t",
    "mx_conv3x3_t2",
    "mx_conv3x3_t4",
    "mx_conv4x4s2",
    "mx_convt2x2s2",
    "mx_gate_mm",
    "mx_gate_mm_t0",
    "mx_gate_mm_t1",
    "mx_mul",
    "mx_gate_apply",
    "mx_resize_axis",
    "mx_block_perm",
    "mx_channel_mean",
    "mx_channel_scale",
    "mx_down2",
    "mx_sigmoid",
];

pub const TOOLKIT_KERNELS: &[&str] = &[
    "lg_add",
    "lg_conv1x1",
    "lg_conv3x3s1p1",
    "lg_channel_layer_norm",
    "lg_gelu_erf",
    "lg_lrelu",
];

pub struct Cuda {
    pub project: Module,
    pub toolkit: Module,
}

impl Cuda {
    pub fn new() -> Result<Cuda, Error> {
        lightgpu::vm::init().map_err(Error)?;
        let project = Module::load(PROJECT_FATBIN).map_err(Error)?;
        let toolkit = Module::load(TOOLKIT_FATBIN).map_err(Error)?;
        for k in PROJECT_KERNELS {
            if !project.has(k) {
                return Err(format!("the project fatbin is missing {k}").into());
            }
        }
        for k in TOOLKIT_KERNELS {
            if !toolkit.has(k) {
                return Err(format!("the toolkit fatbin is missing {k}").into());
            }
        }
        Ok(Cuda { project, toolkit })
    }

    /// Which module owns a kernel. The `mx_` prefix is the engine's, everything
    /// else is the toolkit's, and the check above guarantees the name is there.
    pub fn module_of(&self, kernel: &str) -> &Module {
        if kernel.starts_with("mx_") {
            &self.project
        } else {
            &self.toolkit
        }
    }

    /// Launch `kernel` with `args`. `grid`/`block` are in threads.
    pub fn launch(&self, kernel: &str, grid: (u32, u32, u32), block: (u32, u32, u32), args: &mut Args) -> Result<(), Error> {
        let l = Launch::new(grid, block);
        args.launch(self.module_of(kernel), kernel, l).map_err(Error)
    }

    /// A grid covering `n` elements with `block` threads each.
    pub fn grid_for(n: usize, block: u32) -> (u32, u32, u32) {
        let b = block.max(1);
        (((n as u64 + b as u64 - 1) / b as u64).max(1) as u32, 1, 1)
    }
}

/// A device allocation with a known element count.
pub struct Buf {
    pub dev: DevBuf,
    pub n: usize,
}

impl Buf {
    pub fn new(n: usize) -> Result<Buf, Error> {
        Ok(Buf { dev: DevBuf::alloc(n * 4).map_err(Error)?, n })
    }

    pub fn upload(&mut self, v: &[f32]) -> Result<(), Error> {
        assert_eq!(v.len(), self.n);
        self.dev.upload(v).map_err(Error)
    }

    pub fn download(&self, out: &mut [f32]) -> Result<(), Error> {
        assert_eq!(out.len(), self.n);
        self.dev.download(out).map_err(Error)
    }

    /// The device address, which `Args::ptr` takes by value.
    pub fn ptr(&self) -> lightgpu::ffi::CUdeviceptr {
        self.dev.ptr
    }
}

/// The weights, on the device. A weight is uploaded the first time an op names
/// it, so a plan that resolves a parameter it never uses costs nothing.
pub struct WeightCache {
    bufs: Vec<Option<Buf>>,
    hits: usize,
    uploads: usize,
    /// Total weight bytes the device has been asked for, whether or not it is
    /// resident: `resident_bytes` counts what a successful run left allocated,
    /// which is not what a run that ran out of memory needs to know.
    wanted_bytes: usize,
}

impl WeightCache {
    pub fn new(n: usize) -> WeightCache {
        WeightCache { bufs: (0..n).map(|_| None).collect(), hits: 0, uploads: 0, wanted_bytes: 0 }
    }

    pub fn get(&mut self, id: usize, host: &[Vec<f32>]) -> Result<&Buf, Error> {
        if self.bufs[id].is_none() {
            self.wanted_bytes += host[id].len() * 4;
            // An allocation failure here is the small-card failure mode, and
            // "cuMemAlloc failed" alone does not say how much was wanted.
            let mut b = Buf::new(host[id].len()).map_err(|e| {
                Error(format!(
                    "weight blob {id}: {} MiB, {:.1} MiB of weights wanted so far: {e}",
                    host[id].len() * 4 / 1048576,
                    self.wanted_bytes as f64 / 1048576.0
                )
                .into())
            })?;
            b.upload(&host[id])?;
            self.bufs[id] = Some(b);
            self.uploads += 1;
        } else {
            self.hits += 1;
        }
        Ok(self.bufs[id].as_ref().unwrap())
    }

    pub fn resident_bytes(&self) -> usize {
        self.bufs.iter().flatten().map(|b| b.n * 4).sum()
    }

    pub fn stats(&self) -> (usize, usize) {
        (self.uploads, self.hits)
    }
}

/// Per-buffer shapes, looked up by the launchers that need them.
pub struct Shapes<'a> {
    pub plan: &'a crate::model::Plan,
}

impl<'a> Shapes<'a> {
    pub fn get(&self, id: usize) -> crate::model::BufShape {
        self.plan.bufs[id]
    }
}

/// Convenience: a `HashMap`-free way to keep the per-op scratch the launchers
/// need (the `Args` marshalling keeps its own small allocations alive).
pub type Scratch = HashMap<&'static str, Vec<f32>>;
