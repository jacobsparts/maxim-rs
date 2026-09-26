//! What a CPU pass will need, and what there is.
//!
//! THE DEVICE SIDE ANSWERS THIS EXACTLY, in `exec_gpu::Gpu::new`: a pass is one
//! packed arena whose length the plan's own liveness pass computed, so the number
//! compared against free VRAM is the number the driver is about to be asked for.
//! The CPU side is nearly as exact, because it allocates that SAME arena - as a
//! `Vec<f32>` - and the checkpoint, and nothing else that scales with the image.
//! So this is a small correction on top of two numbers the plan already knows,
//! rather than an estimate of what a graph might do.
//!
//! WHAT IT MODELS, and where each term comes from:
//!
//!   * THE ARENA, exactly: `plan.arena_len` elements, one allocation.
//!   * THE CHECKPOINT TWICE, because `Host` holds its own copy of the weights and
//!     the plan holds another. Two copies is one more than the engine needs; it
//!     is what the two are, and a guard that under-counted it would be wrong in
//!     the direction that matters.
//!   * THE PER-TASK SCRATCH the CPU kernels allocate inside an op. The conv1x1
//!     path splits a plane into at most `12 * threads` tasks and gives each one a
//!     `PB x c_in` tile, a `PB` accumulator and a `c_out` bucket header, so the
//!     ceiling is a function of the WIDEST LAYER IN THE GRAPH and the size of the
//!     pool - not of the image, which is why it stays small next to the arena
//!     even at 2048x2048.
//!   * A FIXED BASE for the process, the allocator's bookkeeping and the PNG
//!     buffers at either end.
//!
//! MEASURED against `Maximum resident set size` (maxim-lol, 24 threads, so 288
//! tasks in the scratch term), the model lands 1.08x to 1.19x OVER the real peak:
//! 1280x853 2451.5 MiB against a model of 2657.0, 640x480 873.7 against 1043.0.
//! The overestimate is the scratch term, which is charged at the task cap for
//! every size although only a large plane reaches it, and it is the right side to
//! err on: better to refuse a pass that would have fitted than to start one that
//! swaps.

use crate::model::{Op, Plan};
use crate::Error;

/// The factor between the modelled footprint and the measured peak RSS. The
/// model is already over the readings above, so this is small: what is left to
/// absorb is allocator slack and the page tables for the arena.
pub const SLACK: f64 = 1.05;

/// The part that does not scale with the image: the process, the loader, the
/// allocator's bookkeeping and the PNG buffers.
pub const BASE: usize = 64 * 1024 * 1024;

/// The CPU pass's memory, as arithmetic, before any of it is allocated.
pub struct CpuPlan {
    /// The activation arena: `plan.arena_len` elements.
    pub arena: usize,
    /// The checkpoint, twice - the plan's copy and `Host`'s.
    pub weights: usize,
    /// The per-task scratch, at the widest layer in the graph.
    pub scratch: usize,
    /// `(arena + weights + scratch) * SLACK + BASE`. This is what the guard
    /// compares against what the machine has.
    pub peak: usize,
}

impl CpuPlan {
    /// The plan for `plan`'s shape - which is the PADDED geometry, since that is
    /// what a pass actually runs.
    pub fn of(plan: &Plan) -> CpuPlan {
        let arena = plan.arena_bytes();
        let weights = plan.weight_bytes() * 2;
        let scratch = scratch_bytes(plan);
        let peak = ((arena + weights + scratch) as f64 * SLACK) as usize + BASE;
        CpuPlan { arena, weights, scratch, peak }
    }
}

/// The per-task scratch of the widest op in the plan.
///
/// The conv1x1 path is the only op that allocates per task, and it allocates a
/// `PB x c_in` tile, a `PB` accumulator and a `c_out` bucket header per task. The
/// task count is capped at `12 * threads` (see the arm that computes `ntask`), so
/// the ceiling follows from the widest `c_in` in the graph.
fn scratch_bytes(plan: &Plan) -> usize {
    let c_in = plan
        .ops
        .iter()
        .filter_map(|op| match op {
            Op::Conv1x1 { c_in, c_out, .. } => Some((*c_in, *c_out)),
            _ => None,
        })
        .max()
        .unwrap_or((1, 1));
    let tasks = 12 * rayon::current_num_threads().max(1);
    tasks * (crate::exec_cpu::PB * c_in.0 * 4 + crate::exec_cpu::PB * 4 + c_in.1 * 4)
}

/// What the kernel says is available for a new allocation, in bytes.
///
/// `MemAvailable` AND NOT `MemFree`: free memory is the part that is already
/// unused, while available is what a process can actually get without swapping -
/// it counts reclaimable page cache, which on a machine that has just read a
/// 57 MB checkpoint is most of it. Guarding on `MemFree` would refuse passes that
/// fit.
///
/// `None` when the file is unreadable or has no such line, and the caller then
/// does not guard at all: refusing every pass because a `/proc` line moved would
/// be worse than running one that might swap.
pub fn host_available() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        // A `?` HERE WOULD RETURN FROM THE WHOLE FUNCTION ON THE FIRST LINE THAT
        // DOES NOT MATCH, and `/proc/meminfo`'s first line is `MemTotal:` - so
        // this loop would skip straight to the `None` at the bottom and the guard
        // would silently do nothing. A missing key is a `continue`, not a failure.
        let Some(rest) = line.strip_prefix("MemAvailable:") else {
            continue;
        };
        let kb: usize = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
        return Some(kb * 1024);
    }
    None
}

/// Refuse the pass if it will not fit, and report what it would have needed
/// either way.
///
/// THE CPU BACKEND IS THE LAST ONE, so there is nothing to fall back to, and a
/// pass that runs a machine out of RAM does not fail - it SWAPS, and a machine
/// that is swapping is worse off than one that was told no. This is called
/// before the arena is allocated, so a refusal costs nothing.
pub fn check(plan: &Plan) -> Result<(), Error> {
    let p = CpuPlan::of(plan);
    let avail = host_available();
    let slack = ((SLACK - 1.0) * 100.0).round() as usize;
    // REPORTED EITHER WAY, like the device plan: which pass is about to run, and
    // against how much memory, is a property of the result rather than progress
    // output.
    match avail {
        Some(a) => eprintln!(
            "maxim: cpu plan {} MiB (arena {} + weights {} + scratch {}, plus {}% slack), {} MiB available",
            p.peak / 1048576,
            p.arena / 1048576,
            p.weights / 1048576,
            p.scratch / 1048576,
            slack,
            a / 1048576,
        ),
        None => eprintln!(
            "maxim: cpu plan {} MiB (arena {} + weights {} + scratch {})",
            p.peak / 1048576,
            p.arena / 1048576,
            p.weights / 1048576,
            p.scratch / 1048576,
        ),
    }
    let Some(avail) = avail else {
        return Ok(());
    };
    if p.peak <= avail {
        return Ok(());
    }
    Err(format!(
        "not enough memory for a {}x{} pass on the CPU\n\
         maxim: it needs about {} MiB ({} of arena + {} of weights, twice + {} of per-task scratch, plus {}% for allocator slack); {} MiB is available\n\
         maxim: the arena is exact - it is the plan's own packed size - so the only modelled part is the slack on top\n\
         maxim: a smaller image, a lighter checkpoint, or the GPU engine is what fits",
        plan.shape.w,
        plan.shape.h,
        p.peak / 1048576,
        p.arena / 1048576,
        p.weights / 1048576,
        p.scratch / 1048576,
        slack,
        avail / 1048576,
    )
    .into())
}
