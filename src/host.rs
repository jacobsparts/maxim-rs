//! What both backends share: the plan, the activation arena, and the weights.
//!
//! The arena is ONE allocation that every activation lives in, at the offset the
//! plan recorded. On the CPU that is a `Vec<f32>`; on the GPU it is a single
//! device buffer. Buffers that the plan says alias (channel views) are just
//! different offsets into it, so a split or a concatenation costs nothing on
//! either backend - which is the reason the graph is a straight line of ops over
//! one arena rather than a tree of allocations.
//!
//! The weights are kept as the checkpoint's own 1760 tensors, transposed only
//! where a kernel's layout demands it. They are NOT composed into GEMM-ready
//! matrices: fusing a bias into a kernel changes the accumulation order, and the
//! engine's whole point is to agree with the reference to the last bit it can.

use crate::model::Plan;
use std::sync::Arc;

/// Everything the executors need that is not a kernel.
pub struct Host {
    pub plan: Plan,
    /// The activation arena, `plan.arena_len` elements.
    pub arena: Vec<f32>,
    /// The weight blobs, indexed the way `Op`s index them.
    pub weights: Vec<Vec<f32>>,
    /// Bumped by every op that touches a buffer, for `--profile`. Shared so the
    /// GPU executor can count launches from inside its op loop.
    pub ops_run: Arc<std::sync::atomic::AtomicUsize>,
}

impl Host {
    /// The full host arena - what the CPU backend and the dump/verify paths need.
    pub fn new(plan: Plan) -> Host {
        let arena = vec![0.0f32; plan.arena_len];
        let weights = plan.weights.clone();
        Host { plan, arena, weights, ops_run: Arc::new(std::sync::atomic::AtomicUsize::new(0)) }
    }

    /// A host whose arena is sized to what the caller will actually use it for.
    ///
    /// `new` allocates `plan.arena_len` elements - 942.7 MiB at the eval size - and
    /// a GPU run never reads an activation out of the host arena: `main` copies the
    /// input image in and, unless the run dumps or crops, never looks at it again.
    /// The device executes off its own copy of whatever `begin` uploaded.
    ///
    /// So the caller says how much host arena it wants. `arena_len` is the real
    /// length for the CPU backend, `--dump` and `--verify-gpu`; for a plain GPU run
    /// it is enough to hold the input buffer. Passing the full length is always
    /// correct, so this cannot silently break a caller that forgets to think about
    /// it - it would just cost the allocation. The arena is still ZEROED either
    /// way, so no reading path can see uninitialized memory: the saving is the
    /// 942.7 MiB of untouched pages, not a relaxation of the initialization.
    pub fn with_arena_len(plan: Plan, arena_len: usize) -> Host {
        // The input buffer is the ONLY part of the arena a GPU run touches, and it
        // lives at `offs[input]`, so that offset plus its length is the real floor
        // - not the buffer's length alone, which would shift the image.
        let need = plan.offs[plan.input] + plan.bufs[plan.input].len();
        assert!(
            arena_len >= need,
            "host arena of {arena_len} elements cannot hold the input buffer, which ends at {need}"
        );
        let arena = vec![0.0f32; arena_len];
        let weights = plan.weights.clone();
        Host { plan, arena, weights, ops_run: Arc::new(std::sync::atomic::AtomicUsize::new(0)) }
    }

    /// The image the model takes.
    pub fn input_mut(&mut self) -> &mut [f32] {
        let o = self.plan.offs[self.plan.input];
        let n = self.plan.bufs[self.plan.input].len();
        &mut self.arena[o..o + n]
    }

    pub fn output(&self) -> &[f32] {
        let o = self.plan.offs[self.plan.output];
        let n = self.plan.bufs[self.plan.output].len();
        &self.arena[o..o + n]
    }

    /// A named buffer's shape and contents, for `--dump` and the parity test.
    pub fn find(&self, name: &str) -> Option<usize> {
        self.plan.dumps.iter().find(|(n, _)| n == name).map(|(_, id)| *id)
    }

    pub fn take(&self, id: usize) -> Vec<f32> {
        let o = self.plan.offs[id];
        let n = self.plan.bufs[id].len();
        self.arena[o..o + n].to_vec()
    }

}
