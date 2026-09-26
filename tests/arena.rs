//! The packed arena must stay close to the graph's live set, and the thing that
//! once put it at twice that was the dump names.
//!
//! THE FINDING, so the assertions below are not numbers someone picked. `rec`
//! names an activation, and the packer extends every named buffer's last use to
//! the end of the run so that `--dump` can snapshot it after every op has run.
//! 43 names held to the end of an 1840-op graph cost more than the graph's own
//! peak live set: at 480x640 the arena was 1010.1 MiB against a true live-set peak
//! of 501.5, a factor of 2.01x, and the same 2.01x at every size tested, because
//! what the names cost scales with the image exactly as the graph does.
//!
//! `--dump` is a `--features dev` flag that a release build refuses BY NAME, so a
//! release build was paying that memory for a feature it does not contain.
//! `without_dumps()` takes the same graphs to 1.07x, and the rest of the way from
//! there was the PACKING ORDER: placing the largest buffers first rather than in
//! first-use order, which is what the second test measures.
//!
//! These run on the PLAN alone - no arena is allocated and no op is executed - so
//! they cost a fraction of a second and need no GPU.

use maxim::model::{Builder, Op, Plan};
use maxim::weights::Weights;

const CKPT: &str = "/home/jacob/lightgpu-family/models/maxim-lol.safetensors";

fn plan_with(h: usize, wd: usize, dumps: bool) -> Plan {
    let w = Weights::open(CKPT).expect("open the checkpoint");
    let cfg = w.config.clone();
    let b = Builder::new(&w, cfg);
    let b = if dumps { b } else { b.without_dumps() };
    b.build(h, wd).expect("build the plan")
}

/// The peak of the live set, in bytes, and the op it happens at.
///
/// A buffer is charged to the storage that owns its bytes: a view is a channel
/// range of its parent, and counting both would count the same memory twice.
fn live_set_peak(plan: &Plan) -> (usize, usize) {
    let n = plan.bufs.len();
    fn root(plan: &Plan, i: usize) -> usize {
        match plan.alias[i] {
            None => i,
            Some((p, _)) => root(plan, p),
        }
    }
    let owner: Vec<usize> = (0..n).map(|i| root(plan, i)).collect();
    let mut first = vec![usize::MAX; n];
    let mut last = vec![0usize; n];
    for (oi, op) in plan.ops.iter().enumerate() {
        let mut ids = Vec::new();
        op.operands(&mut ids);
        for b in ids {
            let b = owner[b];
            if first[b] == usize::MAX {
                first[b] = oi;
            }
            last[b] = oi;
        }
    }
    let (mut live, mut peak, mut peak_op) = (0usize, 0usize, 0usize);
    let mut cur = vec![false; n];
    for i in 0..plan.ops.len() {
        for b in 0..n {
            if owner[b] != b {
                continue;
            }
            let want = first[b] != usize::MAX && first[b] <= i && i <= last[b];
            if want && !cur[b] {
                live += plan.bufs[b].len() * 4;
                cur[b] = true;
            } else if !want && cur[b] {
                live -= plan.bufs[b].len() * 4;
                cur[b] = false;
            }
        }
        if live > peak {
            peak = live;
            peak_op = i;
        }
    }
    (peak, peak_op)
}

#[test]
fn the_arena_is_close_to_the_live_set() {
    for (h, w) in [(480, 640), (832, 1280)] {
        let plan = plan_with(h, w, false);
        let (peak, at) = live_set_peak(&plan);
        let arena = plan.arena_bytes();
        let ratio = arena as f64 / peak as f64;
        println!(
            "{h}x{w}: arena {:.1} MiB against a live-set peak of {:.1} MiB at op {at} -> {ratio:.2}x",
            arena as f64 / 1048576.0,
            peak as f64 / 1048576.0
        );
        assert!(
            ratio <= 1.20,
            "{h}x{w}: the arena is {ratio:.2}x the live-set peak ({} MiB against {} MiB). It was \
             1.07x; a factor near 1.2 means the packing order stopped placing the large buffers \
              first, and near 2 means named activations are being held to the end of the run \
              again. See the note at the top of this file.",
            arena / 1048576,
            peak / 1048576
        );
    }
}

#[test]
fn naming_activations_is_what_costs_the_memory() {
    let named = plan_with(480, 640, true);
    let lean = plan_with(480, 640, false);
    let ratio = named.arena_bytes() as f64 / lean.arena_bytes() as f64;
    println!(
        "480x640: {} names hold the arena at {:.1} MiB against {:.1} MiB unnamed -> {ratio:.2}x",
        named.dumps.len(),
        named.arena_bytes() as f64 / 1048576.0,
        lean.arena_bytes() as f64 / 1048576.0
    );
    assert!(
        ratio > 1.5,
        "naming {} activations only cost {ratio:.2}x: either the packer stopped extending a named \
         buffer to the end of the run - which would make `--dump` write whatever was written there \
         last, and is what tests/packing.rs also catches - or the graph stopped naming anything",
        named.dumps.len()
    );
}
