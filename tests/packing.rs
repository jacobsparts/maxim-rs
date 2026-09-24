//! The arena packer must be transparent: naming a buffer (which extends its
//! live range to the end of the run, so `--dump` can snapshot it) must not
//! change a single number the graph computes.
//!
//! This is not a hypothetical. `Op::ChanScale`'s gate operand was missing from
//! the liveness list, so the packer reused the CALayer's sigmoid buffer before
//! the multiply read it. Adding a `rec()` inside `calayer` happened to name that
//! buffer, which extended its life and hid the bug - the run looked correct and
//! was correct only by accident. A test that runs the SAME graph with and
//! without dumps and demands bit-identical output is what turns that class of
//! bug from invisible into a failure.

use maxim::config::Config;
use maxim::exec_cpu;
use maxim::host::Host;
use maxim::model::Builder;
use maxim::weights::Weights;

const CKPT: &str = "/home/jacob/lightgpu-family/models/maxim-lol.safetensors";

fn small_input(h: usize, w: usize) -> Vec<f32> {
    // A deterministic, non-symmetric image: a symmetric one could hide an axis
    // swap in a permutation op.
    (0..3 * h * w)
        .map(|i| {
            let x = (i % w) as f32;
            let y = (i / w % h) as f32;
            ((x * 0.03).sin() * 0.5 + (y * 0.017).cos() * 0.3 + 0.5).clamp(0.0, 1.0)
        })
        .collect()
}

/// 64 is the smallest size S-2 can run: with depth 3 the deepest level is
/// h/8 = 8, which is exactly the low-res block/grid size. Anything smaller asks
/// for a sub-block feature map and `gmlp_axis` refuses.
const SIDE: usize = 64;

fn run(record: bool) -> Vec<f32> {
    let w = Weights::open(CKPT).expect("open the checkpoint");
    let cfg = Config::for_task("enhancement").expect("the S-2 config");
    let b = Builder::new(&w, cfg);
    let b = if record { b } else { b.without_dumps() };
    let plan = b.build(SIDE, SIDE).expect("build the plan");
    let mut host = Host::new(plan);
    let data = small_input(host.plan.shape.h, host.plan.shape.w);
    host.input_mut()[..data.len()].copy_from_slice(&data);
    let (plan, weights) = (&host.plan, &host.weights);
    exec_cpu::run(plan, weights, &mut host.arena);
    let o = host.plan.offs[host.plan.output];
    let n = host.plan.bufs[host.plan.output].len();
    host.arena[o..o + n].to_vec()
}

#[test]
fn dumps_do_not_change_the_result() {
    let named = run(true);
    let bare = run(false);
    assert_eq!(named.len(), bare.len(), "the two plans disagree on the output size");
    let mut worst = 0.0f32;
    for (a, b) in named.iter().zip(&bare) {
        worst = worst.max((a - b).abs());
    }
    // Bit-identical, not approximately: this is the same op list over the same
    // arithmetic, so any difference at all means the two runs executed different
    // work - which is exactly the failure the packer has to be free of.
    assert_eq!(worst, 0.0, "naming buffers changed the output by {worst}");
}

#[test]
fn the_plan_is_the_size_it_claims() {
    let w = Weights::open(CKPT).expect("open the checkpoint");
    let cfg = Config::for_task("enhancement").expect("the S-2 config");
    let plan = Builder::new(&w, cfg).build(SIDE, SIDE).expect("build the plan");
    for (label, id) in plan.dumps.iter() {
        let o = plan.offs[*id];
        let n = plan.bufs[*id].len();
        assert!(
            o + n <= plan.arena_len,
            "`{label}` runs past the arena: {} + {} > {}",
            o,
            n,
            plan.arena_len
        );
    }
}

/// An op's destination must not overlap any source it reads AT THAT INSTANT.
/// The packer models this with live ranges, but liveness is derived from the
/// same operand list that a missing entry would corrupt, so this checks the
/// resulting offsets directly - which is what the executor would trip over.
#[test]
fn no_op_overlaps_its_own_inputs() {
    let w = Weights::open(CKPT).expect("open the checkpoint");
    let cfg = Config::for_task("enhancement").expect("the S-2 config");
    let plan = Builder::new(&w, cfg).without_dumps().build(SIDE, SIDE).expect("build the plan");
    let mut bad = Vec::new();
    for (i, op) in plan.ops.iter().enumerate() {
        let mut ids = Vec::new();
        op.operands(&mut ids);
        if ids.len() < 2 {
            continue;
        }
        let (dst, rest) = (ids[0], &ids[1..]);
        let d = plan.offs[dst];
        let dn = plan.bufs[dst].len();
        for &s in rest {
            let so = plan.offs[s];
            let sn = plan.bufs[s].len();
            if d < so + sn && so < d + dn {
                bad.push(format!(
                    "op {i}: dst {} `{}` [{}..{}] overlaps src {} `{}` [{}..{}]",
                    dst, plan.labels[dst], d, d + dn, s, plan.labels[s], so, so + sn
                ));
            }
        }
    }
    assert!(bad.is_empty(), "{} op(s) alias their inputs:\n{}", bad.len(), bad.join("\n"));
}
