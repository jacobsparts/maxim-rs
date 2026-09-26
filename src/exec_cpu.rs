//! The CPU executor: the arithmetic twin of every op, in plain Rust.
//!
//! This is the CPU backend: the path this engine takes when there is no GPU,
//! and it is held to the same performance standard as the GPU executor: it runs
//! one op at a time in graph order, as it always has, but each op's own channels
//! are spread across a rayon pool (see `par_slice`). Having both backends walk
//! the same op list is what makes the engine testable without a CUDA driver.
//!
//! Each arm is written next to the kernel it mirrors and in the same order of
//! operations, so a mismatch is a real difference rather than a rounding one. The
//! flat elementwise arms apply the twin's formula from `lightgpu::ops::cpu` per
//! element instead of calling it on the whole slice, because a task has to be able
//! to own a flat range of the buffer; they use the twin's constants and its order
//! of operations, and the result is verified bit-identical to the serial path on
//! every activation the plan names. `chan_ln` is the same trade one level up: the
//! twin's arithmetic, the twin's order, but a schedule that can use the pool.
//!
//! Every op below is therefore both parallel and parity-preserving, and the
//! measurement says which one to look at next: run `--profile` and the table is
//! sorted by total time (see `report_cpu_timeline` in `main.rs`).

use crate::model::{BufShape as BufShape2, Op, Plan, EPS};
use lightgpu::ops::cpu as t;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// Set by `run` between ops so a single op can be forced down the serial path
/// without a rebuild. It is an atomic rather than an environment variable
/// because the workers read it while the driver thread writes it: `set_var`
/// while other threads are live is a data race, and the bisection it drove was
/// silently doing nothing.
static SERIAL: AtomicBool = AtomicBool::new(false);

/// The pixels a conv1x1 task owns at a time: its tile is `PB x c_in` and its
/// accumulator is `PB`. Public because `memguard` models the per-task scratch
/// this number sets, and a copy of it over there would be a second place to
/// change when this one moves.
pub const PB: usize = 256;

fn serial() -> bool {
    SERIAL.load(Ordering::Relaxed)
}

/// Execute exactly one op, so a backend can be compared against this one op by
/// op instead of only at the end of the graph - it is what `--verify-gpu` walks
/// and what `--profile` times, because "the plot is wrong somewhere" is not a
/// usable diagnostic and an op index is. See docs/DEVELOPING.md.
pub fn step(plan: &Plan, weights: &[Vec<f32>], arena: &mut [f32], i: usize) {
    let mut ctx = Ctx { plan, weights, arena };
    ctx.op(&plan.ops[i]);
}

/// A one-line description of an op, for traces and mismatch reports.
pub fn describe(op: &Op) -> String {
    short(op)
}

/// Runs the whole plan. `arena` is `plan.arena_len` elements and the input image
/// is expected to be in `plan.input`'s bytes already.
pub fn run(plan: &Plan, weights: &[Vec<f32>], arena: &mut [f32]) {
    let trace = std::env::var_os("MAXIM_TRACE").is_some();
    // Bisection hook: `MAXIM_SERIAL_UPTO=n` runs the first `n` ops with the
    // pool forced off and the rest normally. That is how a divergence is pinned
    // to a single op when every activation the plan names already differs: the
    // op after which the difference appears is the one whose arm is wrong.
    let upto: Option<usize> = std::env::var("MAXIM_SERIAL_UPTO").ok().and_then(|v| v.parse().ok());
    let from: Option<usize> = std::env::var("MAXIM_SERIAL_FROM").ok().and_then(|v| v.parse().ok());
    let mut ctx = Ctx { plan, weights, arena };
    for (i, op) in plan.ops.iter().enumerate() {
        if let Some(n) = upto {
            SERIAL.store(i < n, Ordering::Relaxed);
        }
        if let Some(n) = from {
            SERIAL.store(i >= n, Ordering::Relaxed);
        }
        if trace {
            let lb = |id: usize| format!("{}#{}", plan.labels[id], id);
            let names: Vec<String> = match op {
                crate::model::Op::Copy { dst, src, .. } => vec![lb(*dst), lb(*src)],
                crate::model::Op::Add { dst, a, b, .. }
                | crate::model::Op::Mul { dst, a, b, .. }
                | crate::model::Op::GateApply { dst, u: a, v: b, .. } => vec![lb(*dst), lb(*a), lb(*b)],
                crate::model::Op::Gelu { dst, src, .. }
                | crate::model::Op::Lrelu { dst, src, .. }
                | crate::model::Op::Sigmoid { dst, src, .. }
                | crate::model::Op::ChanLn { dst, src, .. }
                | crate::model::Op::ChanMean { dst, src, .. }
                | crate::model::Op::ChanScale { dst, src, .. }
                | crate::model::Op::Conv1x1 { dst, src, .. }
                | crate::model::Op::Conv3x3 { dst, src, .. }
                | crate::model::Op::Conv4x4s2 { dst, src, .. }
                | crate::model::Op::ConvT2x2 { dst, src, .. }
                | crate::model::Op::GateMm { dst, src, .. }
                | crate::model::Op::BlockPerm { dst, src, .. }
                | crate::model::Op::DownS { dst, src, .. } => vec![lb(*dst), lb(*src)],
                crate::model::Op::Resize { dst, src, .. } => vec![lb(*dst), lb(*src)],
            };
            eprintln!("op {i}: {names:?} {}", short(op));
        }
        ctx.op(op);
    }
}

/// Below this many ELEMENTS PER TASK an op is left to one thread. The unit is
/// deliberately per task and not per op: what makes the pool worth waking is the
/// work in each task, and these ops split along a channel or a row axis, where a
/// task can be arbitrarily thin even when the op as a whole is enormous.
const PAR_MIN: usize = 16384;

/// How many tasks a destination of `n` units is worth: one per `PAR_MIN` units,
/// clamped to the pool. Clamping matters for correctness only in that a run on
/// one core then gets no tasks at all instead of tasks that queue behind each
/// other, which would still be correct but strictly slower.
fn nparts(n: usize) -> usize {
    (n / PAR_MIN).clamp(1, rayon::current_num_threads().max(1))
}

/// Split the arena so that the destination range is mutable and EVERY source
/// range is shared, with ONE `split_at_mut`.
///
/// This is the whole of the parallel framing. It replaces an earlier design that
/// cut the arena into fixed windows and let each task look for its own buffers
/// inside its window; that cannot work, and the arithmetic is worth keeping
/// because it is the reason these ops have to be split by direction instead. An
/// op's destination and its sources are separate allocations inside one arena, so
/// a task handed a window of `S` elements can only reach the source element it
/// needs if `S` covers the span from the nearest source to the last destination
/// element - which, for a full-size window, means one task. Measured on the three
/// largest ops: a 16384-element window cannot reach a conv's source plane at all,
/// because the source sits hundreds of thousands of elements away.
///
/// Two disjoint contiguous ranges are always separable by one split - either the
/// destination is entirely before the sources or entirely after them - and the
/// packer guarantees disjointness, because a block is only reused once its
/// previous occupant is dead. `None` is returned for a set that no single split
/// can separate (a source straddling or enveloping the destination); the caller
/// then runs the op serially, which is always correct.
///
/// Nothing here is `unsafe`: the `&mut` and the `&`s are disjoint borrows of
/// disjoint memory produced by one call, so the borrow checker checks this rather
/// than a comment asserting it.
fn par_slice<'a>(
    arena: &'a mut [f32],
    d0: usize,
    dn: usize,
    srcs: &[(usize, usize)],
) -> Option<(&'a mut [f32], Vec<&'a [f32]>)> {
    let d1 = d0 + dn;
    for &(s0, sn) in srcs {
        assert!(s0 + sn <= d0 || d1 <= s0, "par_slice: buffer ranges overlap");
    }
    if srcs.iter().all(|&(s0, sn)| s0 + sn <= d0) {
        // Destination in the right half, every source in the left: split AT the
        // destination, so the right half begins exactly where the destination
        // does and the left half holds nothing but sources.
        let (l, r) = arena.split_at_mut(d0);
        let shares: Vec<&[f32]> = srcs.iter().map(|&(s0, sn)| &l[s0..s0 + sn]).collect();
        return Some((&mut r[..dn], shares));
    }
    if srcs.iter().all(|&(s0, _)| d1 <= s0) {
        // Destination in the left half, every source in the right: split at the
        // destination's end.
        let (l, r) = arena.split_at_mut(d1);
        let shares: Vec<&[f32]> = srcs.iter().map(|&(s0, sn)| &r[s0 - d1..s0 - d1 + sn]).collect();
        return Some((&mut l[d0..d1], shares));
    }
    None
}

/// `par_slice` for exactly one source, which is almost every op. Existential
/// rather than `Option` by `panic!`, because two disjoint contiguous ranges
/// ALWAYS separate at one of their boundaries: reaching the panic means an offset
/// is wrong, and a wrong answer is worse than a loud stop.
fn split2<'a>(
    arena: &'a mut [f32],
    d0: usize,
    dn: usize,
    s0: usize,
    sn: usize,
) -> (&'a mut [f32], &'a [f32]) {
    match par_slice(arena, d0, dn, &[(s0, sn)]) {
        Some((d, srcs)) => (d, srcs[0]),
        None => panic!("split2: source overlaps destination"),
    }
}

/// Run `f` on the destination in chunks, handing each task the offset of its own
/// chunk inside the destination. That offset is the one thing rayon does not
/// supply and the one thing every arm needs: a chunk is a slice of the
/// destination while the source is a slice of a different buffer, so a task has
/// to know which of the source's indices its own elements correspond to. Chunks
/// are contiguous, so chunk `i` starts at `i * per` and the serial path starts at
/// zero; `at` is therefore always a multiple of the chunk length.
fn chunked<S: Sync, F>(dst: &mut [f32], per: usize, srcs: &[S], f: F)
where
    F: Fn(&mut [f32], usize, &[S]) + Send + Sync,
{
    if per == 0 || dst.len() <= per || serial() {
        f(dst, 0, srcs);
        return;
    }
    dst.par_chunks_mut(per)
        .enumerate()
        .for_each(|(i, d)| f(d, i * per, srcs));
}

/// The flat elementwise op: the output index equals the input index, so a task's
/// chunk IS the range of input indices it reads and the bookkeeping is the
/// chunk's own offset. Gelu, Lrelu, Sigmoid and Copy all reduce to this.
fn elem1<F>(arena: &mut [f32], dp: usize, n: usize, sp: usize, f: F)
where
    F: Fn(f32) -> f32 + Send + Sync,
{
    let (d, s) = split2(arena, dp, n, sp, n);
    let per = n.div_ceil(nparts(n));
    chunked(d, per, &[s], |chunk, at, srcs| {
        let s = srcs[0];
        for (i, x) in chunk.iter_mut().enumerate() {
            *x = f(s[at + i]);
        }
    });
}

/// The two-source elementwise form (`a + b`, `a * b`, `v * (u + 1)`).
///
/// Three ranges of one arena, and only the destination needs to be mutable, so
/// both sources come from the ONE shared half. That works whenever the sources
/// are on the same side of the destination, which is what the packer produces -
/// both inputs of an op are live at the same time as its output, so they cannot
/// be placed around it. The remaining case panics rather than computing
/// something plausible and wrong.
fn elem2<F>(arena: &mut [f32], dp: usize, n: usize, x0: usize, y0: usize, f: F)
where
    F: Fn(f32, f32) -> f32 + Send + Sync,
{
    let per = n.div_ceil(nparts(n));
    // Two splits, which is what makes this cover every arrangement the packer can
    // produce with no panic: the first takes the destination's own range out of
    // the arena, the second cuts that range off the front of the remainder. What
    // is left is one shared half BELOW the destination and one ABOVE it, and each
    // source is then taken from whichever half contains it. A single split cannot
    // do this - measured on a real 256x256 plan, the very first Add has one input
    // below the output and one above - and the arms are the reason this has to
    // work for all four orderings rather than the two that looked likely.
    let (below, rest) = arena.split_at_mut(dp);
    let (d, above) = rest.split_at_mut(n);
    let pick = |s0: usize, sn: usize| -> &[f32] {
        if s0 + sn <= dp {
            &below[s0..s0 + sn]
        } else {
            &above[s0 - (dp + n)..s0 - (dp + n) + sn]
        }
    };
    let x = pick(x0, n);
    let y = pick(y0, n);
    chunked(d, per, &[(x, y)], |chunk, at, srcs| {
        let (x, y) = srcs[0];
        for (i, v) in chunk.iter_mut().enumerate() {
            *v = f(x[at + i], y[at + i]);
        }
    });
}

/// The channel-axis form: the destination is `nch` whole units of `per` elements
/// each, the source is `sn` elements, and `xs` is an optional extra shared buffer
/// of its own length (ChanScale's `[c]` scale, which lives in the arena and so has
/// to be reached by the same split). A task owns whole units, so every index it
/// touches is an index the serial loop would have touched, and the unit it starts
/// on is `at / per`.
fn par_ch<F>(
    arena: &mut [f32],
    dp: usize,
    sp: usize,
    sn: usize,
    nch: usize,
    per: usize,
    xs: Option<(usize, usize)>,
    f: F,
) where
    F: Fn(&mut [f32], usize, &[f32], &[f32]) + Send + Sync,
{
    let n = nch * per;
    // `sn` is the source's own length, which is NOT `n` in general: a downsampling
    // op reads a plane larger than the one it writes, and handing a task a slice
    // cut to the destination's length is exactly how it reads off the end. The two
    // are only equal for the ops that preserve their shape.
    //
    // Two splits, for the same reason `elem2` needs two: with an extra buffer
    // (ChanScale's `[c]` scale) there are three ranges, and the packer is free to
    // put one of them on each side of the destination. The destination is lifted
    // out of the arena first, which leaves one shared half below it and one above,
    // and each source is then taken from whichever half holds it.
    let (below, rest) = arena.split_at_mut(dp);
    let (d, above) = rest.split_at_mut(n);
    let pick = |s0: usize, sn: usize| -> &[f32] {
        if s0 + sn <= dp {
            &below[s0..s0 + sn]
        } else {
            &above[s0 - (dp + n)..s0 - (dp + n) + sn]
        }
    };
    let s = pick(sp, sn);
    let x: &[f32] = xs.map(|(x0, xn)| pick(x0, xn)).unwrap_or(&[]);
    // A task must own whole units, so the chunk length is a multiple of `per`;
    // `at / per` is then the first unit of the task and `at % per == 0`. The
    // number of units per task is chosen so each task gets at least PAR_MIN
    // ELEMENTS: sizing the split by the UNIT COUNT instead gives a 32- or
    // 128-channel conv exactly one task, because `nch` never approaches 16384 -
    // and the four convs are most of this model's work.
    let units = (PAR_MIN / per.max(1)).max(1);
    let per_t = per * units.min(nch);
    chunked(d, per_t, &[(s, x)], |chunk, at, srcs| {
        let (s, x) = srcs[0];
        // The task is handed the WHOLE source, not the part of it that lines up
        // with its own chunk: a conv's output channel reads every input channel,
        // so a source slice cut to the task's own length would make `ic * plane`
        // run off the end of its own slice. `at / per` is where the task starts in
        // units, which is all the arms need to find their own outputs.
        f(chunk, at / per, s, x);
    });
}

/// Two disjoint slices of one arena, by element offset. The plan's packer never
/// overlaps buffers that are live at the same op, which is what makes the
/// `assert` a check of the packing rather than of the caller.
fn io2(arena: &mut [f32], d: usize, dn: usize, s: usize, sn: usize) -> (&mut [f32], &[f32]) {
    assert!(d != s, "io2: destination and source are the same arena offset");
    if d < s {
        // Splitting at the source leaves the destination in the left half.
        let (l, r) = arena.split_at_mut(s);
        (&mut l[d..d + dn], &r[..sn])
    } else {
        // Splitting at the DESTINATION leaves the source in the left half, so
        // the two halves come back in the other order: getting this backwards
        // writes the output over its own input and leaves the real destination
        // untouched, which then shows up as a panic somewhere unrelated.
        let (l, r) = arena.split_at_mut(d);
        (&mut r[..dn], &l[s..s + sn])
    }
}

/// Channel LayerNorm, in two parallel passes instead of one serial one.
///
/// The arithmetic is `lightgpu::ops::cpu::channel_layer_norm`'s, term for term and
/// in the same order - one left-to-right pass over a pixel's `c` channels for the
/// mean and the sum of squares, then `(x - mean) * scale * w[ch] + b[ch]` per
/// element, association included - because the twin and this must agree to the
/// last bit. What changes is the SCHEDULE, and it had to change: this op was
/// 2727 ms of a 21 s run (12.9%) and it was the single reason the backend peaked at
/// 1392% of 2400% CPU, since the twin is one serial loop over pixels.
///
/// It cannot be chunked as it stands, and the old comment here said so correctly:
/// a pixel of a plane-major `[c][hw]` tensor is a STRIDED set of `c` elements, so
/// the pixels one task owns are not a contiguous range of the buffer, and a
/// contiguous chunk of the buffer is a set of whole CHANNELS - which cuts the
/// pixel's reduction in half.
///
/// The way out is to split the reduction from the write:
///
///   * pass A is over PIXELS. Each pixel reads its `c` strided values and writes
///     the two scalars it derived - mean and reciprocal standard deviation -
///     interleaved in a small `[hw][2]` table of its own. The source is read-only
///     and shared, and each task owns a disjoint pair range of the table, so
///     there is no strided mutable view and nothing for `unsafe` to do.
///   * pass B is over CHANNELS. `y[ch][p] = (x[ch][p] - mean[p]) * rstd[p] *
///     w[ch] + b[ch]`, which with the table in place needs only its own `p`, so a
///     contiguous range of the destination is a set of its own pixels and the
///     per-channel scale is hoisted out of the inner loop.
///
/// The traffic is the same as the one-pass form - one read of `x`, one write of
/// `y`, plus `2*hw` scalars written and read again - so the cost is one extra pass
/// over a table that is 1/128th of the activation at this model's widest plane.
///
/// The two passes are separated by a scope rather than by a barrier because pass
/// A's immutable borrow of the arena has to end before pass B takes the
/// destination out of it mutably; the compiler enforces that, which is the point.
fn chan_ln(
    arena: &mut [f32],
    dp: usize,
    sp: usize,
    c: usize,
    hw: usize,
    scale: &[f32],
    bias: &[f32],
) {
    let n = c * hw;
    // One allocation for both columns of the table, so the op allocates once.
    let mut table = vec![0.0f32; 2 * hw];
    {
        let s = &arena[sp..sp + n];
        let per = hw.div_ceil(nparts(hw)).max(1);
        let one = |p0: usize, ch: &mut [f32]| {
            let nf = c as f32;
            for (k, pair) in ch.chunks_mut(2).enumerate() {
                let p = p0 + k;
                let mut s1 = 0.0f32;
                let mut s2 = 0.0f32;
                for ic in 0..c {
                    let v = s[ic * hw + p];
                    s1 += v;
                    s2 += v * v;
                }
                let m = s1 / nf;
                let var = s2 / nf - m * m;
                pair[0] = m;
                pair[1] = 1.0 / (var.max(0.0) + EPS).sqrt();
            }
        };
        if serial() || hw <= per {
            let _ = per;
            one(0, &mut table);
        } else {
            table.par_chunks_mut(2 * per).enumerate().for_each(|(i, ch)| one(i * per, ch));
        }
    }
    // Pass B: whole contiguous channels per task, the table read by pixel index.
    // `io2` is what makes the destination mutable and the source shared in one
    // call, and the assertion inside it re-checks that the two do not overlap.
    let (d, s) = io2(arena, dp, n, sp, n);
    let per_ch = (PAR_MIN / hw.max(1)).max(1);
    let unit = hw * per_ch.min(c.max(1));
    chunked(d, unit, &[(s, &table)], |ch_, at, srcs| {
        let (s, table) = srcs[0];
        let ch0 = at / hw;
        for (j, oc) in ch_.chunks_mut(hw).enumerate() {
            let ch = ch0 + j;
            let (w, b) = (scale[ch], bias[ch]);
            let sp_ = &s[ch * hw..(ch + 1) * hw];
            for (p, v) in oc.iter_mut().enumerate() {
                *v = (sp_[p] - table[2 * p]) * table[2 * p + 1] * w + b;
            }
        }
    });
}

struct Ctx<'a> {
    plan: &'a Plan,
    weights: &'a [Vec<f32>],
    arena: &'a mut [f32],
}

impl<'a> Ctx<'a> {
    /// Every op carries its own dimensions, and the arena is one allocation, so
    /// an op whose dimensions disagree with its buffer's shape would quietly read
    /// or write a neighbour instead of failing. This turns that into a named
    /// panic at the op that is actually wrong - the difference between a
    /// five-minute fix and an afternoon of bisecting NaN.
    fn check(&self, op: &Op) {
        if !cfg!(debug_assertions) {
            return;
        }
        let b = |id: usize| self.plan.bufs[id];
        let label = |id: usize| format!("buffer {id} ({:?})", b(id));
        let eq = |op: &str, id: usize, want: (usize, usize, usize), got: crate::model::BufShape| {
            assert_eq!(
                got,
                crate::model::BufShape { c: want.0, h: want.1, w: want.2 },
                "{op}: {id} expects {want:?}, {} says {got:?}",
                label(id)
            );
        };
        match *op {
            Op::Copy { dst, src, n } => {
                assert!(n <= b(dst).len() && n <= b(src).len(), "Copy: n={n} exceeds {}", label(dst));
            }
            Op::Add { dst, a, b: bb, n } | Op::Mul { dst, a, b: bb, n } => {
                assert_eq!(b(dst).len(), n, "Add/Mul: {}", label(dst));
                assert_eq!(b(a).len(), n, "Add/Mul: {}", label(a));
                assert_eq!(b(bb).len(), n, "Add/Mul: {}", label(bb));
            }
            Op::GateApply { dst, u, v, n } => {
                assert_eq!(b(dst).len(), n, "GateApply: {}", label(dst));
                assert_eq!(b(u).len(), n, "GateApply: {}", label(u));
                assert_eq!(b(v).len(), n, "GateApply: {}", label(v));
            }
            Op::Gelu { dst, src, n } | Op::Lrelu { dst, src, n, .. } | Op::Sigmoid { dst, src, n } => {
                assert!(n <= b(dst).len() && n <= b(src).len(), "elemwise: n={n} exceeds {}", label(dst));
            }
            Op::ChanLn { dst, src, c, hw, .. } => {
                // Both sides are [c][hw]; the LayerNorm's scale/bias are held in
                // the weight blob, so nothing else is sized here.
                assert_eq!(b(dst).len(), c * hw, "ChanLn: {}", label(dst));
                assert_eq!(b(src).len(), c * hw, "ChanLn: {}", label(src));
            }
            Op::ChanMean { dst, src, c, hw } => {
                // The mean is ONE value per channel: the destination is [c], not
                // [c][hw], which is the whole point of the op.
                assert_eq!(b(dst).len(), c, "ChanMean: {} is not [c]", label(dst));
                assert_eq!(b(src).len(), c * hw, "ChanMean: {}", label(src));
            }
            Op::ChanScale { dst, src, s: sid, c, hw } => {
                assert_eq!(b(sid).len(), c, "ChanScale: {} is not [c]", label(sid));
                let s = BufShape2 { c, h: 1, w: hw };
                assert!(s.len() <= b(dst).len() && s.len() <= b(src).len(), "ChanScale: {c}x{hw} exceeds {}", label(src));
            }
            Op::Conv1x1 { dst, src, c_in, c_out, h, wd, .. } | Op::Conv3x3 { dst, src, c_in, c_out, h, wd, .. } => {
                eq("conv out", dst, (c_out, h, wd), b(dst));
                eq("conv in", src, (c_in, h, wd), b(src));
            }
            Op::Conv4x4s2 { dst, src, c_in, c_out, h, wd, oh, ow, .. } => {
                eq("conv4x4 out", dst, (c_out, oh, ow), b(dst));
                eq("conv4x4 in", src, (c_in, h, wd), b(src));
            }
            Op::ConvT2x2 { dst, src, c_in, c_out, h, wd, .. } => {
                eq("convt out", dst, (c_out, h * 2, wd * 2), b(dst));
                eq("convt in", src, (c_in, h, wd), b(src));
            }
            Op::GateMm { dst, src, c, outer, inner, .. } => {
                eq("gatemm out", dst, (c, outer, inner), b(dst));
                eq("gatemm in", src, (c, outer, inner), b(src));
            }
            Op::BlockPerm { dst, src, .. } => {
                // Both sides are the same plane, one of them read as a
                // (grid, patch) blocking: the SHAPE differs, the element count
                // does not.
                assert_eq!(b(dst).len(), b(src).len(), "BlockPerm: {dst} and {src} differ in size");
            }
            Op::Resize { dst, src, tmp, c, hin, win, hout, wout } => {
                eq("resize out", dst, (c, hout, wout), b(dst));
                eq("resize in", src, (c, hin, win), b(src));
                eq("resize tmp", tmp, (c, hin, wout), b(tmp));
            }
            Op::DownS { dst, src, c, h, wd, oh, ow, .. } => {
                eq("down out", dst, (c, oh, ow), b(dst));
                eq("down in", src, (c, h, wd), b(src));
            }
        }
    }

    fn op(&mut self, op: &Op) {
        // Split the context once: `plan`/`weights` are shared, `arena` is
        // mutable, and the ops need both at the same time.
        self.check(op);
        let (plan, weights, arena) = (self.plan, self.weights, &mut *self.arena);
        let o = |id: usize| plan.offs[id];
        match *op {
            Op::Copy { dst, src, n } => {
                // A Copy is the one op whose source and destination have the
                // same length and shape, so a task owns a plain flat range of
                // elements and reads the matching range of the source: no anchor,
                // no window, and no index arithmetic beyond the range it owns.
                let (sd, dp) = (o(src), o(dst));
                elem1(arena, dp, n, sd, |x| x);
            }
            Op::Add { dst, a: x, b: y, n } => {
                let (xd, yd, dp) = (o(x), o(y), o(dst));
                elem2(arena, dp, n, xd, yd, |a, b| a + b);
            }
            Op::Mul { dst, a: x, b: y, n } => {
                let (xd, yd, dp) = (o(x), o(y), o(dst));
                elem2(arena, dp, n, xd, yd, |a, b| a * b);
            }
            Op::GateApply { dst, u, v, n } => {
                // `u * (v + 1)`, the order the reference and the kernel both use
                // (`mx_gate_apply`: `y[i] = u[i] * (v[i] + 1.0f)`), so the two
                // agree bit for bit rather than nearly. The operands are NOT
                // interchangeable: `u` is the signal half and `v` is the gate.
                let (ud, vd, dp) = (o(u), o(v), o(dst));
                elem2(arena, dp, n, ud, vd, |u, v| u * (v + 1.0));
            }
            // The elementwise family: an output depends only on the input at the
            // same index, so the axis is irrelevant and a flat range is the
            // simplest correct decomposition. `erf`/`exp` are the same functions
            // the toolkit twins call, so the arithmetic matches the kernel.
            Op::Gelu { dst, src, n } => {
                let (sd, dp) = (o(src), o(dst));
                // The twin's formula, term for term and with the same constant
                // (`t::gelu_erf` uses `FRAC_1_SQRT_2` spelled out), but applied
                // per element so a task can own a flat range of the buffer.
                elem1(arena, dp, n, sd, |x| 0.5 * x * (1.0 + t::erf(x * std::f32::consts::FRAC_1_SQRT_2)));
            }
            Op::Lrelu { dst, src, n, slope } => {
                let (sd, dp) = (o(src), o(dst));
                elem1(arena, dp, n, sd, move |x| if x >= 0.0 { x } else { slope * x });
            }
            Op::Sigmoid { dst, src, n } => {
                let (sd, dp) = (o(src), o(dst));
                elem1(arena, dp, n, sd, |x| 1.0 / (1.0 + (-x).exp()));
            }
            Op::ChanScale { dst, src, s: sid, c, hw } => {
                // `sid` is a `[c]` BUFFER - the CALayer's sigmoid output - not a
                // weight: it is an activation the plan allocated, so this op has
                // two sources and needs the three-buffer form. The split is still
                // by channel: a task owns whole channels, so it reads exactly the
                // `[c]` entries it needs and every accumulation stays inside it.
                let (dp, sd, sp) = (o(dst), o(src), o(sid));
                par_ch(arena, dp, sd, c * hw, c, hw, Some((sp, c)), move |d, c0, s, sc| {
                    // `s` is the WHOLE source and `i` is local to this task's
                    // chunk, so the source index is the task's own offset
                    // (`c0 * hw`) plus the local one. Omitting that offset reads
                    // channel 0 for every task: the serial path has c0 = 0 and
                    // hides it, and the parallel path then disagrees with itself
                    // from the first chunk on.
                    for i in 0..d.len() {
                        d[i] = s[c0 * hw + i] * sc[c0 + i / hw];
                    }
                });
            }
            // `wt` is the GPU's transposed twin of `w`; this executor reads the
            // original because its `ic`-outer, plane-contiguous loop wants the
            // `[c_out][c_in]` layout, and a CPU-only plan does not build it.
            Op::Conv1x1 { dst, src, w, bias, c_in, c_out, h, wd, .. } => {
                let (dp, sd) = (o(dst), o(src));
                let (wt, plane) = (&weights[w], h * wd);
                let bs = bias.map(|b| &weights[b]);
                // One PIXEL RANGE per task holding EVERY output channel, with the
                // input for a block of pixels STAGED in a reusable scratch buffer.
                //
                // This replaces an axpy form that owned one output channel per task
                // and swept the whole input once per output channel: `c_out * c_in *
                // plane` element reads for `c_in * c_out * plane` multiply-adds, so
                // every input element was fetched `c_out` times for `c_out` uses -
                // 2.35 GB of reads per op at the 64->32 @448x640 convs. Staging a
                // block of `PB` pixels of every input channel into a scratch tile
                // (PB * c_in floats, 64 KiB at PB=256 and c_in=64) makes every output
                // channel in the task read that tile instead, so the input is read
                // from memory ONCE and used `c_out` times.
                //
                // Measured in isolation at the real 64->32 @448x640 shape, 24 threads,
                // every variant verified bit-identical to the axpy form: axpy 16.13 ms,
                // PB=16 with 68 tasks 18.40 ms, PB=64 with 68 tasks 6.43 ms, PB=256
                // with 68 tasks 5.59 ms. The PB=16 result is why the first attempt at
                // this idea (a 16-pixel tile) was rejected and recorded as a 13% loss:
                // at that size the per-tile copy from 64 strided planes costs more than
                // the reuse saves. Task count matters too - 68 beats 34 beats 17 - and
                // the axpy form is stuck at `c_out` = 32 tasks because a task cannot
                // share a channel plane with another.
                //
                // The task's destination is `c_out` separate `per`-long ranges, one per
                // channel plane, which no flat chunk of a plane-major buffer can
                // express (a chip of `n * per` elements sits inside ONE plane whenever
                // `per < plane`), so the planes are cut sequentially and each plane's
                // piece is pushed onto its own task's list. That is `c_out` slices
                // borrowed apart once per op, checked by the borrow checker.
                //
                // The accumulation order is untouched - every output element starts at
                // its bias and adds `ic` ascending - so the output is bit-identical.
                let (d, s) = io2(arena, dp, c_out * plane, sd, c_in * plane);
                // The task count is `plane / PB` clamped to the pool, NOT the
                // element-count rule `nparts` uses: `nparts(17920)` is 1, so
                // multiplying it by 4 gave the 112x160 convs - which are most of the
                // conv1x1 time - only FOUR tasks, and the isolated sweep at those
                // shapes says that is the worst region there is: 256->128 @112x160
                // 10.32-11.5 ms at 8 tasks against 5.21-5.30 at 48-96, and 128->128
                // @112x160 5.94-6.19 at 7 against 2.59-2.77 at 96. One task per PB
                // pixels would be 70 tasks for a 112x160 plane, right in the winning
                // band, so that is the rule, capped at 12 per thread so the large
                // planes (1120 blocks at 448x640) do not queue 1120 tasks behind 24.
                let ntask = plane.div_ceil(PB).clamp(1, (12 * rayon::current_num_threads()).max(1));
                let per = plane.div_ceil(ntask);
                let npix = plane.div_ceil(per);
                let mut buckets: Vec<Vec<&mut [f32]>> =
                    (0..npix).map(|_| Vec::with_capacity(c_out)).collect();
                for pl in d.chunks_mut(plane) {
                    for (k, piece) in pl.chunks_mut(per).enumerate() {
                        buckets[k].push(piece);
                    }
                }
                buckets.par_iter_mut().enumerate().for_each(|(k, pieces)| {
                    let p0 = k * per;
                    // The last task's piece can be shorter than `per` (the plane
                    // rarely divides evenly), and the source run has to be cut to
                    // the same length or the last task reads off the end.
                    let np = pieces[0].len();
                    let mut tile = vec![0.0f32; PB * c_in];
                    let mut acc = vec![0.0f32; PB];
                    let mut pt = 0usize;
                    while pt < np {
                        let p = (np - pt).min(PB);
                        for ic in 0..c_in {
                            let row = &s[ic * plane + p0 + pt..ic * plane + p0 + pt + p];
                            tile[ic * PB..ic * PB + p].copy_from_slice(row);
                        }
                        for (oc, out) in pieces.iter_mut().enumerate() {
                            let wp = &wt[oc * c_in..(oc + 1) * c_in];
                            let bv = bs.map(|b| b[oc]).unwrap_or(0.0);
                            acc[..p].fill(bv);
                            for ic in 0..c_in {
                                let wv = wp[ic];
                                let ti = &tile[ic * PB..ic * PB + p];
                                for (a, v) in acc[..p].iter_mut().zip(ti) {
                                    *a += wv * *v;
                                }
                            }
                            out[pt..pt + p].copy_from_slice(&acc[..p]);
                        }
                        pt += p;
                    }
                });
            }
            // `wt` is the GPU's `[tap][ci][oc]` twin of `w`; this executor reads
            // the checkpoint layout because its loop wants `[c_out][c_in][tap]`
            // with the tap innermost, and a CPU-only plan does not build the twin.
            Op::Conv3x3 { dst, src, w, wt: _, bias, c_in, c_out, h, wd } => {
                // `wt` is the GPU's `[tap][ci][oc]` twin of `w`; this executor
                // reads the checkpoint layout, which is already
                // `[c_out][c_in][ky][kx]`, and a CPU-only plan does not build the
                // twin at all.
                let (dp, sd) = (o(dst), o(src));
                let (wt, plane) = (&weights[w], h * wd);
                let bs = bias.map(|b| &weights[b]);
                par_ch(arena, dp, sd, c_in * plane, c_out, plane, None, move |d, c0, s, _| {
                    // One ROW of the output at a time, in a buffer `wd + 2` wide.
                    //
                    // The arm used to loop `for y, for x, for ky, for kx, for ic`,
                    // so an output element re-read all `9 * c_in` of its channel's
                    // weights - 1152 loads per element at the 128->128 convs - and
                    // `s[iy * wd + ix]` was recomputed per tap. Inverting it needs
                    // an accumulator wider than a scalar, and a row is the
                    // smallest one that fits in L1: a whole output plane is 1.1 MiB
                    // at this model's widest activation, a row is 2.5 KiB.
                    //
                    // The nesting is `ky`, `kx`, `ic`, then the row scan, and that
                    // order is deliberate rather than incidental: it visits every
                    // input element that contributes to a given output in exactly
                    // the sequence the old form did, so the sums round identically
                    // and the images are byte-for-byte the ones this backend
                    // produced before. The (ky, ic, kx) nesting a first version used
                    // was 2.9x faster still but shifted 22 of 720000 output samples
                    // by one LSB at 600x400, and a bit-exact backend is worth more
                    // than that.
                    //
                    // What is bought here: each input row is loaded once per
                    // (ky, kx) instead of once per (ky, kx, and every output column
                    // it lands on), each weight is loaded once per row instead of
                    // once per output element, and the three vertical taps are
                    // resolved once per row rather than 3 * wd times.
                    let mut row = vec![0.0f32; wd + 2];
                    for (j, oc_plane) in d.chunks_mut(plane).enumerate() {
                        let oc = c0 + j;
                        let bv = bs.map(|b| b[oc]).unwrap_or(0.0);
                        let wp = &wt[oc * c_in * 9..(oc + 1) * c_in * 9];
                        for y in 0..h {
                            row[..wd + 2].fill(bv);
                            // Which input rows are live is the same set for every
                            // (kx, ic), so it is resolved once per output row: at
                            // the image's top and bottom edges one of the three
                            // vertical taps is out of range and contributes
                            // nothing, which is the padding the op documents.
                            for ky in 0..3 {
                                let iy = y as isize + ky as isize - 1;
                                if iy < 0 || iy >= h as isize {
                                    continue;
                                }
                                let iy = iy as usize;
                                for kx in 0..3 {
                                    let shift = 2 - kx;
                                    for ic in 0..c_in {
                                        let wv = wp[ic * 9 + ky * 3 + kx];
                                        let srow = &s[ic * plane + iy * wd..ic * plane + iy * wd + wd];
                                        // Input column `i` feeds output column
                                        // `i - kx + 1`, which sits at buffer index
                                        // `i - kx + 2` because the buffer is offset
                                        // by one. So `kx = 2` writes `row[i]`,
                                        // `kx = 1` writes `row[i + 1]` (and is the
                                        // only tap that reaches `row[1]`, the
                                        // output's first column) and `kx = 0`
                                        // writes `row[i + 2]`, whose last slot
                                        // `row[wd + 1]` is dropped by the copy.
                                        for (i, v) in srow.iter().enumerate() {
                                            row[i + shift] += wv * *v;
                                        }
                                    }
                                }
                            }
                            let out = &mut oc_plane[y * wd..(y + 1) * wd];
                            out.copy_from_slice(&row[1..wd + 1]);
                        }
                    }
                });
            }
            Op::Conv4x4s2 { dst, src, w, bias, c_in, c_out, h, wd, pad_top, pad_left, oh, ow } => {
                let (dp, sd) = (o(dst), o(src));
                let (wt, plane) = (&weights[w], h * wd);
                let bs = bias.map(|b| &weights[b]);
                let oplane = oh * ow;
                // One output ROW per iteration, with the weight for each
                // (ky, kx, ic) loaded ONCE and swept across the whole output row.
                //
                // The per-output-position gather this replaces has the ky/kx/ic
                // loops innermost, so it re-reads the whole (c_in x 4 x 4) weight
                // set - 8 KiB at c_in=128, D-cache resident but 128 bytes of it
                // touched per output element - for every one of the `oh * ow` output
                // positions. Hoisting those loops above the row turns each weight
                // into a single broadcast against a contiguous run of inputs.
                //
                // Measured in isolation on the shape that matters (the model's two
                // biggest convdowns are 128->128 @112x160, 94 ms each; nothing else
                // is close): 64.48 ms / 36.4 GFLOP/s for the gather against 16.27 ms
                // / 144.3 GFLOP/s for this form, 3.96x. The two shapes probed
                // earlier (8->16 @448x640, 4->8 @896x1280) are NOT in this model and
                // gave a mixed answer for that reason.
                //
                // A two-output-row variant was also tried, since consecutive output
                // rows share two of their four input rows - it was both slower (37.90
                // ms) and wrong, so it is not here.
                //
                // `lo`/`hi` are the output columns whose input window is inside the
                // source: the input column is `2 * ox + kx - pad_left` and has to be
                // in `[0, wd)`. Clamping the sweep to that range is what keeps the
                // padding out of the inner loop - and the accumulation order for a
                // given output element is still (ky, kx, ic), so the result is
                // bit-identical to the gather.
                par_ch(arena, dp, sd, c_in * plane, c_out, oplane, None, move |d, c0, s, _| {
                    for (j, oc_plane) in d.chunks_mut(oplane).enumerate() {
                        let oc = c0 + j;
                        let bv = bs.map(|b| b[oc]).unwrap_or(0.0);
                        let wp = &wt[oc * c_in * 16..(oc + 1) * c_in * 16];
                        let mut acc = vec![0.0f32; ow];
                        for oy in 0..oh {
                            acc[..ow].fill(bv);
                            for ky in 0..4 {
                                let iy = oy * 2 + ky;
                                if iy < pad_top || iy - pad_top >= h {
                                    continue;
                                }
                                let iy = iy - pad_top;
                                for kx in 0..4 {
                                    let lo = (pad_left as isize - kx as isize + 1)
                                        .div_euclid(2)
                                        .max(0) as usize;
                                    // The input column `ox * 2 + kx - pad_left`
                                    // has to be at most `wd - 1`, so
                                    // `ox <= (wd - 1 + pad_left - kx) / 2`.
                                    let hi = ((wd as isize - 1 + pad_left as isize
                                        - kx as isize)
                                        .div_euclid(2)
                                        + 1)
                                        .clamp(0, ow as isize)
                                        as usize;
                                    for ic in 0..c_in {
                                        let wv = wp[ic * 16 + ky * 4 + kx];
                                        let srow =
                                            &s[ic * plane + iy * wd..ic * plane + iy * wd + wd];
                                        for ox in lo..hi {
                                            acc[ox] += wv * srow[ox * 2 + kx - pad_left];
                                        }
                                    }
                                }
                            }
                            oc_plane[oy * ow..(oy + 1) * ow].copy_from_slice(&acc[..ow]);
                        }
                    }
                });
            }
            Op::ConvT2x2 { dst, src, w, bias, c_in, c_out, h, wd } => {
                let (oh, ow) = (h * 2, wd * 2);
                let (wt, plane) = (&weights[w], h * wd);
                let bs = bias.map(|b| &weights[b]);
                let oplane = oh * ow;
                // One output ROW PAIR per iteration, with the four taps of a single
                // input row held in four accumulators, so each source element is
                // loaded ONCE and used four times.
                //
                // The form this replaces rode the scatter the kernel uses - for
                // (y, x, ky, kx) accumulate `s[ic][y][x] * w[ic][oc][ky][kx]` into
                // one output position - which means every source element is loaded
                // four times (once per tap), each time against a different weight,
                // and each output position is touched once per input channel with a
                // read-modify-write. Measured in isolation at the 128->64 @112x160
                // shape, 24 threads: 52.08 ms scatter against 4.54 ms for this form,
                // 11.5x, and all four output pushes are bit-identical to the scatter
                // (the source element's product is added to the same accumulator at
                // the same point in the same ascending `ic` order, and the bias is
                // added last in both).
                //
                // The four accumulators are the four (ky, kx) taps: for a fixed input
                // row `y` the outputs 2y and 2y+1 are affine combinations of the SAME
                // input row, which is what makes one pass over the row enough. A
                // version that also staged a block of input rows for all channels (so
                // the source is read from memory once per block rather than once per
                // output channel) was measured too and did not help - 4.60/5.10/6.30 ms
                // for row blocks of 2/4/8 against 4.54 ms - because the source for one
                // output channel is `c_in * wd` floats per row, and the tile only pays
                // when the reuse across channels outweighs the strided staging reads.
                let (dp, sd) = (o(dst), o(src));
                par_ch(arena, dp, sd, c_in * plane, c_out, oplane, None, move |d, c0, s, _| {
                    for (j, oc_plane) in d.chunks_mut(oplane).enumerate() {
                        let oc = c0 + j;
                        let bv = bs.map(|b| b[oc]).unwrap_or(0.0);
                        let mut a0 = vec![0.0f32; wd];
                        let mut a1 = vec![0.0f32; wd];
                        let mut a2 = vec![0.0f32; wd];
                        let mut a3 = vec![0.0f32; wd];
                        for y in 0..h {
                            a0[..wd].fill(0.0);
                            a1[..wd].fill(0.0);
                            a2[..wd].fill(0.0);
                            a3[..wd].fill(0.0);
                            for ic in 0..c_in {
                                let srow = &s[ic * plane + y * wd..ic * plane + y * wd + wd];
                                let wb = &wt[(ic * c_out + oc) * 4..(ic * c_out + oc) * 4 + 4];
                                for (x, v) in srow.iter().enumerate() {
                                    a0[x] += wb[0] * *v;
                                    a1[x] += wb[1] * *v;
                                    a2[x] += wb[2] * *v;
                                    a3[x] += wb[3] * *v;
                                }
                            }
                            // The four taps of output row `2y` and `2y+1` interleave
                            // into two contiguous strides: tap (ky, kx) lands at
                            // 2x + kx of row 2y + ky.
                            let o0 = 2 * y * ow;
                            let (r0, r1) = oc_plane[o0..o0 + 2 * ow].split_at_mut(ow);
                            for x in 0..wd {
                                r0[2 * x] = bv + a0[x];
                                r0[2 * x + 1] = bv + a1[x];
                                r1[2 * x] = bv + a2[x];
                                r1[2 * x + 1] = bv + a3[x];
                            }
                        }
                    }
                });
            }
            Op::GateMm { dst, src, w, bias, c, outer, inner, mode } => {
                let (dp, sd) = (o(dst), o(src));
                let (wt, _) = (&weights[w], ());
                let bs = bias.map(|b| &weights[b]);
                let pl = outer * inner;
                // The gate replaces the axis it reduces over, so both modes are
                // `c` independent small matmuls with the other axis as the batch,
                // and `c` is therefore the parallel axis: each task owns a range of
                // whole gates and touches only its own planes. The indexing is the
                // same as the GPU's tiled `mx_gate_mm_t0`/`_t1`, deliberately, so a
                // CPU/GPU difference is a real difference.
                // The parallel axis is `c` (the gate), and for mode 1 the `outer`
                // axis is split as well, which is what finally moved this op.
                //
                // One task per gate is only `c` tasks - 32 of them at the widest
                // gate in this model, against 24 threads, so the gate axis alone
                // leaves the pool with nothing to balance and no room for the
                // scheduler. For mode 1 the layout on both sides is `[outer][inner]`,
                // so a contiguous range of `outer` rows is ALSO a contiguous range of
                // both the source and the destination, which means splitting it needs
                // no gather at all: a task owns `nr` rows of one gate. Measured in
                // isolation at c=32, 1120x256, 24 threads: one task per gate 69.35 and
                // 79.86 ms in two runs, rows-16 blocks 56.58 and 57.85 ms, so ~1.25x
                // and 512 tasks instead of 32.
                //
                // Mode 0 is left on the gate axis alone: there the split axis would be
                // the reduced one, so a task would still need every source row.
                // `nr` must DIVIDE `outer`, not merely be its ceil-division: the
                // chunking below cuts the destination every `nr * inner` ELEMENTS,
                // and when `nr * ceil(outer/nr) > outer` a chunk boundary falls
                // inside a gate and the next chunk starts a couple of rows into the
                // FOLLOWING gate - which reads past the end of that gate's source
                // (the panic) and, where it does not, writes the wrong rows. So the
                // split factor is derived from a divisor of `outer`, searched down
                // from the value that gives the pool ~8 tasks per thread. `outer` is
                // 1120, 280 and 64 at this model's gates, all divisible by 8.
                let want = (outer / ((8 * rayon::current_num_threads()) / c.max(1)).max(1)).max(1);
                let nr = if mode == 1 {
                    (1..=want).rev().find(|&r| outer % r == 0).unwrap_or(outer)
                } else {
                    outer
                };
                let rs = outer / nr;
                par_ch(arena, dp, sd, c * pl, c * rs, nr * inner, None, move |d, u0, s, _| {
                    // A chunk is a whole number of units but may span several, so the
                    // unit index inside the chunk is what identifies the gate.
                    for (u, y) in d.chunks_mut(nr * inner).enumerate() {
                        let unit = u0 + u;
                        let ch = unit / rs;
                        let r0 = (unit % rs) * nr;
                        let x = &s[ch * pl..(ch + 1) * pl];
                        let nrow = y.len() / inner;
                        let xr = &x[r0 * inner..];
                        if mode == 0 {
                            // reduce over `outer`: out[r][b] = bias[r] + sum_t w[r][t] x[t][b]
                            //
                            // The row accumulator again, and here the prize is the
                            // READ PATTERN rather than the weight loads. The direct
                            // form has `t_` innermost, so consecutive iterations of
                            // one output element's reduction are `inner` floats
                            // apart - 4.5 KiB at this model's 1120-wide gate, a new
                            // cache line every step, for `outer` steps per output
                            // element. At the 256x1120 gates that is 256 * 1120 * 256
                            // = 73 million strided reads per op; the measured
                            // 125 ms/op is what that costs.
                            //
                            // Making `b` innermost instead turns every read into a
                            // sequential sweep and hoists the weight out of the
                            // loop, and it does not touch the arithmetic: for a
                            // fixed output the taps are still accumulated in
                            // ascending `t_`, starting from the same bias, so the
                            // rounding is identical. `row` is a stack-sized buffer
                            // per task (`inner` floats) rather than the plane, which
                            // is what keeps this in L1.
                            let mut row = vec![0.0f32; inner];
                            for r in 0..outer {
                                let wa = &wt[r * outer..(r + 1) * outer];
                                let bv = bs.map(|b| b[r]).unwrap_or(0.0);
                                row[..inner].fill(bv);
                                for t_ in 0..outer {
                                    let wv = wa[t_];
                                    let xr = &x[t_ * inner..(t_ + 1) * inner];
                                    for (b, v) in xr.iter().enumerate() {
                                        row[b] += wv * *v;
                                    }
                                }
                                y[r * inner..(r + 1) * inner].copy_from_slice(&row);
                            }
                        } else {
                            // reduce over `inner`: out[b][a] = bias[a] + sum_t w[a][t] x[b][t]
                            //
                            // `nrow` rows of THIS gate: `xr` is the first of them and the
                            // destination chunk holds the same rows. Four output channels
                            // are accumulated at once, with their four weight rows STAGED
                            // into a small local buffer first.
                            //
                            // Measured in isolation at c=32, 1120x256, 24 threads, with
                            // eight distinct activations cycled so the caches are cold at
                            // the start of every call and the median of eight reported: the
                            // one-channel form 56.9-60.0 ms, the same with four or eight
                            // interleaved accumulators 54.6-61.4 ms (no better - splitting
                            // one `a_` stream into four does not pay), and the staged-weight
                            // form 43.2-47.6 ms. So the 1.3x comes from the STAGING, not
                            // from the extra accumulators: with the weight rows staged, the
                            // inner loop reads one contiguous run of four rows instead of
                            // interleaving four streams `inner` floats apart, while `x` is
                            // still swept sequentially - which is the property that makes
                            // this form fast at all.
                            //
                            // The obvious alternative, interleaving the accumulators with
                            // `x`'s sweep on the OUTSIDE (for t_ { for i { for u } }), is
                            // 2.3-3.5x SLOWER (141-216 ms): there `x` is stepped with stride
                            // `inner` instead of sequentially, and that costs far more than
                            // the dependency chain it breaks.
                            //
                            // The arithmetic is untouched - each output still starts at its
                            // bias and adds `t_` ascending - so this is bit-identical.
                            const AU: usize = 4;
                            let mut wl = vec![0.0f32; AU * inner];
                            let mut a_ = 0usize;
                            while a_ + AU <= inner {
                                for u in 0..AU {
                                    wl[u * inner..(u + 1) * inner].copy_from_slice(
                                        &wt[(a_ + u) * inner..(a_ + u + 1) * inner],
                                    );
                                }
                                for i in 0..nrow {
                                    let xb = &xr[i * inner..i * inner + inner];
                                    let mut acc = [0.0f32; AU];
                                    for u in 0..AU {
                                        acc[u] = bs.map(|bsv| bsv[a_ + u]).unwrap_or(0.0);
                                    }
                                    for (t_, xv) in xb.iter().enumerate() {
                                        for u in 0..AU {
                                            acc[u] += wl[u * inner + t_] * *xv;
                                        }
                                    }
                                    for u in 0..AU {
                                        y[i * inner + a_ + u] = acc[u];
                                    }
                                }
                                a_ += AU;
                            }
                            while a_ < inner {
                                let wa = &wt[a_ * inner..(a_ + 1) * inner];
                                let bv = bs.map(|bsv| bsv[a_]).unwrap_or(0.0);
                                for i in 0..nrow {
                                    let xb = &xr[i * inner..i * inner + inner];
                                    let mut acc = bv;
                                    for t_ in 0..inner {
                                        acc += wa[t_] * xb[t_];
                                    }
                                    y[i * inner + a_] = acc;
                                }
                                a_ += 1;
                            }
                        }
                    }
                });
            }
            Op::BlockPerm { dst, src, c, h, wd, gh, gw, fh, fw, swap, forward } => {
                let (dp, sd) = (o(dst), o(src));
                let (gsz, psz, plane) = (gh * gw, fh * fw, h * wd);
                par_ch(arena, dp, sd, c * plane, c, plane, None, move |d, c0, s, _| {
                    for (j, dl) in d.chunks_mut(plane).enumerate() {
                        let ch = c0 + j;
                        let pl = &s[ch * plane..(ch + 1) * plane];
                        for gy in 0..gh {
                            for gx in 0..gw {
                                let g = gy * gw + gx;
                                for fy in 0..fh {
                                    for fx in 0..fw {
                                        let p = fy * fw + fx;
                                        let pix = (gy * fh + fy) * wd + gx * fw + fx;
                                        let blk = if swap { p * gsz + g } else { g * psz + p };
                                        if forward {
                                            dl[blk] = pl[pix];
                                        } else {
                                            dl[pix] = pl[blk];
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }
            Op::DownS { dst, src, c, h, wd, stride, off, oh, ow } => {
                let (dp, sd) = (o(dst), o(src));
                let (ip, op_) = (h * wd, oh * ow);
                par_ch(arena, dp, sd, c * ip, c, op_, None, move |d, c0, s, _| {
                    for (j, dl) in d.chunks_mut(op_).enumerate() {
                        let ch = c0 + j;
                        let pl = &s[ch * ip..(ch + 1) * ip];
                        for y in 0..oh {
                            for x in 0..ow {
                                dl[y * ow + x] = pl[(y * stride + off) * wd + x * stride + off];
                            }
                        }
                    }
                });
            }
            Op::ChanLn { dst, src, scale, bias, c, hw } => {
                // `chan_ln` mirrors the toolkit twin term for term but schedules
                // it as two passes instead of one serial loop; see its comment for
                // why the obvious chunked split cannot work and what this costs.
                let (dp, sd) = (o(dst), o(src));
                chan_ln(arena, dp, sd, c, hw, &weights[scale], &weights[bias]);
            }
            Op::ChanMean { dst, src, c, hw } => {
                let (dp, sd) = (o(dst), o(src));
                // A scalar left-to-right sum per channel. That is deliberately NOT
                // `lg_channel_mean`'s order: the kernel reduces with a strided
                // partial per thread plus a halving tree, and mirroring it here
                // would be a different sum - a change to this backend's output, not
                // a bug fix. This op has been through that experiment once already
                // and the tree version was reverted; do not repeat it without
                // re-running the whole-image comparison.
                //
                // Left serial on purpose: the destination is one value per channel
                // (`[c]`, not `[c][hw]`), so the op's whole output is a few dozen
                // numbers and there is nothing here worth waking the pool for.
                let (d, s) = io2(arena, dp, c, sd, c * hw);
                for ch in 0..c {
                    let p = &s[ch * hw..ch * hw + hw];
                    d[ch] = p.iter().sum::<f32>() / hw as f32;
                }
            }
            Op::Resize { dst, src, tmp, c, hin, win, hout, wout } => {
                // Two passes, each over one axis's TAP TABLE. See `ResizeTaps`:
                // the filter is a function of the axis alone, so tabulating it
                // once per axis replaces `c * rows * cols` evaluations of it (two
                // divides, a ceil, a floor, an abs, a compare) with `n_out`.
                let (sp, tp, dp) = (o(src), o(tmp), o(dst));
                let (ip, mp) = (hin * win, hin * wout);
                let tx = ResizeTaps::new(win, wout);
                let ty = ResizeTaps::new(hin, hout);
                // Horizontal: one destination row per (channel, input row). Both
                // the row it reads and the row it writes are contiguous, and the
                // split is by row so that stays true inside every task.
                {
                    let (d, s) = io2(arena, tp, c * mp, sp, c * ip);
                    par_rows(d, wout, &[s], |row, r, srcs| {
                        let sr = &srcs[0][r * win..r * win + win];
                        for (x, v) in row.iter_mut().enumerate() {
                            *v = tx.point(sr, x);
                        }
                    });
                }
                // Vertical: one destination row per (channel, OUTPUT row), whose
                // taps are input ROWS of `tmp` and whose write is one contiguous
                // row. The old form walked a COLUMN of `tmp` and wrote a COLUMN
                // of the destination, one element at a time with `wout` between
                // consecutive steps; this is the same arithmetic in the order the
                // cache wants, and it is why the split can be rows at all.
                {
                    let (d, s) = io2(arena, dp, c * hout * wout, tp, c * mp);
                    par_rows(d, wout, &[s], |row, r, srcs| {
                        let ch = r / hout;
                        let pl = &srcs[0][ch * mp..(ch + 1) * mp];
                        ty.row(pl, wout, r % hout, row);
                    });
                }
            }
        }
    }
}


/// The taps of ONE axis of a bilinear resize, tabulated.
///
/// The CPU twin of `mx_resize_axis`, with everything that does not depend on the
/// data lifted out of the loop. The filter is a function of the AXIS alone - which
/// input indices contribute to each output, and with what weight - so it is built
/// once per axis instead of once per output element. The old form evaluated it
/// inside `resize_line` for every (channel, row, column) of the image: two float
/// divides, a ceil, a floor, an abs and a compare, `c * rows * cols` times, for a
/// table of `n_out` entries. At the 600x400 input's first resize that is
/// 3 * 112 * 640 = 215040 evaluations of 640 distinct ones.
///
/// The jax convention, spelled out because the off-by-half is the whole of it:
/// output `o` samples the input at `(o + 0.5) * inv - 0.5` where `inv = n_in/n_out`,
/// the kernel support is `max(inv, 1)` either side, and input `j` contributes with
/// weight `1 - |sample - j| / kscale`, renormalised by the sum of the weights
/// actually used. Three details are kept EXACTLY as the kernel has them, because
/// each one changes the output:
///
///   * the clamp to `[0, n_in-1]`,
///   * the `|sample - j| < kscale` cutoff - for a downscale this is a real
///     triangular filter, not two-point interpolation,
///   * `if wsum > 0 { acc / wsum } else { acc }`. A zero weight sum with a nonzero
///     accumulator happens at the edges, and dividing unconditionally is a NaN the
///     kernel does not produce.
///
/// The weights are stored FLATTENED, one run per output, each run remembering the
/// input index it starts at. That is what removes the per-element work: the loop
/// body becomes `acc += w[w0 + k] * src[j0 + k]` with no branch, no abs and no
/// divide, and the summation order is still the kernel's (ascending `j`, dropped
/// taps simply absent), so the result is bit-identical rather than merely close.
/// The one thing NOT folded in is the final `/ wsum`: `x * (1/s) != x / s` in
/// general and the two backends have to agree on the last bit, so the division
/// stays.
struct ResizeTaps {
    /// Per output: `(first input index, first weight index)`. A run's length is
    /// the next entry's weight index less this one's, so a run is exactly the set
    /// of taps that survived the cutoff.
    runs: Vec<(u32, u32)>,
    /// Every run's weights, one after another.
    w: Vec<f32>,
    /// Each run's weight sum - the kernel's `wsum`.
    wsum: Vec<f32>,
    /// The axis's INPUT length, which the row form cross-checks its plane against.
    /// The output length is `runs.len()`.
    n_in: usize,
}

impl ResizeTaps {
    fn new(n_in: usize, n_out: usize) -> ResizeTaps {
        let inv = n_in as f32 / n_out as f32;
        let kscale = if inv > 1.0 { inv } else { 1.0 };
        let mut runs = Vec::with_capacity(n_out);
        let mut w = Vec::with_capacity(n_out * 2);
        let mut wsum = Vec::with_capacity(n_out);
        for o in 0..n_out {
            let sample = (o as f32 + 0.5) * inv - 0.5;
            let mut j0 = (sample - kscale).ceil() as isize;
            let mut j1 = (sample + kscale).floor() as isize;
            if j0 < 0 {
                j0 = 0;
            }
            if j1 > n_in as isize - 1 {
                j1 = n_in as isize - 1;
            }
            let first = w.len() as u32;
            let mut s = 0.0f32;
            let mut j = j0;
            while j <= j1 {
                let d = (sample - j as f32).abs() / kscale;
                if d < 1.0 {
                    let wv = 1.0 - d;
                    w.push(wv);
                    s += wv;
                }
                j += 1;
            }
            runs.push((j0.max(0) as u32, first));
            wsum.push(s);
        }
        ResizeTaps { runs, w, wsum, n_in }
    }

    /// One output element of this axis, reading `src` from its own start. This is
    /// the horizontal pass's whole loop body; `o` is the index within the axis.
    fn point(&self, src: &[f32], o: usize) -> f32 {
        let (j0, w0) = self.runs[o];
        let w1 = if o + 1 < self.runs.len() { self.runs[o + 1].1 } else { self.w.len() as u32 };
        let (j0, w0, n) = (j0 as usize, w0 as usize, (w1 - w0) as usize);
        let mut acc = 0.0f32;
        for k in 0..n {
            acc += self.w[w0 + k] * src[j0 + k];
        }
        let s = self.wsum[o];
        if s > 0.0 {
            acc / s
        } else {
            acc
        }
    }

    /// One whole ROW of the vertical axis: the same weights, but the taps are
    /// input ROWS of a plane of `stride` floats, so the output row is a weighted
    /// sum of whole contiguous rows.
    ///
    /// The loop order is the point. The old form walked a COLUMN of the input and
    /// wrote a COLUMN of the output, one element at a time with `stride` between
    /// consecutive steps - 2560 bytes apart at the 640-wide resize, for every
    /// channel and every column. Here the weights are hoisted into registers once
    /// per row and the inner loop over `x` is a straight sweep of two (or three)
    /// contiguous rows.
    fn row(&self, src_plane: &[f32], stride: usize, o: usize, dst: &mut [f32]) {
        debug_assert_eq!(self.n_in, src_plane.len() / stride.max(1), "resize: wrong input row count");
        debug_assert_eq!(dst.len(), stride, "resize: destination row is not one stride long");
        let (j0, w0) = self.runs[o];
        let w1 = if o + 1 < self.runs.len() { self.runs[o + 1].1 } else { self.w.len() as u32 };
        let (j0, w0, n) = (j0 as usize, w0 as usize, (w1 - w0) as usize);
        debug_assert!(n <= Self::MAX_TAPS, "resize: {n} taps exceeds MAX_TAPS");
        let n = n.min(Self::MAX_TAPS);
        // The weights, hoisted out of the inner loop. A small fixed array rather
        // than a slice so the loop below has no length it has to re-read.
        let mut wv = [0.0f32; Self::MAX_TAPS];
        wv[..n].copy_from_slice(&self.w[w0..w0 + n]);
        let s = self.wsum[o];
        for (x, d) in dst.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += wv[k] * src_plane[(j0 + k) * stride + x];
            }
            // The division is on the finished sum, matching the kernel's order; a
            // zero weight sum passes the accumulator through, as there.
            *d = if s > 0.0 { acc / s } else { acc };
        }
    }

    /// The most weights one row's hoist can hold. The jax support is
    /// `2 * max(inv, 1)` wide, so this bounds `ceil(2 * inv)`: every ratio this
    /// model builds (the 4x and 1x pyramid steps) gives 2 or 3.
    const MAX_TAPS: usize = 64;
}

/// Run `f` once per contiguous ROW of `stride` floats in the destination, with the
/// row's GLOBAL index and the shared sources.
///
/// `chunked` cannot express this split: it hands a task a flat range plus an
/// offset, and these two passes index their SOURCE by row, so a task has to know
/// which row it is on rather than how many elements in. Splitting in rows also
/// avoids a real hazard a flat split would have here, since a chunk boundary
/// landing mid-row would leave a task a partial row and a resize row is written by
/// a single call that cannot write half of itself.
fn par_rows<S, F>(dst: &mut [f32], stride: usize, srcs: &[S], f: F)
where
    S: Sync,
    F: Fn(&mut [f32], usize, &[S]) + Send + Sync,
{
    let stride = stride.max(1);
    let rows = dst.len() / stride;
    if serial() || rows <= 1 {
        for (r, row) in dst.chunks_mut(stride).enumerate() {
            f(row, r, srcs);
        }
        return;
    }
    let per = (PAR_MIN / stride).max(1).min(rows);
    dst.par_chunks_mut(per * stride).enumerate().for_each(|(i, chunk)| {
        for (j, row) in chunk.chunks_mut(stride).enumerate() {
            f(row, i * per + j, srcs);
        }
    });
}

/// A one-line summary of an op for the `MAXIM_TRACE` dump.
fn short(op: &Op) -> String {
    match op {
        Op::Conv1x1 { c_in, c_out, h, wd, .. } => format!("conv1x1 {c_in}->{c_out} @{h}x{wd}"),
        Op::Conv3x3 { c_in, c_out, h, wd, .. } => format!("conv3x3 {c_in}->{c_out} @{h}x{wd}"),
        Op::Conv4x4s2 { c_in, c_out, h, wd, .. } => format!("convdown {c_in}->{c_out} @{h}x{wd}"),
        Op::ConvT2x2 { c_in, c_out, h, wd, .. } => format!("convt {c_in}->{c_out} @{h}x{wd}"),
        Op::GateMm { c, outer, inner, mode, .. } => format!("gatemm c={c} {outer}x{inner} mode{mode}"),
        Op::BlockPerm { gh, gw, fh, fw, forward, .. } => format!("blockperm {gh}x{gw} {fh}x{fw} {}", if *forward { "in" } else { "out" }),
        Op::Resize { hin, win, hout, wout, .. } => format!("resize {hin}x{win}->{hout}x{wout}"),
        Op::ChanLn { c, hw, .. } => format!("chanln {c}x{hw}"),
        Op::ChanScale { c, hw, .. } => format!("chanscale {c}x{hw}"),
        Op::ChanMean { c, hw, .. } => format!("chanmean {c}x{hw}"),
        Op::DownS { stride, off, .. } => format!("down stride{stride} off{off}"),
        Op::Copy { n, .. } => format!("copy {n}"),
        Op::Add { n, .. } => format!("add {n}"),
        Op::Mul { n, .. } => format!("mul {n}"),
        Op::GateApply { n, .. } => format!("gate {n}"),
        Op::Gelu { n, .. } => format!("gelu {n}"),
        Op::Lrelu { n, .. } => format!("lrelu {n}"),
        Op::Sigmoid { n, .. } => format!("sigmoid {n}"),
    }
}
