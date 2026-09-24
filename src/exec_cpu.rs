//! The CPU executor: the arithmetic twin of every op, in plain Rust.
//!
//! This is not a fallback. It is the reference the GPU executor is diffed
//! against, and it is what makes the engine testable without a CUDA driver. Ops
//! the toolkit already has a twin for (`lightgpu::ops::cpu`) call it rather than
//! re-implementing it, so the two cannot drift; the ops this engine adds are
//! written below, each next to the kernel it mirrors and in the same order of
//! operations, so a mismatch is a real difference rather than a rounding one.

use crate::model::{BufShape as BufShape2, Op, Plan, EPS};
use lightgpu::ops::cpu as t;

/// Runs the whole plan. `arena` is `plan.arena_len` elements and the input image
/// is expected to be in `plan.input`'s bytes already.
/// Execute exactly one op, so a backend can be compared against this one op by
/// op instead of only at the end of the graph. See `main.rs --verify-gpu`:
/// "the plot is wrong somewhere" is not a usable diagnostic, an op index is.
pub fn step(plan: &Plan, weights: &[Vec<f32>], arena: &mut [f32], i: usize) {
    let mut ctx = Ctx { plan, weights, arena };
    ctx.op(&plan.ops[i]);
}

/// A one-line description of an op, for traces and mismatch reports.
pub fn describe(op: &Op) -> String {
    short(op)
}

pub fn run(plan: &Plan, weights: &[Vec<f32>], arena: &mut [f32]) {
    let trace = std::env::var_os("MAXIM_TRACE").is_some();
    let mut ctx = Ctx { plan, weights, arena };
    for (i, op) in plan.ops.iter().enumerate() {
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

/// Two disjoint slices of one arena, by element offset. The plan's packer never
/// overlaps buffers that are live at the same op, which is what makes the
/// `assert` here a check of the packing rather than of the caller.
/// `dst` and one source, which the plan guarantees do not overlap. Their lengths
/// differ (a conv reads `c_in*h*w` and writes `c_out*h*w`), so both are passed:
/// sharing one length would silently truncate the larger one.
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

/// Destination plus two sources, from raw pointers. A three-way split needs an
/// ordering the borrow checker cannot express; the assertions cover the two ways
/// the plan could alias, and the packer is what guarantees it does not.
fn io3(arena: &mut [f32], d: usize, a: usize, b: usize, n: usize) -> (&mut [f32], &[f32], &[f32]) {
    assert!(d != a && d != b, "op {d} overwrites its own input");
    assert_ne!(a, b, "op reads the same buffer twice");
    let p = arena.as_mut_ptr();
    unsafe {
        (
            std::slice::from_raw_parts_mut(p.add(d), n),
            std::slice::from_raw_parts(p.add(a), n),
            std::slice::from_raw_parts(p.add(b), n),
        )
    }
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
                let (d, s) = io2(arena, o(dst), n, o(src), n);
                d.copy_from_slice(s);
            }
            Op::Add { dst, a, b, n } => {
                let (d, x, y) = io3(arena, o(dst), o(a), o(b), n);
                t::add(x, y, d);
            }
            Op::Mul { dst, a, b, n } => {
                let (d, x, y) = io3(arena, o(dst), o(a), o(b), n);
                for i in 0..n {
                    d[i] = x[i] * y[i];
                }
            }
            Op::GateApply { dst, u, v, n } => {
                let (d, x, y) = io3(arena, o(dst), o(u), o(v), n);
                for i in 0..n {
                    d[i] = x[i] * (y[i] + 1.0);
                }
            }
            Op::Gelu { dst, src, n } => {
                let (d, s) = io2(arena, o(dst), n, o(src), n);
                t::gelu_erf(s, d);
            }
            Op::Lrelu { dst, src, n, slope } => {
                let (d, s) = io2(arena, o(dst), n, o(src), n);
                if slope == 0.0 {
                    t::relu(s, d);
                } else {
                    for i in 0..n {
                        d[i] = if s[i] >= 0.0 { s[i] } else { slope * s[i] };
                    }
                }
            }
            Op::Sigmoid { dst, src, n } => {
                let (d, s) = io2(arena, o(dst), n, o(src), n);
                t::sigmoid(s, d);
            }
            Op::ChanLn { dst, src, scale, bias, c, hw } => {
                let (d, s) = io2(arena, o(dst), c * hw, o(src), c * hw);
                t::channel_layer_norm(s, &weights[scale], &weights[bias], d, c, hw, EPS);
            }
            Op::ChanMean { dst, src, c, hw } => {
                let (d, s) = io2(arena, o(dst), c * hw, o(src), c * hw);
                for ch in 0..c {
                    let p = &s[ch * hw..ch * hw + hw];
                    d[ch] = p.iter().sum::<f32>() / hw as f32;
                }
            }
            Op::ChanScale { dst, src, s: sid, c, hw } => {
                // `sid` is a `[c]` BUFFER - the CALayer's sigmoid output - not a
                // weight: it is an activation the plan allocated, so it is a
                // third arena reader rather than a blob lookup.
                let (d, x, sc) = io3(arena, o(dst), o(src), o(sid), c * hw);
                for i in 0..c * hw {
                    d[i] = x[i] * sc[i / hw];
                }
            }
            Op::Conv1x1 { dst, src, w, bias, c_in, c_out, h, wd } => {
                let (d, s) = io2(arena, o(dst), c_out * h * wd, o(src), c_in * h * wd);
                let wt = &weights[w];
                let bs = bias.map(|b| &weights[b]);
                let plane = h * wd;
                for oc in 0..c_out {
                    let wp = &wt[oc * c_in..(oc + 1) * c_in];
                    let bv = bs.map(|b| b[oc]).unwrap_or(0.0);
                    let op_ = &mut d[oc * plane..(oc + 1) * plane];
                    op_.fill(bv);
                    for ic in 0..c_in {
                        let sp = &s[ic * plane..(ic + 1) * plane];
                        let k = wp[ic];
                        for i in 0..plane {
                            op_[i] += k * sp[i];
                        }
                    }
                }
            }
            Op::Conv3x3 { dst, src, w, bias, c_in, c_out, h, wd } => {
                let (d, s) = io2(arena, o(dst), c_out * h * wd, o(src), c_in * h * wd);
                let wt = &weights[w];
                let bs = bias.map(|b| &weights[b]);
                let plane = h * wd;
                for oc in 0..c_out {
                    let bv = bs.map(|b| b[oc]).unwrap_or(0.0);
                    for y in 0..h {
                        for x in 0..wd {
                            let mut acc = bv;
                            for ky in 0..3 {
                                let iy = y as isize + ky as isize - 1;
                                if iy < 0 || iy >= h as isize {
                                    continue;
                                }
                                for kx in 0..3 {
                                    let ix = x as isize + kx as isize - 1;
                                    if ix < 0 || ix >= wd as isize {
                                        continue;
                                    }
                                    let o = iy as usize * wd + ix as usize;
                                    // w is [c_out][c_in][3][3] with the tap
                                    // outermost, matching the kernel.
                                    for ic in 0..c_in {
                                        acc += wt[((oc * c_in + ic) * 9) + ky * 3 + kx] * s[ic * plane + o];
                                    }
                                }
                            }
                            d[oc * plane + y * wd + x] = acc;
                        }
                    }
                }
            }
            Op::Conv4x4s2 { dst, src, w, bias, c_in, c_out, h, wd, pad_top, pad_left, oh, ow } => {
                let (d, s) = io2(arena, o(dst), c_out * oh * ow, o(src), c_in * h * wd);
                let wt = &weights[w];
                let bs = bias.map(|b| &weights[b]);
                let plane = h * wd;
                for oc in 0..c_out {
                    let bv = bs.map(|b| b[oc]).unwrap_or(0.0);
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let mut acc = bv;
                            for ky in 0..4 {
                                let iy = oy * 2 + ky;
                                if iy < pad_top || iy - pad_top >= h {
                                    continue;
                                }
                                for kx in 0..4 {
                                    let ix = ox * 2 + kx;
                                    if ix < pad_left || ix - pad_left >= wd {
                                        continue;
                                    }
                                    let o = (iy - pad_top) * wd + (ix - pad_left);
                                    for ic in 0..c_in {
                                        acc += wt[((oc * c_in + ic) * 16) + ky * 4 + kx] * s[ic * plane + o];
                                    }
                                }
                            }
                            d[oc * oh * ow + oy * ow + ox] = acc;
                        }
                    }
                }
            }
            Op::ConvT2x2 { dst, src, w, bias, c_in, c_out, h, wd } => {
                let (oh, ow) = (h * 2, wd * 2);
                let (d, s) = io2(arena, o(dst), c_out * oh * ow, o(src), c_in * h * wd);
                let wt = &weights[w];
                let bs = bias.map(|b| &weights[b]);
                let plane = h * wd;
                let oplane = oh * ow;
                // The kernel scatters, and so does this: accumulate in the
                // destination, in the same (ky, kx, oc) order, so the two agree
                // term by term.
                for oc in 0..c_out {
                    let bv = bs.map(|b| b[oc]).unwrap_or(0.0);
                    d[oc * oplane..(oc + 1) * oplane].fill(bv);
                }
                for y in 0..h {
                    for x in 0..wd {
                        for ky in 0..2 {
                            for kx in 0..2 {
                                let o = (y * 2 + ky) * ow + x * 2 + kx;
                                for oc in 0..c_out {
                                    let mut acc = 0.0f32;
                                    for ic in 0..c_in {
                                        acc += s[ic * plane + y * wd + x]
                                            * wt[((ic * c_out + oc) * 2 + ky) * 2 + kx];
                                    }
                                    d[oc * oplane + o] += acc;
                                }
                            }
                        }
                    }
                }
            }
            Op::GateMm { dst, src, w, bias, c, outer, inner, mode } => {
                let (d, s) = io2(arena, o(dst), c * outer * inner, o(src), c * outer * inner);
                let wt = &weights[w];
                let bs = bias.map(|b| &weights[b]);
                // The gate replaces the axis it reduces over, so both modes are
                // `c` independent small matmuls with the other axis as the batch.
                // Same indexing as `mx_gate_mm`, deliberately, so a CPU/GPU
                // difference is a real difference.
                for ch in 0..c {
                    let x = &s[ch * outer * inner..(ch + 1) * outer * inner];
                    let y = &mut d[ch * outer * inner..(ch + 1) * outer * inner];
                    if mode == 0 {
                        // reduce over `outer`: out[a][b] = bias[a] + sum_s w[a][s] x[s][b]
                        for a in 0..outer {
                            let wa = &wt[a * outer..(a + 1) * outer];
                            let bv = bs.map(|b| b[a]).unwrap_or(0.0);
                            let row = &mut y[a * inner..(a + 1) * inner];
                            for b in 0..inner {
                                let mut acc = bv;
                                for s_ in 0..outer {
                                    acc += wa[s_] * x[s_ * inner + b];
                                }
                                row[b] = acc;
                            }
                        }
                    } else {
                        // reduce over `inner`: out[b][a] = bias[a] + sum_s w[a][s] x[b][s]
                        for b in 0..outer {
                            let xr = &x[b * inner..(b + 1) * inner];
                            let yr = &mut y[b * inner..(b + 1) * inner];
                            for a in 0..inner {
                                let wa = &wt[a * inner..(a + 1) * inner];
                                let mut acc = bs.map(|bsv| bsv[a]).unwrap_or(0.0);
                                for s_ in 0..inner {
                                    acc += wa[s_] * xr[s_];
                                }
                                yr[a] = acc;
                            }
                        }
                    }
                }
            }
            Op::BlockPerm { dst, src, c, h, wd, gh, gw, fh, fw, swap, forward } => {
                let (d, s) = io2(arena, o(dst), c * h * wd, o(src), c * h * wd);
                let gsz = gh * gw;
                let psz = fh * fw;
                for ch in 0..c {
                    let pl = &s[ch * h * wd..(ch + 1) * h * wd];
                    let dl = &mut d[ch * h * wd..(ch + 1) * h * wd];
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
            }
            Op::Resize { dst, src, tmp, c, hin, win, hout, wout } => {
                // The horizontal pass first (into `tmp`), then the vertical.
                {
                    let (t_, s) = io2(arena, o(tmp), c * hin * wout, o(src), c * hin * win);
                    for ch in 0..c {
                        for y in 0..hin {
                            let row = &s[ch * hin * win + y * win..ch * hin * win + (y + 1) * win];
                            let out = &mut t_[ch * hin * wout + y * wout..ch * hin * wout + (y + 1) * wout];
                            resize_line(row, out, win, wout);
                        }
                    }
                }
                {
                    let (d, t_) = io2(arena, o(dst), c * hout * wout, o(tmp), c * hin * wout);
                    let mut col_in = vec![0.0f32; hin];
                    let mut col_out = vec![0.0f32; hout];
                    for ch in 0..c {
                        for x in 0..wout {
                            for y in 0..hin {
                                col_in[y] = t_[ch * hin * wout + y * wout + x];
                            }
                            resize_line(&col_in, &mut col_out, hin, hout);
                            for y in 0..hout {
                                d[ch * hout * wout + y * wout + x] = col_out[y];
                            }
                        }
                    }
                }
            }
            Op::DownS { dst, src, c, h, wd, stride, off, oh, ow } => {
                let (d, s) = io2(arena, o(dst), c * oh * ow, o(src), c * h * wd);
                for ch in 0..c {
                    let pl = &s[ch * h * wd..(ch + 1) * h * wd];
                    let dl = &mut d[ch * oh * ow..(ch + 1) * oh * ow];
                    for y in 0..oh {
                        for x in 0..ow {
                            dl[y * ow + x] = pl[(y * stride + off) * wd + x * stride + off];
                        }
                    }
                }
            }
        }
    }
}

/// One axis of a bilinear resize, in jax's convention (see `cuda/maxim.cu` for
/// the derivation). Mirrors `mx_resize_axis` term for term.
fn resize_line(src: &[f32], dst: &mut [f32], n_in: usize, n_out: usize) {
    let inv = n_in as f32 / n_out as f32;
    let kscale = if inv > 1.0 { inv } else { 1.0 };
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
        let mut wsum = 0.0f32;
        let mut acc = 0.0f32;
        let mut j = j0;
        while j <= j1 {
            let d = (sample - j as f32).abs() / kscale;
            if d < 1.0 {
                let wv = 1.0 - d;
                wsum += wv;
                acc += wv * src[j as usize];
            }
            j += 1;
        }
        dst[o] = if wsum > 0.0 { acc / wsum } else { acc };
    }
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
