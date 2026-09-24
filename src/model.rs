//! The MAXIM graph: a flat list of semantic ops over one activation arena.
//!
//! The plan is built for ONE input shape, because every buffer size is fixed by
//! it and re-deriving them at run time is where an engine like this accumulates
//! special cases. Building it also resolves and transposes every parameter, so a
//! missing or mis-shaped weight fails at startup rather than mid-run, and the
//! same op list drives both backends: the CPU executor matches on the op and
//! calls the arithmetic twin, the GPU executor matches on the same op and
//! launches the kernel beside it.
//!
//! The graph is transcribed from `maxim/models/maxim.py`. `tools/reference.py` is
//! a torch transcription of the same model that has been diffed against the
//! original JAX one (85.75 dB) and against the officially published LOL result
//! images (51.2 dB, its own PSNR-vs-GT 21.11 against the published 20.98), and
//! `tests/graph.rs` checks the parts of this file that are easy to get wrong.
//!
//! Four details are load-bearing:
//!
//! * `Conv_down` is 4x4 stride 2 with padding='SAME' - ceil(h/2) - and NOT the
//!   toolkit's `lg_conv4x4s4` stride-4 patch embed.
//! * flax's `ConvTranspose` is the adjoint of its `Conv`, so its kernel is
//!   spatially reversed for a scatter (`weights::convt2x2`).
//! * the multi-scale input pyramid uses jax's `nearest`, which for a 2x
//!   downsample keeps the ODD samples, not the even ones (`Op::DownS`).
//! * the gMLP's two gating units act on different axes of the same blocked
//!   layout, and which is which is fixed by the layout: see `Graph::gating`.

use crate::config::Config;
use crate::image::same_pad;
use crate::weights::Weights;
use crate::Error;

/// LayerNorm epsilon: flax's `nn.LayerNorm` default, used by every LayerNorm in
/// the model (the gMLP, the RCAB and the cross-gating norms alike).
pub const EPS: f32 = 1e-6;

/// A `[c][h][w]` plane triple: the shape of a buffer, the unit the arena is
/// measured in and what `--dump` writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufShape {
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl BufShape {
    pub fn len(&self) -> usize {
        self.c * self.h * self.w
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The plane size, `h*w`: the unit the channel-axis ops stride in.
    pub fn hw(&self) -> usize {
        self.h * self.w
    }
}

/// A semantic op. Ops name buffers by index, never by host offset, so the
/// builder is free to move them (which is what the arena packing does) and a
/// buffer may be a *view* into another one's channels - which is how the
/// channel-axis splits and concatenations in this model cost nothing.
#[derive(Clone, Debug)]
pub enum Op {
    Copy { dst: usize, src: usize, n: usize },
    Add { dst: usize, a: usize, b: usize, n: usize },
    Mul { dst: usize, a: usize, b: usize, n: usize },
    /// `u * (v + 1)`: the gMLP gate, in place in `u`'s buffer when the builder
    /// asks for it (`dst == u`).
    GateApply { dst: usize, u: usize, v: usize, n: usize },
    /// `y = gelu(x)`, the erf form, matching the reference's `F.gelu` default.
    Gelu { dst: usize, src: usize, n: usize },
    Lrelu { dst: usize, src: usize, n: usize, slope: f32 },
    Sigmoid { dst: usize, src: usize, n: usize },
    /// The channel-axis LayerNorm: `y[c][p] = (x[c][p] - mean_p) * rsqrt(var_p)
    /// * scale[c] + bias[c]`. `hw` is the plane size, so this op serves both an
    /// NCHW tensor and a blocked gMLP tensor (whose channel axis is contiguous
    /// too, with the block position as the "spatial" axis).
    ChanLn { dst: usize, src: usize, scale: usize, bias: usize, c: usize, hw: usize },
    /// The channel mean, the CALayer's global average pool.
    ChanMean { dst: usize, src: usize, c: usize, hw: usize },
    /// `out[c][p] = in[c][p] * s[c]`: the CALayer's excitation.
    ChanScale { dst: usize, src: usize, s: usize, c: usize, hw: usize },
    Conv1x1 { dst: usize, src: usize, w: usize, bias: Option<usize>, c_in: usize, c_out: usize, h: usize, wd: usize },
    Conv3x3 { dst: usize, src: usize, w: usize, bias: Option<usize>, c_in: usize, c_out: usize, h: usize, wd: usize },
    Conv4x4s2 { dst: usize, src: usize, w: usize, bias: Option<usize>, c_in: usize, c_out: usize, h: usize, wd: usize, pad_top: usize, pad_left: usize, oh: usize, ow: usize },
    ConvT2x2 { dst: usize, src: usize, w: usize, bias: Option<usize>, c_in: usize, c_out: usize, h: usize, wd: usize },
    /// A Dense over the CONTIGUOUS axis of a `[c][outer][inner]` tensor, the
    /// form every gMLP gating matmul reduces to (see `Graph::gating`).
    GateMm { dst: usize, src: usize, w: usize, bias: Option<usize>, c: usize, outer: usize, inner: usize, mode: u8 },
    /// Space <-> block permutation. `swap` puts the patch axis first; `forward`
    /// gathers, otherwise it scatters and drops blocks that fall outside.
    BlockPerm { dst: usize, src: usize, c: usize, h: usize, wd: usize, gh: usize, gw: usize, fh: usize, fw: usize, swap: bool, forward: bool },
    /// Bilinear resize in jax's convention, as two axis passes: `tmp` is
    /// `[c][hin][wout]`, the horizontal pass's output.
    Resize { dst: usize, src: usize, tmp: usize, c: usize, hin: usize, win: usize, hout: usize, wout: usize },
    /// Nearest subsample with a stride and offset: `out[y][x] = in[2y+o][2x+o]`
    /// for the input pyramid's 2x and 4x levels.
    DownS { dst: usize, src: usize, c: usize, h: usize, wd: usize, stride: usize, off: usize, oh: usize, ow: usize },
}

impl Op {
    /// The buffers this op reads or writes. Weights are NOT included: they live
    /// in `Plan::weights`, not in the arena, and are never reused.
    pub fn operands(&self, out: &mut Vec<usize>) {
        match *self {
            Op::Copy { dst, src, .. } => out.extend([dst, src]),
            Op::Add { dst, a, b, .. } | Op::Mul { dst, a, b, .. } => out.extend([dst, a, b]),
            Op::GateApply { dst, u, v, .. } => out.extend([dst, u, v]),
            Op::Gelu { dst, src, .. }
            | Op::Lrelu { dst, src, .. }
            | Op::Sigmoid { dst, src, .. }
            | Op::ChanLn { dst, src, .. }
            | Op::ChanMean { dst, src, .. }
            | Op::Conv1x1 { dst, src, .. }
            | Op::Conv3x3 { dst, src, .. }
            | Op::Conv4x4s2 { dst, src, .. }
            | Op::ConvT2x2 { dst, src, .. }
            | Op::GateMm { dst, src, .. }
            | Op::BlockPerm { dst, src, .. }
            | Op::DownS { dst, src, .. } => out.extend([dst, src]),
            // `s` is the gate, a `[c]` buffer the plan allocated - NOT a weight.
            Op::ChanScale { dst, src, s, .. } => out.extend([dst, src, s]),
            // `tmp` is the horizontal pass of the two-pass resize.
            Op::Resize { dst, src, tmp, .. } => out.extend([dst, src, tmp]),
        }
    }
}

// ------------------------------------------------------------------- packing

/// Every buffer an op reads or writes, for the liveness computation.
///
/// This delegates to [`Op::operands`] so there is exactly ONE list of an op's
/// buffer operands. That matters more than it looks: an operand missing here is
/// invisible - the packer simply believes the buffer dies early and reuses its
/// storage - and the failure only shows up as a wrong number several ops later,
/// or not at all, depending on whether the buffer happens to be a named dump.
fn op_touch(op: &Op, out: &mut Vec<usize>) {
    op.operands(out);
}

/// The storage a buffer needs, and where it sits inside the arena.
///
/// Most buffers own their bytes. A buffer created by `Builder::view` is a
/// channel range of another buffer instead, which is how the model's channel
/// splits and concatenations are free: `basic` resolves a view to the buffer
/// that actually holds the data.
pub struct Layout {
    pub off: Vec<usize>,
    pub base: Vec<usize>,
    pub arena_len: usize,
}

impl Layout {
    /// The arena offset a buffer's first element lands on.
    pub fn at(&self, id: usize) -> usize {
        self.off[id]
    }
}

// -------------------------------------------------------------------- graph

/// Builds the plan. One method per block of the reference model, with the same
/// names, so this file can be read next to `tools/reference.py`.
pub struct Builder<'a> {
    w: &'a Weights,
    cfg: Config,
    weights: Vec<Vec<f32>>,
    bufs: Vec<BufShape>,
    labels: Vec<String>,
    alias: Vec<Option<(usize, usize)>>,
    ops: Vec<Op>,
    named: Vec<(String, usize)>,
    up_index: usize,
    /// Whether `rec` records anything. Clearing it produces the SAME graph with
    /// different live ranges, which is how the packing transparency test below
    /// checks that no operand is missing from `Op::operands`.
    record: bool,
    /// Whether the deep (per-block-internal) dumps are recorded too. They are
    /// what localises a divergence to a single line of a block, but a named
    /// buffer has to stay live to the end of the run for `--dump` to snapshot
    /// it, and naming hundreds of them costs a large multiple of the arena. So
    /// they are opt-in: `MAXIM_DEEP_DUMPS=1`.
    deep: bool,
}

impl<'a> Builder<'a> {
    pub fn new(w: &'a Weights, cfg: Config) -> Builder<'a> {
        Builder {
            w,
            cfg,
            weights: Vec::new(),
            bufs: Vec::new(),
            labels: Vec::new(),
            alias: Vec::new(),
            ops: Vec::new(),
            named: Vec::new(),
            up_index: 0,
            record: true,
            deep: std::env::var_os("MAXIM_DEEP_DUMPS").is_some(),
        }
    }

    fn buf(&mut self, c: usize, h: usize, w: usize, label: &str) -> usize {
        self.bufs.push(BufShape { c, h, w });
        self.labels.push(label.to_string());
        self.alias.push(None);
        self.bufs.len() - 1
    }

    /// A channel range of another buffer. No storage; the ops that use it see
    /// the parent's bytes at an offset.
    fn view(&mut self, base: usize, ch: usize, c: usize, label: &str) -> usize {
        let s = self.bufs[base];
        self.bufs.push(BufShape { c, h: s.h, w: s.w });
        self.labels.push(label.to_string());
        self.alias.push(Some((base, ch * s.h * s.w)));
        self.bufs.len() - 1
    }

    fn rec(&mut self, name: &str, id: usize) {
        if self.record {
            self.named.push((name.to_string(), id));
        }
    }

    /// A per-block-INTERNAL activation: recorded only when the deep dumps are on.
    fn deep_rec(&mut self, name: &str, id: usize) {
        if self.deep {
            self.rec(name, id);
        }
    }

    /// The same graph with no named activations, for the packing test.
    pub fn without_dumps(mut self) -> Builder<'a> {
        self.record = false;
        self
    }

    fn weight(&mut self, v: Vec<f32>) -> usize {
        self.weights.push(v);
        self.weights.len() - 1
    }

    fn own(&mut self, name: &str) -> Result<usize, Error> {
        let v = self.w.owned(name)?;
        Ok(self.weight(v))
    }

    /// A bias that may be absent from the checkpoint, in which case a zero
    /// vector stands in: flax's `use_bias=False` is the same arithmetic with a
    /// zero bias, and one code path is better than two.
    fn own_or_zeros(&mut self, name: &str, n: usize) -> Result<usize, Error> {
        if self.w.has(name) {
            self.own(name)
        } else {
            Ok(self.weight(vec![0.0; n]))
        }
    }

    /// A gating Dense weight as `[c_out][c_in]`, the layout `mx_gate_mm` walks.
    fn dense(&mut self, name: &str) -> Result<usize, Error> {
        let v = self.w.dense(name)?;
        Ok(self.weight(v))
    }

    fn ln_pair(&mut self, prefix: &str, c: usize) -> Result<(usize, usize), Error> {
        let s = self.own(&format!("{prefix}/scale"))?;
        let b = self.own_or_zeros(&format!("{prefix}/bias"), c)?;
        Ok((s, b))
    }

    fn chan_ln(&mut self, dst: usize, src: usize, prefix: &str, c: usize, hw: usize) -> Result<(), Error> {
        let (s, b) = self.ln_pair(prefix, c)?;
        self.ops.push(Op::ChanLn { dst, src, scale: s, bias: b, c, hw });
        Ok(())
    }

    /// A 1x1 conv (or a Dense, which `weights::conv1x1` transposes into the same
    /// thing).
    fn conv1x1(&mut self, dst: usize, src: usize, prefix: &str, c_in: usize, c_out: usize, h: usize, wd: usize) -> Result<(), Error> {
        let k = self.w.conv1x1(&format!("{prefix}/kernel"))?;
        let w = self.weight(k);
        let bias = if self.w.has(&format!("{prefix}/bias")) {
            Some(self.own(&format!("{prefix}/bias"))?)
        } else {
            None
        };
        self.ops.push(Op::Conv1x1 { dst, src, w, bias, c_in, c_out, h, wd });
        Ok(())
    }

    fn conv3x3(&mut self, dst: usize, src: usize, prefix: &str, c_in: usize, c_out: usize, h: usize, wd: usize) -> Result<(), Error> {
        let k = self.w.convkxk(&format!("{prefix}/kernel"))?;
        let w = self.weight(k);
        let bias = if self.w.has(&format!("{prefix}/bias")) {
            Some(self.own(&format!("{prefix}/bias"))?)
        } else {
            None
        };
        self.ops.push(Op::Conv3x3 { dst, src, w, bias, c_in, c_out, h, wd });
        Ok(())
    }

    /// `Conv_down`: 4x4 stride 2 with SAME padding, halving the resolution.
    fn conv_down(&mut self, dst: usize, src: usize, prefix: &str, c_in: usize, c_out: usize, h: usize, wd: usize) -> Result<(), Error> {
        let k = self.w.convkxk(&format!("{prefix}/kernel"))?;
        let w = self.weight(k);
        let bias = if self.w.has(&format!("{prefix}/bias")) {
            Some(self.own(&format!("{prefix}/bias"))?)
        } else {
            None
        };
        let oh = (h + 1) / 2;
        let ow = (wd + 1) / 2;
        let (pt, _) = same_pad(h, 2, 4);
        let (pl, _) = same_pad(wd, 2, 4);
        self.ops.push(Op::Conv4x4s2 { dst, src, w, bias, c_in, c_out, h, wd, pad_top: pt, pad_left: pl, oh, ow });
        Ok(())
    }

    /// `ConvT_up`: 2x2 stride 2 transposed, doubling the resolution.
    fn convt_up(&mut self, dst: usize, src: usize, prefix: &str, c_in: usize, c_out: usize, h: usize, wd: usize) -> Result<(), Error> {
        let k = self.w.convt2x2(&format!("{prefix}/kernel"))?;
        let w = self.weight(k);
        let bias = if self.w.has(&format!("{prefix}/bias")) {
            Some(self.own(&format!("{prefix}/bias"))?)
        } else {
            None
        };
        self.ops.push(Op::ConvT2x2 { dst, src, w, bias, c_in, c_out, h, wd });
        Ok(())
    }

    fn gelu(&mut self, dst: usize, src: usize, n: usize) {
        self.ops.push(Op::Gelu { dst, src, n });
    }

    fn add(&mut self, dst: usize, a: usize, b: usize, n: usize) {
        self.ops.push(Op::Add { dst, a, b, n });
    }

    fn mul(&mut self, dst: usize, a: usize, b: usize, n: usize) {
        self.ops.push(Op::Mul { dst, a, b, n });
    }

    /// `jax.image.resize(method='nearest')` for an integer downsample: output
    /// `i` reads input `floor((i + 0.5) * factor)`, i.e. stride `factor` with
    /// offset `factor / 2`. For 2x that is the ODD samples, and for 4x it is
    /// 2, 6, 10, ... - not the odd samples of the odd samples, which is why the
    /// engine offers one strided kernel instead of two halvings.
    fn nearest_down(&mut self, dst: usize, src: usize, c: usize, h: usize, w: usize, factor: usize) {
        let (oh, ow) = (h / factor, w / factor);
        self.ops.push(Op::DownS { dst, src, c, h, wd: w, stride: factor, off: factor / 2, oh, ow });
    }

    /// Bilinear resize in jax's convention, as two axis passes; a no-op when the
    /// shape does not change (which the reference also short-circuits, and which
    /// happens at every level where the two scales already agree).
    fn resize(&mut self, dst: usize, src: usize, c: usize, hin: usize, win: usize, hout: usize, wout: usize) {
        if hin == hout && win == wout {
            self.ops.push(Op::Copy { dst, src, n: c * hin * win });
            return;
        }
        let tmp = self.buf(c, hin, wout, "resize_tmp");
        self.ops.push(Op::Resize { dst, src, tmp, c, hin, win, hout, wout });
    }


}
impl<'a> Builder<'a> {
    // -- leaf blocks ------------------------------------------------------

    /// CALayer: squeeze-and-excitation channel attention, `x * sigmoid(...)`.
    fn calayer(&mut self, prefix: &str, x: usize, c: usize, h: usize, wd: usize, red: usize) -> Result<usize, Error> {
        let hw = h * wd;
        let mean = self.buf(c, 1, 1, "ca_mean");
        self.ops.push(Op::ChanMean { dst: mean, src: x, c, hw });
        self.deep_rec(&format!("{prefix}_mean"), mean);
        let sq = self.buf(c / red, 1, 1, "ca_squeeze");
        self.conv1x1(sq, mean, &format!("{prefix}/Conv_0"), c, c / red, 1, 1)?;
        self.deep_rec(&format!("{prefix}_sq"), sq);
        let rl = self.buf(c / red, 1, 1, "ca_relu");
        // relu is lrelu with slope 0, which is what the toolkit's `lg_lrelu`
        // offers; there is no separate relu kernel to call.
        self.ops.push(Op::Lrelu { dst: rl, src: sq, n: c / red, slope: 0.0 });
        self.deep_rec(&format!("{prefix}_relu"), rl);
        let ex = self.buf(c, 1, 1, "ca_excite");
        self.conv1x1(ex, rl, &format!("{prefix}/Conv_1"), c / red, c, 1, 1)?;
        self.deep_rec(&format!("{prefix}_excite"), ex);
        let sg = self.buf(c, 1, 1, "ca_sigmoid");
        self.ops.push(Op::Sigmoid { dst: sg, src: ex, n: c });
        self.deep_rec(&format!("{prefix}_sigmoid"), sg);
        let out = self.buf(c, h, wd, "ca_out");
        self.ops.push(Op::ChanScale { dst: out, src: x, s: sg, c, hw });
        Ok(out)
    }

    /// RCAB: LayerNorm, 3x3 conv, leaky relu, 3x3 conv, channel attention.
    fn rcab(&mut self, prefix: &str, x: usize, c: usize, h: usize, wd: usize, red: usize) -> Result<usize, Error> {
        let y = self.buf(c, h, wd, "rcab_ln");
        self.chan_ln(y, x, &format!("{prefix}/LayerNorm"), c, h * wd)?;
        self.deep_rec(&format!("{prefix}_ln"), y);
        let t = self.buf(c, h, wd, "rcab_conv1");
        self.conv3x3(t, y, &format!("{prefix}/conv1"), c, c, h, wd)?;
        self.deep_rec(&format!("{prefix}_conv1"), t);
        let r = self.buf(c, h, wd, "rcab_lrelu");
        self.ops.push(Op::Lrelu { dst: r, src: t, n: c * h * wd, slope: 0.2 });
        self.deep_rec(&format!("{prefix}_lrelu"), r);
        let t2 = self.buf(c, h, wd, "rcab_conv2");
        self.conv3x3(t2, r, &format!("{prefix}/conv2"), c, c, h, wd)?;
        self.deep_rec(&format!("{prefix}_conv2"), t2);
        let ca = self.calayer(&format!("{prefix}/channel_attention"), t2, c, h, wd, red)?;
        self.deep_rec(&format!("{prefix}_ca"), ca);
        let out = self.buf(c, h, wd, "rcab_out");
        self.add(out, ca, x, c * h * wd);
        Ok(out)
    }

    /// RDCAB: the bottleneck's residual dense channel attention block.
    fn rdcab(&mut self, prefix: &str, x: usize, c: usize, h: usize, wd: usize, red: usize) -> Result<usize, Error> {
        let y = self.buf(c, h, wd, "rdcab_ln");
        self.chan_ln(y, x, &format!("{prefix}/LayerNorm"), c, h * wd)?;
        let c1 = self.buf(c, h, wd, "rdcab_dense0");
        self.conv1x1(c1, y, &format!("{prefix}/channel_mixing/Dense_0"), c, c, h, wd)?;
        let g = self.buf(c, h, wd, "rdcab_gelu");
        self.gelu(g, c1, c * h * wd);
        let c2 = self.buf(c, h, wd, "rdcab_dense1");
        self.conv1x1(c2, g, &format!("{prefix}/channel_mixing/Dense_1"), c, c, h, wd)?;
        let ca = self.calayer(&format!("{prefix}/channel_attention"), c2, c, h, wd, red)?;
        let out = self.buf(c, h, wd, "rdcab_out");
        self.add(out, ca, x, c * h * wd);
        Ok(out)
    }

    /// GridGatingUnit: `u * (layer_norm(v) ~ Dense over the GRID axis + 1)`.
    ///
    /// `x` is the blocked `[c][G][P]` tensor and `u`/`v` are its two channel
    /// halves; the gate writes into `g` and the product goes back into `u`'s
    /// buffer (`dst`), which is the concatenation's low half.
    fn gating(
        &mut self,
        prefix: &str,
        u: usize,
        v: usize,
        dst: usize,
        c: usize,
        gg: usize,
        pp: usize,
        over_grid: bool,
    ) -> Result<(), Error> {
        let n = c * gg * pp;
        let ln = self.buf(c, gg, pp, "gmlp_gate_ln");
        self.chan_ln(ln, v, &format!("{prefix}/intermediate_layernorm"), c, gg * pp)?;
        let g = self.buf(c, gg, pp, "gmlp_gate");
        let w = self.dense(&format!("{prefix}/Dense_0/kernel"))?;
        // The Dense replaces the axis it reduces over, so its bias is sized by
        // that axis: the grid axis for the grid gate, the patch axis for the
        // block one.
        let nb = if over_grid { gg } else { pp };
        let b = self.own_or_zeros(&format!("{prefix}/Dense_0/bias"), nb)?;
        let mode = if over_grid { 0 } else { 1 };
        self.ops.push(Op::GateMm { dst: g, src: ln, w, bias: Some(b), c, outer: gg, inner: pp, mode });
        self.ops.push(Op::GateApply { dst, u, v: g, n });
        Ok(())
    }

    /// One half of a multi-axis gMLP layer: block the plane at `cells`, mix
    /// along one axis, project, unblock, and add the residual. The result lands
    /// in `dst` (the caller's own buffer, so the two halves can share one).
    ///
    /// `over_grid` picks the axis: the grid gMLP blocks the plane at the grid
    /// size and mixes over the grid axis, the block gMLP blocks it at the block
    /// size and mixes over the within-block axis. Both use `Op::BlockPerm` with
    /// the patch axis contiguous, so the difference is only which axis the Dense
    /// reduces - and that is `mx_gate_mm`'s `mode`.
    fn gmlp_axis(
        &mut self,
        prefix: &str,
        x: usize,
        dst: usize,
        c: usize,
        h: usize,
        wd: usize,
        size: usize,
        over_grid: bool,
    ) -> Result<(), Error> {
        // The two callers pass DIFFERENT KINDS of number, and that asymmetry is
        // in the reference: `grid_gmlp`'s argument is a count of cells per axis
        // (it computes gh, gw = grid_size and derives fh, fw), while
        // `block_gmlp`'s is the block size in PIXELS (it computes fh, fw =
        // block_size and derives gh, gw). Treating both as cell counts swaps the
        // two axes, which is invisible to the shape checker because the op stays
        // internally consistent.
        let (gh, gw, fh, fw) = if over_grid {
            (size, size, h / size, wd / size)
        } else {
            (h / size, wd / size, size, size)
        };
        if gh == 0 || gw == 0 || fh == 0 || fw == 0 {
            return Err(format!(
                "{prefix}: a {size}-{kind} does not fit a {h}x{wd} feature map                  (cells {gh}x{gw}, {fh}x{fw} pixels) - the input is too small for this model",
                kind = if over_grid { "cell grid" } else { "pixel block" }
            )
            .into());
        }
        let (gg, pp) = (gh * gw, fh * fw);
        let xb = self.buf(c, gg, pp, "gmlp_blocked");
        self.ops.push(Op::BlockPerm { dst: xb, src: x, c, h, wd, gh, gw, fh, fw, swap: false, forward: true });
        self.deep_rec(&format!("{prefix}_blocked"), xb);
        let ln = self.buf(c, gg, pp, "gmlp_ln");
        self.chan_ln(ln, xb, &format!("{prefix}/LayerNorm"), c, gg * pp)?;
        self.deep_rec(&format!("{prefix}_ln"), ln);
        let t = self.buf(2 * c, gg, pp, "gmlp_inproject");
        self.conv1x1(t, ln, &format!("{prefix}/in_project"), c, 2 * c, gg, pp)?;
        self.deep_rec(&format!("{prefix}_inproject"), t);
        let tg = self.buf(2 * c, gg, pp, "gmlp_gelu");
        self.gelu(tg, t, 2 * c * gg * pp);
        self.deep_rec(&format!("{prefix}_gelu"), tg);
        let u = self.view(tg, 0, c, "gmlp_u");
        let v = self.view(tg, c, c, "gmlp_v");
        let gated = self.buf(c, gg, pp, "gmlp_gated");
        let unit = if over_grid { "GridGatingUnit" } else { "BlockGatingUnit" };
        let gprefix = format!("{prefix}/{unit}");
        self.gating(
            &gprefix,
            u,
            v,
            gated,
            c,
            gg,
            pp,
            over_grid,
        )?;
        self.deep_rec(&format!("{prefix}_gated"), gated);
        let proj = self.buf(c, gg, pp, "gmlp_outproject");
        self.conv1x1(proj, gated, &format!("{prefix}/out_project"), c, c, gg, pp)?;
        self.deep_rec(&format!("{prefix}_proj"), proj);
        let mixed = self.buf(c, gg, pp, "gmlp_residual");
        self.add(mixed, proj, xb, c * gg * pp);
        self.ops.push(Op::BlockPerm { dst, src: mixed, c, h, wd, gh, gw, fh, fw, swap: false, forward: false });
        Ok(())
    }

    /// ResidualSplitHeadMultiAxisGmlpLayer: the grid and block gMLPs run on the
    /// two channel halves in parallel, and the results are concatenated.
    fn split_head_gmlp(
        &mut self,
        prefix: &str,
        x: usize,
        c: usize,
        h: usize,
        wd: usize,
        block_size: usize,
        grid_size: usize,
    ) -> Result<usize, Error> {
        let ln = self.buf(c, h, wd, "shg_ln");
        self.chan_ln(ln, x, &format!("{prefix}/LayerNorm_in"), c, h * wd)?;
        let t = self.buf(2 * c, h, wd, "shg_inproject");
        self.conv1x1(t, ln, &format!("{prefix}/in_project"), c, 2 * c, h, wd)?;
        let g = self.buf(2 * c, h, wd, "shg_gelu");
        self.gelu(g, t, 2 * c * h * wd);
        // Each half writes its result back into its own channels, so the
        // concatenation is free and there is no separate cat buffer.
        let u = self.view(g, 0, c, "shg_u");
        let v = self.view(g, c, c, "shg_v");
        self.gmlp_axis(&format!("{prefix}/GridGmlpLayer"), u, u, c, h, wd, grid_size, true)?;
        self.gmlp_axis(&format!("{prefix}/BlockGmlpLayer"), v, v, c, h, wd, block_size, false)?;
        let proj = self.buf(c, h, wd, "shg_outproject");
        self.conv1x1(proj, g, &format!("{prefix}/out_project"), 2 * c, c, h, wd)?;
        let out = self.buf(c, h, wd, "shg_out");
        self.add(out, proj, x, c * h * wd);
        Ok(out)
    }

    /// GetSpatialGatingWeights: the cross-gating block's weight branch.
    ///
    /// The two gating matmuls act on the same `u`/`v` halves but at different
    /// blockings, which is the whole point of the module: `u` is mixed over the
    /// grid axis (global) and `v` over the within-block axis (local). Both stay
    /// in the blocked layout and are concatenated there, so the single
    /// `Op::GateMm` covers both with a different `mode`.
    fn spatial_gating_weights(
        &mut self,
        prefix: &str,
        x: usize,
        c: usize,
        h: usize,
        wd: usize,
        block_size: usize,
        grid_size: usize,
    ) -> Result<usize, Error> {
        let ln = self.buf(c, h, wd, "sgw_ln");
        self.chan_ln(ln, x, &format!("{prefix}/LayerNorm_in"), c, h * wd)?;
        let t = self.buf(2 * c, h, wd, "sgw_inproject");
        self.conv1x1(t, ln, &format!("{prefix}/in_project"), c, 2 * c, h, wd)?;
        let g = self.buf(2 * c, h, wd, "sgw_gelu");
        self.gelu(g, t, 2 * c * h * wd);
        let u = self.view(g, 0, c, "sgw_u");
        let v = self.view(g, c, c, "sgw_v");

        // The grid half, blocked at the grid size.
        let (gh, gw) = (grid_size, grid_size);
        let (fh, fw) = (h / gh, wd / gw);
        let (gg, pp) = (gh * gw, fh * fw);
        let ub = self.buf(c, gg, pp, "sgw_u_blocked");
        self.ops.push(Op::BlockPerm { dst: ub, src: u, c, h, wd, gh, gw, fh, fw, swap: false, forward: true });
        let um = self.buf(c, gg, pp, "sgw_u_mixed");
        let uw = self.dense(&format!("{prefix}/Dense_0/kernel"))?;
        let ubias = self.own_or_zeros(&format!("{prefix}/Dense_0/bias"), gg)?;
        self.ops.push(Op::GateMm { dst: um, src: ub, w: uw, bias: Some(ubias), c, outer: gg, inner: pp, mode: 0 });
        let uu = self.buf(c, h, wd, "sgw_u_out");
        self.ops.push(Op::BlockPerm { dst: uu, src: um, c, h, wd, gh, gw, fh, fw, swap: false, forward: false });

        // The block half, blocked at the block size.
        let (fh2, fw2) = (block_size, block_size);
        let (gh2, gw2) = (h / fh2, wd / fw2);
        let (gg2, pp2) = (gh2 * gw2, fh2 * fw2);
        let vb = self.buf(c, gg2, pp2, "sgw_v_blocked");
        self.ops.push(Op::BlockPerm { dst: vb, src: v, c, h, wd, gh: gh2, gw: gw2, fh: fh2, fw: fw2, swap: false, forward: true });
        let vm = self.buf(c, gg2, pp2, "sgw_v_mixed");
        let vw = self.dense(&format!("{prefix}/Dense_1/kernel"))?;
        let vbias = self.own_or_zeros(&format!("{prefix}/Dense_1/bias"), pp2)?;
        self.ops.push(Op::GateMm { dst: vm, src: vb, w: vw, bias: Some(vbias), c, outer: gg2, inner: pp2, mode: 1 });
        let vv = self.buf(c, h, wd, "sgw_v_out");
        self.ops.push(Op::BlockPerm { dst: vv, src: vm, c, h, wd, gh: gh2, gw: gw2, fh: fh2, fw: fw2, swap: false, forward: false });

        // Concat and project. The two halves are copied into one buffer rather
        // than aliased because the block permutation writes them separately.
        let cat = self.buf(2 * c, h, wd, "sgw_cat");
        let lo = self.view(cat, 0, c, "sgw_cat_lo");
        let hi = self.view(cat, c, c, "sgw_cat_hi");
        self.ops.push(Op::Copy { dst: lo, src: uu, n: c * h * wd });
        self.ops.push(Op::Copy { dst: hi, src: vv, n: c * h * wd });
        let out = self.buf(c, h, wd, "sgw_out");
        self.conv1x1(out, cat, &format!("{prefix}/out_project"), 2 * c, c, h, wd)?;
        Ok(out)
    }

    /// CrossGatingBlock, returning `(x, y)`.
    fn cross_gating(
        &mut self,
        prefix: &str,
        x: usize,
        cx: usize,
        y: usize,
        cy: usize,
        h: usize,
        wd: usize,
        features: usize,
        block_size: usize,
        grid_size: usize,
        upsample_y: bool,
    ) -> Result<(usize, usize), Error> {
        let y_up = if upsample_y {
            let o = self.buf(features, h, wd, "cg_y_up");
            self.convt_up(o, y, &format!("{prefix}/ConvTranspose_0"), cy, features, h / 2, wd / 2)?;
            o
        } else {
            y
        };
        let x0 = self.buf(features, h, wd, "cg_x0");
        self.conv1x1(x0, x, &format!("{prefix}/Conv_0"), cx, features, h, wd)?;
        // `Conv_1` reads `y_up`, whose channel count is `features` after the
        // transposed conv regardless of what `y` started with - the reference's
        // `num_channels` is x's post-projection width, which is `features`.
        let y0 = self.buf(features, h, wd, "cg_y0");
        self.conv1x1(y0, y_up, &format!("{prefix}/Conv_1"), features, features, h, wd)?;

        let xl = self.buf(features, h, wd, "cg_xln");
        self.chan_ln(xl, x0, &format!("{prefix}/LayerNorm_x"), features, h * wd)?;
        let xd = self.buf(features, h, wd, "cg_xdense");
        self.conv1x1(xd, xl, &format!("{prefix}/in_project_x"), features, features, h, wd)?;
        let xg = self.buf(features, h, wd, "cg_xgelu");
        self.gelu(xg, xd, features * h * wd);
        let gx = self.spatial_gating_weights(
            &format!("{prefix}/SplitHeadMultiAxisGating_x"), xg, features, h, wd, block_size, grid_size)?;

        let yl = self.buf(features, h, wd, "cg_yln");
        self.chan_ln(yl, y0, &format!("{prefix}/LayerNorm_y"), features, h * wd)?;
        let yd = self.buf(features, h, wd, "cg_ydense");
        self.conv1x1(yd, yl, &format!("{prefix}/in_project_y"), features, features, h, wd)?;
        let yg = self.buf(features, h, wd, "cg_ygelu");
        self.gelu(yg, yd, features * h * wd);
        let gy = self.spatial_gating_weights(
            &format!("{prefix}/SplitHeadMultiAxisGating_y"), yg, features, h, wd, block_size, grid_size)?;

        // y is gated by x's weights and x by y's: the crossing.
        let ym = self.buf(features, h, wd, "cg_y_mul");
        self.mul(ym, yg, gx, features * h * wd);
        let yp = self.buf(features, h, wd, "cg_y_proj");
        self.conv1x1(yp, ym, &format!("{prefix}/out_project_y"), features, features, h, wd)?;
        let yo = self.buf(features, h, wd, "cg_y_out");
        self.add(yo, yp, y0, features * h * wd);

        let xm = self.buf(features, h, wd, "cg_x_mul");
        self.mul(xm, xg, gy, features * h * wd);
        let xp = self.buf(features, h, wd, "cg_x_proj");
        self.conv1x1(xp, xm, &format!("{prefix}/out_project_x"), features, features, h, wd)?;
        let xs = self.buf(features, h, wd, "cg_x_sum");
        self.add(xs, xp, yo, features * h * wd);
        let xo = self.buf(features, h, wd, "cg_x_out");
        self.add(xo, xs, x0, features * h * wd);
        Ok((xo, yo))
    }
}


impl<'a> Builder<'a> {
    // -- the model --------------------------------------------------------

    /// The whole graph, one stage at a time. `h`/`wd` are the PADDED input
    /// dimensions, which the caller has already made a multiple of 64.
    pub fn build(mut self, h: usize, wd: usize) -> Result<Plan, Error> {
        let depth = self.cfg.depth;
        let stages = self.cfg.num_stages;
        let features = self.cfg.features;
        let nss = self.cfg.num_supervision_scales;

        let input = self.buf(3, h, wd, "input");
        self.rec("input", input);

        // The multi-scale input pyramid. jax's `nearest`, so a 2x downsample
        // keeps the odd samples: `shortcuts[i]` is NOT `shortcuts[i-1]`
        // downsampled again.
        let mut shortcuts = vec![input];
        for i in 1..nss {
            let f = 1 << i;
            let d = self.buf(3, h / f, wd / f, "shortcut");
            self.nearest_down(d, input, 3, h, wd, f);
            shortcuts.push(d);
        }

        let mut outputs_all: Vec<Vec<usize>> = Vec::new();
        let mut sam_prev: Vec<usize> = Vec::new();
        let mut encs_prev: Vec<usize> = Vec::new();
        let mut decs_prev: Vec<usize> = Vec::new();

        for s in 0..stages {
            // Input convs per scale, fused with the previous stage's SAM
            // features where the checkpoint has a fuse block (stage > 0).
            let mut x_scales = Vec::new();
            for i in 0..nss {
                let c = (1 << i) * features;
                let xs = self.buf(c, h >> i, wd >> i, "input_conv");
                self.conv3x3(xs, shortcuts[i], &format!("stage_{s}_input_conv_{i}"), 3, c, h >> i, wd >> i)?;
                if s > 0 {
                    let (bs, gs) = self.block_grid(i);
                    let sam = sam_prev.pop().ok_or("stage input fuse: no SAM feature left")?;
                    let (fused, _) = self.cross_gating(
                        &format!("stage_{s}_input_fuse_sam_{i}"), xs, c, sam, c, h >> i, wd >> i, c, bs, gs, false)?;
                    x_scales.push(fused);
                } else {
                    x_scales.push(xs);
                }
                self.rec(&format!("stage{s}_input_conv{i}"), *x_scales.last().unwrap());
            }

            // Encoder.
            let mut encs = Vec::new();
            let mut x = x_scales[0];
            for i in 0..depth {
                let (bs, gs) = self.block_grid(i);
                let features_i = (1 << i) * features;
                let (ch, cw) = (h >> i, wd >> i);
                let skip = if i < nss { Some(x_scales[i]) } else { None };
                let enc_prev = if s > 0 { Some(encs_prev.pop().ok_or("encoder: no enc feature left")?) } else { None };
                let dec_prev = if s > 0 { Some(decs_prev.pop().ok_or("encoder: no dec feature left")?) } else { None };
                // Every encoder level downsamples, the last one included: the
                // checkpoint has a `Conv_1` for i = depth-1 too, so the chain
                // ends at h / 2^depth, which is where the bottleneck runs.
                let (x_down, bridge) = self.unet_encoder_block(
                    &format!("stage_{s}_encoder_block_{i}"),
                    x, features_i, bs, gs, self.cfg.num_groups, self.cfg.channels_reduction,
                    true, skip, enc_prev, dec_prev, ch, cw,
                )?;
                self.rec(&format!("stage{s}_enc{i}_bridge"), bridge);
                self.rec(&format!("stage{s}_enc{i}_out"), x_down);
                x = x_down;
                encs.push(bridge);
            }

            // The global bottleneck.
            // `depth` downsamplings have happened (one per encoder level), so
            // the bottleneck runs at h / 2^depth at (2^(depth-1))*features.
            let global_c = (1 << (depth - 1)) * features;
            let (bh, bw) = (h >> depth, wd >> depth);
            for i in 0..self.cfg.num_bottleneck_blocks {
                x = self.bottleneck(
                    &format!("stage_{s}_global_block_{i}"), x, global_c, bh, bw,
                    self.cfg.block_size_lr, self.cfg.num_groups, self.cfg.channels_reduction)?;
            }
            self.rec(&format!("stage{s}_global"), x);
            let mut global_feature = x;

            // Cross gating: multi-scale encoder features against the global one.
            //
            // `global_feature` feeds the NEXT (finer) block as `y`, and its
            // channel count is whatever the previous level produced - the
            // coarsest level's is the bottleneck's. It is NOT a fixed
            // `2^(depth-1) * features`: after the first iteration it is that
            // level's `features_i`, and hard-coding the coarsest count made the
            // transposed conv read past its input one level down.
            let mut global_c = (1 << (depth - 1)) * features;
            let mut skip_features = Vec::new();
            for i in (0..depth).rev() {
                let (bs, gs) = self.block_grid(i);
                let features_i = (1 << i) * features;
                let (ch, cw) = (h >> i, wd >> i);
                let signal = self.multiscale_signal(&encs, i, features_i, h, wd, &format!("stage_{s}"))?;
                let (skips, gf) = self.cross_gating(
                    &format!("stage_{s}_cross_gating_block_{i}"), signal, 3 * features_i,
                    global_feature, global_c, ch, cw, features_i, bs, gs, true)?;
                global_feature = gf;
                global_c = features_i;
                self.rec(&format!("stage{s}_skip{i}"), skips);
                skip_features.push(skips);
            }

            // Decoder.
            let mut outputs = Vec::new();
            let mut decs = Vec::new();
            let mut sam_new = Vec::new();
            for i in (0..depth).rev() {
                let (bs, gs) = self.block_grid(i);
                let features_i = (1 << i) * features;
                let (ch, cw) = (h >> i, wd >> i);
                let signal = self.decoder_signal(&skip_features, i, features_i, h, wd, depth, &format!("stage_{s}"))?;
                x = self.unet_decoder_block(
                    &format!("stage_{s}_decoder_block_{i}"), x, signal, features_i, bs, gs,
                    self.cfg.num_groups, self.cfg.channels_reduction, ch, cw)?;
                self.rec(&format!("stage{s}_dec{i}"), x);
                decs.push(x);
                if i < nss {
                    if s + 1 < stages {
                        let (sam, output) = self.sam(
                            &format!("stage_{s}_supervised_attention_module_{i}"), x, shortcuts[i], features_i, ch, cw)?;
                        outputs.push(output);
                        sam_new.push(sam);
                        self.rec(&format!("stage{s}_sam{i}"), sam);
                    } else {
                        let o = self.buf(3, ch, cw, "output");
                        self.conv3x3(o, x, &format!("stage_{s}_output_conv_{i}"), features_i, 3, ch, cw)?;
                        let out = self.buf(3, ch, cw, "stage_out");
                        self.add(out, o, shortcuts[i], 3 * ch * cw);
                        outputs.push(out);
                    }
                    self.rec(&format!("stage{s}_out{i}"), *outputs.last().unwrap());
                }
            }

            encs_prev = encs;
            encs_prev.reverse();
            decs_prev = decs;
            sam_prev = sam_new;
            outputs_all.push(outputs);
        }

        let output = *outputs_all.last().and_then(|v| v.last()).ok_or("the model produced no output")?;
        self.rec("final", output);
        Ok(self.finish(input, output, BufShape { c: 3, h, w: wd }))
    }

    /// The block and grid size at a level: the high-res values below
    /// `high_res_stages`, the low-res ones at and above it. Note that the
    /// reference uses `block_size_lr` for the low-res GRID size as well, which is
    /// a quirk of the published code and not a typo here.
    fn block_grid(&self, level: usize) -> (usize, usize) {
        if level < self.cfg.high_res_stages {
            (self.cfg.block_size_hr, self.cfg.grid_size_hr)
        } else {
            (self.cfg.block_size_lr, self.cfg.grid_size_lr)
        }
    }

    /// `UpSampleRatio`: bilinear-resize a feature to level `i`'s resolution, then
    /// 1x1 it to `features_i` channels. The parameter names come from a counter
    /// the reference increments per call, in graph order.
    fn upsampled(&mut self, enc: usize, c_in: usize, from: usize, to: usize, h: usize, wd: usize, features_i: usize) -> Result<usize, Error> {
        let name = format!("UpSampleRatio_{}", self.up_index);
        self.up_index += 1;
        let r = self.buf(c_in, h >> to, wd >> to, "up_resize");
        self.resize(r, enc, c_in, h >> from, wd >> from, h >> to, wd >> to);
        let o = self.buf(features_i, h >> to, wd >> to, "up_conv");
        self.conv1x1(o, r, &format!("{name}/Conv_0"), c_in, features_i, h >> to, wd >> to)?;
        Ok(o)
    }

    /// The cross-gating signal: every encoder feature resampled to level `i` and
    /// concatenated. `encs[j]` sits at level `j`.
    fn multiscale_signal(&mut self, encs: &[usize], i: usize, features_i: usize, h: usize, wd: usize, tag: &str) -> Result<usize, Error> {
        let c = 3 * features_i;
        let (ch, cw) = (h >> i, wd >> i);
        let cat = self.buf(c, ch, cw, "cg_signal");
        for (j, &e) in encs.iter().enumerate() {
            let cj = self.bufs[e].c;
            let p = self.upsampled(e, cj, j, i, h, wd, features_i)?;
            let dst = self.view(cat, j * features_i, features_i, "cg_signal_part");
            self.ops.push(Op::Copy { dst, src: p, n: features_i * ch * cw });
        }
        let _ = tag;
        Ok(cat)
    }

    /// The decoder signal: every cross-gated skip resampled to level `i`, where
    /// `skip_features[j]` is the skip for encoder level `depth-1-j`.
    fn decoder_signal(&mut self, skips: &[usize], i: usize, features_i: usize, h: usize, wd: usize, depth: usize, tag: &str) -> Result<usize, Error> {
        let c = 3 * features_i;
        let (ch, cw) = (h >> i, wd >> i);
        let cat = self.buf(c, ch, cw, "dec_signal");
        for (j, &sk) in skips.iter().enumerate() {
            let from = depth - 1 - j;
            let cj = self.bufs[sk].c;
            let p = self.upsampled(sk, cj, from, i, h, wd, features_i)?;
            let dst = self.view(cat, j * features_i, features_i, "dec_signal_part");
            self.ops.push(Op::Copy { dst, src: p, n: features_i * ch * cw });
        }
        let _ = tag;
        Ok(cat)
    }

    /// UNetEncoderBlock, returning `(downsampled, bridge)`. `skip` is the
    /// same-level input feature to concatenate, `enc`/`dec` the previous stage's
    /// features for the cross-gating block.
    #[allow(clippy::too_many_arguments)]
    fn unet_encoder_block(
        &mut self,
        prefix: &str,
        x: usize,
        features: usize,
        block_size: usize,
        grid_size: usize,
        num_groups: usize,
        red: usize,
        downsample: bool,
        skip: Option<usize>,
        enc: Option<usize>,
        dec: Option<usize>,
        h: usize,
        wd: usize,
    ) -> Result<(usize, usize), Error> {
        let c_in = self.bufs[x].c + skip.map(|s| self.bufs[s].c).unwrap_or(0);
        let xin = if let Some(s) = skip {
            // flax concatenates along the channel axis, in the order (x, skip).
            let cat = self.buf(c_in, h, wd, "enc_skip_cat");
            let a = self.view(cat, 0, self.bufs[x].c, "enc_skip_x");
            let b = self.view(cat, self.bufs[x].c, self.bufs[s].c, "enc_skip_skip");
            self.ops.push(Op::Copy { dst: a, src: x, n: self.bufs[x].len() });
            self.ops.push(Op::Copy { dst: b, src: s, n: self.bufs[s].len() });
            cat
        } else {
            x
        };
        self.deep_rec(&format!("{prefix}_mid_cat"), xin);
        let t = self.buf(features, h, wd, "enc_conv");
        self.conv1x1(t, xin, &format!("{prefix}/Conv_0"), c_in, features, h, wd)?;
        self.deep_rec(&format!("{prefix}_mid_conv"), t);
        let mut y = t;
        for g in 0..num_groups {
            y = self.split_head_gmlp(
                &format!("{prefix}/SplitHeadMultiAxisGmlpLayer_{g}"), y, features, h, wd, block_size, grid_size)?;
            self.deep_rec(&format!("{prefix}_mid_g{g}_gmlp"), y);
            y = self.rcab(&format!("{prefix}/channel_attention_block_1{g}"), y, features, h, wd, red)?;
            self.deep_rec(&format!("{prefix}_mid_g{g}_rcab"), y);
        }
        let summed = self.buf(features, h, wd, "enc_residual");
        self.add(summed, y, t, features * h * wd);
        self.deep_rec(&format!("{prefix}_mid_sum"), summed);
        let mut out = summed;
        if let (Some(e), Some(d)) = (enc, dec) {
            let yd = self.buf(features, h, wd, "enc_encdec");
            self.add(yd, e, d, features * h * wd);
            let (o, _) = self.cross_gating(&format!("{prefix}/cross_gating_block"), summed, features, yd, features, h, wd, features, block_size, grid_size, false)?;
            out = o;
        }
        if downsample {
            let down = self.buf(features, (h + 1) / 2, (wd + 1) / 2, "enc_down");
            self.conv_down(down, out, &format!("{prefix}/Conv_1"), features, features, h, wd)?;
            Ok((down, out))
        } else {
            Ok((out, out))
        }
    }

    /// UNetDecoderBlock: up, then an encoder block without a downsample, over the
    /// concatenated decoder signal.
    #[allow(clippy::too_many_arguments)]
    fn unet_decoder_block(
        &mut self,
        prefix: &str,
        x: usize,
        bridge: usize,
        features: usize,
        block_size: usize,
        grid_size: usize,
        num_groups: usize,
        red: usize,
        h: usize,
        wd: usize,
    ) -> Result<usize, Error> {
        let up = self.buf(features, h, wd, "dec_up");
        self.convt_up(up, x, &format!("{prefix}/ConvTranspose_0"), self.bufs[x].c, features, h / 2, wd / 2)?;
        let (out, _) = self.unet_encoder_block(
            &format!("{prefix}/UNetEncoderBlock_0"), up, features, block_size, grid_size,
            num_groups, red, false, Some(bridge), None, None, h, wd)?;
        Ok(out)
    }

    /// The supervised attention module: a 3x3 branch, an image branch with the
    /// input added, and their product. Returns `(sam, image)`; `sam` is what the
    /// next stage fuses, `image` is a full-resolution prediction.
    fn sam(&mut self, prefix: &str, x: usize, x_image: usize, features: usize, h: usize, wd: usize) -> Result<(usize, usize), Error> {
        let x1 = self.buf(features, h, wd, "sam_x1");
        self.conv3x3(x1, x, &format!("{prefix}/Conv_0"), features, features, h, wd)?;
        let t = self.buf(3, h, wd, "sam_img_conv");
        self.conv3x3(t, x, &format!("{prefix}/Conv_1"), features, 3, h, wd)?;
        let image = self.buf(3, h, wd, "sam_image");
        self.add(image, t, x_image, 3 * h * wd);
        // Conv_2 has `features` outputs, not 3: this is what makes x2 the same
        // shape as x1 and the product elementwise. `reference.py` originally
        // transcribed it as 3->3 and let torch broadcast a 3-channel gate over
        // the features, which the checkpoint's Conv_2 (3,3,3,features) refutes.
        let s = self.buf(features, h, wd, "sam_img_pre_sigmoid");
        self.conv3x3(s, image, &format!("{prefix}/Conv_2"), 3, features, h, wd)?;
        let gate = self.buf(features, h, wd, "sam_sigmoid");
        self.ops.push(Op::Sigmoid { dst: gate, src: s, n: features * h * wd });
        let g = self.buf(features, h, wd, "sam_gated");
        self.mul(g, x1, gate, features * h * wd);
        let sam = self.buf(features, h, wd, "sam_out");
        self.add(sam, g, x, features * h * wd);
        Ok((sam, image))
    }

    /// BottleneckBlock: an input projection, then the gMLP + RDCAB group.
    fn bottleneck(&mut self, prefix: &str, x: usize, features: usize, h: usize, wd: usize, block_size: usize, num_groups: usize, red: usize) -> Result<usize, Error> {
        let t = self.buf(features, h, wd, "bn_inputproj");
        self.conv1x1(t, x, &format!("{prefix}/input_proj"), self.bufs[x].c, features, h, wd)?;
        let mut y = t;
        for g in 0..num_groups {
            y = self.split_head_gmlp(
                &format!("{prefix}/SplitHeadMultiAxisGmlpLayer_{g}"), y, features, h, wd, block_size, block_size)?;
            y = self.rdcab(&format!("{prefix}/channel_attention_block_1_{g}"), y, features, h, wd, red)?;
        }
        let out = self.buf(features, h, wd, "bn_out");
        self.add(out, y, t, features * h * wd);
        Ok(out)
    }

    // -- packing ----------------------------------------------------------

    /// Assign arena offsets and return the finished plan.
    ///
    /// Each buffer that owns storage gets a live range from its first to its last
    /// use in the op list, and offsets are handed out so that ranges that overlap
    /// in time do not overlap in space. Without this the arena would be one plane
    /// per op - several gigabytes for a 448x640 input - because the graph is a
    /// straight line with hundreds of same-sized activations.
    ///
    /// A `view` is not allocated at all: it resolves to a parent's storage at a
    /// channel offset, which is what makes the model's many channel splits and
    /// concatenations free.
    fn finish(self, input: usize, output: usize, shape: BufShape) -> Plan {
        let n = self.bufs.len();
        // Resolve views to their base buffer and the element delta within it.
        let mut base = vec![0usize; n];
        let mut delta = vec![0usize; n];
        for i in 0..n {
            match self.alias[i] {
                None => {
                    base[i] = i;
                    delta[i] = 0;
                }
                Some((p, d)) => {
                    base[i] = base[p];
                    delta[i] = delta[p] + d;
                }
            }
        }

        // Live ranges over the base buffers. A view's reader keeps its base
        // alive, so views are folded into the base they name.
        let mut first = vec![usize::MAX; n];
        let mut last = vec![0usize; n];
        let mut touch = Vec::new();
        for (oi, op) in self.ops.iter().enumerate() {
            touch.clear();
            op_touch(op, &mut touch);
            let mut seen = Vec::new();
            for &t in touch.iter() {
                let b = base[t];
                if seen.contains(&b) {
                    continue;
                }
                seen.push(b);
                if first[b] == usize::MAX {
                    first[b] = oi;
                }
                last[b] = oi;
            }
        }
        for i in 0..n {
            // A buffer with no use at all is dead: it is either the input, which
            // is written from outside the op list, or a builder temporary whose
            // only consumer was optimised away.
            if first[i] == usize::MAX {
                first[i] = usize::MAX - 1;
                last[i] = 0;
            }
        }

        // A NAMED buffer must survive to the end of the run: those are the
        // activations `--dump` writes, and the dump happens after every op has
        // run. Without this the packer would reuse a named buffer's storage as
        // soon as its last reader was done, and the dump would show whatever
        // tensor happened to be written there last - which looks exactly like a
        // wrong result rather than a wrong dump.
        for &(_, id) in &self.named {
            let b = base[id];
            if first[b] != usize::MAX - 1 {
                last[b] = self.ops.len().saturating_sub(1);
            }
        }

        // Offsets, in first-use order. Candidate offsets are the boundaries of
        // everything already placed; the first candidate whose block does not
        // overlap a block that is still live during [first, last] wins. This is
        // not optimal (an optimal packing is NP-hard) but it is one pass and it
        // gets the arena down to a few planes per level, which is what matters.
        let mut order: Vec<usize> = (0..n).filter(|&i| base[i] == i && first[i] != usize::MAX - 1).collect();
        order.sort_by_key(|&i| first[i]);
        struct Block {
            start: usize,
            end: usize,
            off: usize,
            size: usize,
        }
        let mut placed: Vec<Block> = Vec::new();
        let mut off = vec![0usize; n];
        let mut arena_len = 0usize;
        for &i in &order {
            let size = self.bufs[i].len();
            let (start, end) = (first[i], last[i]);
            let mut cands = vec![0usize];
            for b in &placed {
                cands.push(b.off + b.size);
            }
            cands.sort_unstable();
            cands.dedup();
            let mut chosen = None;
            for &c in &cands {
                let clashes = placed.iter().any(|b| {
                    // overlap in space AND overlap in time
                    c < b.off + b.size && b.off < c + size && start <= b.end && b.start <= end
                });
                if !clashes {
                    chosen = Some(c);
                    break;
                }
            }
            let c = chosen.unwrap_or(arena_len);
            off[i] = c;
            arena_len = arena_len.max(c + size);
            placed.push(Block { start, end, off: c, size });
        }

        // Views resolve into their parent's storage.
        let mut resolved = vec![0usize; n];
        for i in 0..n {
            resolved[i] = off[base[i]] + delta[i];
        }

        Plan {
            bufs: self.bufs,
            offs: resolved,
            ops: self.ops,
            weights: self.weights,
            arena_len,
            input,
            output,
            dumps: self.named,
            shape,
            labels: self.labels,
        }
    }
}

pub struct Plan {
    pub bufs: Vec<BufShape>,
    /// Absolute element offset of every buffer in the arena.
    pub offs: Vec<usize>,
    pub ops: Vec<Op>,
    pub weights: Vec<Vec<f32>>,
    pub arena_len: usize,
    pub input: usize,
    pub output: usize,
    /// `(name, buffer)`: what `--dump` writes and where the parity test looks.
    pub dumps: Vec<(String, usize)>,
    /// The padded input shape the plan was built for.
    pub shape: BufShape,
    /// A human-readable label per buffer, from the builder that made it. Only
    /// used by the debug trace and the error messages: the shippable engine never
    /// reads it, and a shape bug is otherwise very hard to localise.
    pub labels: Vec<String>,
}

impl Plan {
    pub fn arena_bytes(&self) -> usize {
        self.arena_len * 4
    }

    pub fn weight_bytes(&self) -> usize {
        self.weights.iter().map(|w| w.len() * 4).sum()
    }

    pub fn shape_of(&self, id: usize) -> BufShape {
        self.bufs[id]
    }

    /// The largest single allocation, the number that decides whether a device
    /// can run a given input size.
    pub fn peak_bytes(&self) -> usize {
        let mut used = vec![0i64; self.arena_len + 1];
        for (i, b) in self.bufs.iter().enumerate() {
            used[self.offs[i]] += b.len() as i64;
        }
        // the arena is packed, so the high-water mark is the arena size; what
        // matters for a device is whether the whole thing fits.
        let _ = used;
        self.arena_bytes()
    }
}
