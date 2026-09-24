//! The GPU executor: one launch (or one small group of them) per op.
//!
//! The op list is the same one the CPU executor walks, so this file is the only
//! place a backend difference can live. Two things are worth reading closely:
//!
//! * `Op::ChanLn` and `Op::ChanScale` need the LayerNorm weights or the CALayer
//!   excitation as DEVICE memory, because the toolkit kernels and `mx_channel_scale`
//!   read them as pointers. Those vectors are uploaded once into a small pool and
//!   cached by (plan weight index), so the per-op cost is a lookup.
//! * `Op::Resize` is two launches over a temporary the plan already allocated,
//!   horizontal first; the vertical pass reads a plane that is already the target
//!   width.

use lightgpu::ffi::CUdeviceptr;
use lightgpu::vm::{copy_d2d, copy_dtoh, Args, Event};

use crate::cuda::{Buf, Cuda, WeightCache};
use crate::model::{Op, Plan};
use crate::Error;

const BLOCK: u32 = 256;

/// A device copy of one of the small per-op vectors (a norm's scale/bias, a
/// CALayer's excitation), cached by its index in the plan's weight list.
struct SmallPool {
    slots: Vec<Option<(usize, Buf)>>,
    bytes: usize,
}

impl SmallPool {
    fn new() -> SmallPool {
        SmallPool { slots: Vec::new(), bytes: 0 }
    }

    fn get(&mut self, key: usize, v: &[f32]) -> Result<CUdeviceptr, Error> {
        if let Some(s) = self.slots.iter_mut().find(|s| s.as_ref().map(|(k, _)| *k) == Some(key)) {
            return Ok(s.as_ref().unwrap().1.ptr());
        }
        let mut b = Buf::new(v.len())?;
        b.upload(v)?;
        self.bytes += v.len() * 4;
        let p = b.ptr();
        self.slots.push(Some((key, b)));
        Ok(p)
    }
}

pub struct Gpu {
    cuda: Cuda,
    arena: Buf,
    pool: SmallPool,
    weights: WeightCache,
    /// `bytes` and `out` are the only host round trips in the whole graph: the
    /// input image in, and the output image out.
    launched: usize,
    /// Launches, which is not the same number: one op is usually one kernel, but
    /// `Op::Resize` is two passes and `Op::Copy` is a memcpy rather than a
    /// launch. Only meaningful with `--timeline`.
    kernels: usize,
    /// Per-op device time, sampled with CUDA events around each op. Off by
    /// default: recording an event per op is cheap but not free, and a profiled
    /// run is not the run being reported.
    timeline: Option<Timeline>,
    want_timeline: bool,
    /// Force the pre-tiling kernels, for an A/B measurement: `--legacy-ops` or
    /// `MAXIM_LEGACY_OPS=1`. Both spellings of every tiled kernel are in the
    /// binary, so this is a runtime switch rather than a rebuild.
    legacy_ops: bool,
}

/// Where the device time went, per op.
pub struct Timeline {
    pub per_op_ms: Vec<f32>,
    pub kernels_per_op: Vec<usize>,
    pub total_ms: f32,
}

impl Gpu {
    pub fn new(plan: &Plan) -> Result<Gpu, Error> {
        Ok(Gpu {
            cuda: Cuda::new()?,
            arena: Buf::new(plan.arena_len)?,
            pool: SmallPool::new(),
            weights: WeightCache::new(plan.weights.len()),
            launched: 0,
            kernels: 0,
            timeline: None,
            want_timeline: std::env::var_os("MAXIM_TIMELINE").is_some(),
            legacy_ops: std::env::var_os("MAXIM_LEGACY_OPS").is_some(),
        })
    }

    pub fn arena_bytes(&self) -> usize {
        self.arena.n * 4
    }

    pub fn weight_bytes(&self) -> usize {
        self.weights.resident_bytes()
    }

    pub fn pool_bytes(&self) -> usize {
        self.pool.bytes
    }

    pub fn ops_launched(&self) -> usize {
        self.launched
    }

    pub fn timeline(&self) -> Option<&Timeline> {
        self.timeline.as_ref()
    }

    /// Turn the per-op timeline on (or off). `MAXIM_TIMELINE` in the environment
    /// does the same thing, so `--timeline` is only the discoverable spelling.
    pub fn set_timeline(&mut self, on: bool) {
        self.want_timeline = on;
    }

    /// Use the kernels as they were before the tiling work. The tiled and the
    /// original forms of every op are both in the binary, so this is how the
    /// speedup is measured rather than asserted.
    pub fn set_legacy_ops(&mut self, on: bool) {
        self.legacy_ops = on;
    }

    /// Run the plan. The arena is zeroed first so an op that reads a buffer the
    /// plan never wrote is a visible NaN/zero rather than stale memory.
    pub fn run(&mut self, plan: &Plan, host: &[Vec<f32>], input: &[f32], output: &mut [f32]) -> Result<(), Error> {
        self.begin(plan, input)?;
        if self.want_timeline {
            self.run_timed(plan, host)?;
        } else {
            for op in &plan.ops {
                self.op(plan, host, op)?;
            }
        }
        self.read_at(plan.offs[plan.output], output)?;
        Ok(())
    }

    /// The same loop with a CUDA event before every op and one after the last,
    /// so per-op device time is measured rather than inferred from the total.
    ///
    /// Nothing here waits per op: all `n + 1` events are recorded into the queue
    /// and the loop synchronises ONCE, at the end. The alternative - resolve each
    /// op before launching the next - measures a serialised version of the queue
    /// and inflates a 8.4 s run to 17 s, which would make the profile a different
    /// program from the one being profiled. Between two events the device runs
    /// exactly one op, so the deltas are the ops' own times and they sum to the
    /// device's total.
    fn run_timed(&mut self, plan: &Plan, host: &[Vec<f32>]) -> Result<(), Error> {
        let n = plan.ops.len();
        let mut ev = Vec::with_capacity(n + 1);
        for _ in 0..=n {
            ev.push(Event::new().map_err(Error)?);
        }
        let mut kernels_per_op = Vec::with_capacity(n);
        for (i, op) in plan.ops.iter().enumerate() {
            let before = self.kernels;
            ev[i].record().map_err(Error)?;
            self.op(plan, host, op)?;
            kernels_per_op.push(self.kernels - before);
        }
        ev[n].record().map_err(Error)?;
        lightgpu::vm::sync().map_err(Error)?;
        let mut per_op_ms = Vec::with_capacity(n);
        let mut total = 0.0f32;
        for i in 0..n {
            let ms = ev[i].elapsed_ms(&ev[i + 1]).map_err(Error)?;
            total += ms;
            per_op_ms.push(ms);
        }
        self.timeline = Some(Timeline { per_op_ms, kernels_per_op, total_ms: total });
        Ok(())
    }

    /// A buffer's device pointer, as a `CUdeviceptr` argument.
    fn dp(&self, plan: &Plan, id: usize) -> CUdeviceptr {
        self.arena.ptr() + (plan.offs[id] * 4) as u64
    }

    /// Copy `v` into the arena at element `off`. The byte count comes from `v`
    /// itself rather than from a separate length: an earlier version took both
    /// and they disagreed, which silently overran the destination.
    fn write_at(&self, off: usize, v: &[f32]) -> Result<(), Error> {
        if v.is_empty() {
            return Ok(());
        }
        assert!(off + v.len() <= self.arena.n, "write of {} at {off} runs past the arena", v.len());
        let dst = self.arena.ptr() + (off * 4) as u64;
        lightgpu::vm::copy_htod(dst, bytemuck_slice(v)).map_err(Error)
    }

    fn read_at(&self, off: usize, out: &mut [f32]) -> Result<(), Error> {
        if out.is_empty() {
            return Ok(());
        }
        let src = self.arena.ptr() + (off * 4) as u64;
        copy_dtoh(bytemuck_slice_mut(out), src).map_err(Error)
    }

    /// The whole device arena, so a caller can read every named activation.
    /// Without it `--dump` on the GPU prints the stale HOST arena: correct for
    /// the output buffer, which `run` copies back, and zero everywhere else.
    pub fn read_arena(&self, out: &mut [f32]) -> Result<(), Error> {
        assert_eq!(out.len(), self.arena.n);
        self.read_at(0, out)
    }

    /// The device copy of weight `id`. The index is into the plan's weight list,
    /// so this needs no shape context - and deliberately does not take one.
    fn wp(&mut self, host: &[Vec<f32>], id: usize) -> Result<CUdeviceptr, Error> {
        Ok(self.weights.get(id, host)?.ptr())
    }

    /// Zero the arena and place the input, ready for `step_op`.
    pub fn begin(&mut self, plan: &Plan, input: &[f32]) -> Result<(), Error> {
        let zero = vec![0.0f32; plan.arena_len];
        self.arena.upload(&zero)?;
        assert_eq!(input.len(), plan.bufs[plan.input].len(), "the input image and the plan's input buffer disagree");
        self.write_at(plan.offs[plan.input], input)?;
        Ok(())
    }

    /// Execute exactly one op. Paired with `exec_cpu::step`, this is what the
    /// op-by-op verification walks.
    pub fn step_op(&mut self, plan: &Plan, host: &[Vec<f32>], i: usize) -> Result<(), Error> {
        let op = plan.ops[i].clone();
        self.op(plan, host, &op)
    }

    /// Read one buffer back from the device, for the per-op comparison.
    pub fn read_buf(&self, plan: &Plan, id: usize, out: &mut [f32]) -> Result<(), Error> {
        assert_eq!(out.len(), plan.bufs[id].len());
        self.read_at(plan.offs[id], out)
    }

    fn op(&mut self, plan: &Plan, host: &[Vec<f32>], op: &Op) -> Result<(), Error> {
        self.launched += 1;
        match *op {
            Op::Copy { dst, src, n } => {
                if n > 0 {
                    copy_d2d(self.dp(plan, dst), self.dp(plan, src), n * 4).map_err(Error)?;
                }
            }
            Op::Add { dst, a, b, n } => {
                let (d, x, y) = (self.dp(plan, dst), self.dp(plan, a), self.dp(plan, b));
                let mut args = Args::new();
                args.ptr(x).ptr(y).ptr(d).i32(n as i32);
                self.go("lg_add", Cuda::grid_for(n, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
            Op::Mul { dst, a, b, n } => {
                let (d, x, y) = (self.dp(plan, dst), self.dp(plan, a), self.dp(plan, b));
                let mut args = Args::new();
                args.ptr(x).ptr(y).ptr(d).i64(n as i64);
                self.go("mx_mul", Cuda::grid_for(n, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
            Op::GateApply { dst, u, v, n } => {
                let (d, x, y) = (self.dp(plan, dst), self.dp(plan, u), self.dp(plan, v));
                let mut args = Args::new();
                args.ptr(x).ptr(y).ptr(d).i64(n as i64);
                self.go("mx_gate_apply", Cuda::grid_for(n, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
            Op::Gelu { dst, src, n } => {
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(d).i32(n as i32);
                self.go("lg_gelu_erf", Cuda::grid_for(n, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
            Op::Lrelu { dst, src, n, slope } => {
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(d).f32(slope).i64(n as i64);
                self.go("lg_lrelu", Cuda::grid_for(n, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
            Op::Sigmoid { dst, src, n } => {
                // The toolkit has no standalone sigmoid (its transformer engines
                // fold it into an attention kernel), so the engine ships one:
                // keeping sigmoid a separate op is what makes the CPU and GPU op
                // lists identical.
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(d).i64(n as i64);
                self.go("mx_sigmoid", Cuda::grid_for(n, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
            Op::ChanLn { dst, src, scale, bias, c, hw } => {
                let sp = self.pool.get(scale, &host[scale])?;
                let bp = self.pool.get(bias, &host[bias])?;
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(sp).ptr(bp).ptr(d).i32(c as i32).i32(hw as i32).f32(crate::model::EPS);
                self.go("lg_channel_layer_norm", (hw.max(1) as u32, 1, 1), (BLOCK, 1, 1), plan, &mut args)?;
            }
            Op::ChanMean { dst, src, c, hw } => {
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(d).i32(c as i32).i32(hw as i32);
                self.go("mx_channel_mean", Cuda::grid_for(c, 128), (128, 1, 1), plan, &mut args)?;
            }
            Op::ChanScale { dst, src, s: sid, c, hw } => {
                // `sid` is an activation buffer (the sigmoid output), so it is
                // already on the device and needs no upload.
                let sp = self.dp(plan, sid);
                let (d, x) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(x).ptr(sp).ptr(d).i32(c as i32).i32(hw as i32);
                self.go("mx_channel_scale", Cuda::grid_for(c * hw, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
            Op::Conv1x1 { dst, src, w, wt, bias, c_in, c_out, h, wd } => {
                // `mx_conv1x1_t`, not the toolkit's `lg_conv1x1`: this op family
                // holds a third of the run and the toolkit's one-output-per-thread
                // form re-reads the input column per output. The tiled kernel
                // needs the transposed weight, which the plan carries as `wt` -
                // the two blobs are the same numbers in the two layouts the two
                // kernels want.
                let bp = match bias {
                    Some(b) => self.wp(host, b)?,
                    None => 0,
                };
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                if self.legacy_ops {
                    let wp = self.wp(host, w)?;
                    let mut args = Args::new();
                    args.ptr(s).ptr(wp).ptr(bp).ptr(d)
                        .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(wd as i32);
                    self.go(
                        "lg_conv1x1",
                        Cuda::grid_for(c_out * h * wd, BLOCK),
                        (BLOCK, 1, 1),
                        plan,
                        &mut args,
                    )?;
                } else {
                    let wt = wt.expect("the plan was built for the CPU: use --device cpu");
                    let wp = self.wp(host, wt)?;
                    let mut args = Args::new();
                    args.ptr(s).ptr(wp).ptr(bp).ptr(d)
                        .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(wd as i32);
                    let plane = (h * wd).max(1) as u32;
                    let octiles = c_out.div_ceil(4).max(1) as u32;
                    self.go(
                        "mx_conv1x1_t",
                        (Cuda::grid_for(plane as usize, BLOCK).0, octiles, 1),
                        (BLOCK, 1, 1),
                        plan,
                        &mut args,
                    )?;
                }
            }
            Op::Conv3x3 { dst, src, w, bias, c_in, c_out, h, wd } => {
                let wp = self.wp(host, w)?;
                let bp = match bias {
                    Some(b) => self.wp(host, b)?,
                    None => 0,
                };
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(wp).ptr(bp).ptr(d).i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(wd as i32);
                // The tiled form wants a 64-pixel-wide row segment; every feature
                // map this engine builds satisfies that (the plan pads to a
                // multiple of 64 and each level halves), but the guard means a
                // caller with an odd width gets the toolkit kernel rather than a
                // silently wrong answer. It also picks the narrow variant for
                // c_out < 64, which keeps a 32-channel layer from running with
                // half the block idle.
                let tiles = if self.legacy_ops || wd % 64 != 0 {
                    None
                } else if c_out >= 64 {
                    Some(("mx_conv3x3_t4", 16u32, 16u32, 64usize))
                } else {
                    Some(("mx_conv3x3_t2", 32, 8, 32usize))
                };
                match tiles {
                    Some((name, bx, by, loc)) => {
                        let grid = (
                            c_out.div_ceil(loc).max(1) as u32,
                            (h * (wd / 64)).max(1) as u32,
                            1,
                        );
                        self.go(name, grid, (bx, by, 1), plan, &mut args)?;
                    }
                    None => {
                        self.go(
                            "lg_conv3x3s1p1",
                            Cuda::grid_for(c_out * h * wd, BLOCK),
                            (BLOCK, 1, 1),
                            plan,
                            &mut args,
                        )?;
                    }
                }
            }
            Op::Conv4x4s2 { dst, src, w, bias, c_in, c_out, h, wd, pad_top, pad_left, oh, ow } => {
                let wp = self.wp(host, w)?;
                let bp = match bias {
                    Some(b) => self.wp(host, b)?,
                    None => 0,
                };
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(wp).ptr(bp).ptr(d)
                    .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(wd as i32)
                    .i32(oh as i32).i32(ow as i32).i32(pad_top as i32).i32(pad_left as i32);
                self.go("mx_conv4x4s2", Cuda::grid_for(c_out * oh * ow, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
            Op::ConvT2x2 { dst, src, w, bias, c_in, c_out, h, wd } => {
                let wp = self.wp(host, w)?;
                let bp = match bias {
                    Some(b) => self.wp(host, b)?,
                    None => 0,
                };
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(wp).ptr(bp).ptr(d).i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(wd as i32);
                self.go("mx_convt2x2s2", Cuda::grid_for(h * wd, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
            Op::GateMm { dst, src, w, bias, c, outer, inner, mode } => {
                let wp = self.wp(host, w)?;
                let bp = match bias {
                    Some(b) => self.wp(host, b)?,
                    None => 0,
                };
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(wp).ptr(bp).ptr(d)
                    .i32(c as i32).i32(outer as i32).i32(inner as i32);
                // The tiled implicit GEMM, one entry point per mode because the
                // mode decides which axis of the activation is the contiguous one
                // and that has to be a compile-time choice inside the kernel.
                // The alternative - `mx_gate_mm`, one thread per output element -
                // is still linked and still what `--legacy-ops` uses.
                if self.legacy_ops {
                    let mut args = Args::new();
                    args.ptr(s).ptr(wp).ptr(bp).ptr(d)
                        .i32(c as i32).i32(outer as i32).i32(inner as i32).i32(mode as i32);
                    self.go(
                        "mx_gate_mm",
                        Cuda::grid_for(c * outer * inner, BLOCK),
                        (BLOCK, 1, 1),
                        plan,
                        &mut args,
                    )?;
                } else {
                    // A is the axis the Dense replaces, K the axis it reduces.
                    // They are equal for this model's gating Dense, but the grid
                    // is built from A and the kernel is told both.
                    let a = if mode == 0 { outer } else { inner };
                    let b = if mode == 0 { c * inner } else { c * outer };
                    let grid = (
                        a.div_ceil(64).max(1) as u32,
                        b.div_ceil(64).max(1) as u32,
                        1,
                    );
                    let name = if mode == 0 { "mx_gate_mm_t0" } else { "mx_gate_mm_t1" };
                    self.go(name, grid, (16, 16, 1), plan, &mut args)?;
                }
            }
            Op::BlockPerm { dst, src, c, h, wd, gh, gw, fh, fw, swap, forward } => {
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(d)
                    .i32(c as i32).i32(h as i32).i32(wd as i32)
                    .i32(gh as i32).i32(gw as i32).i32(fh as i32).i32(fw as i32)
                    .i32(swap as i32).i32(forward as i32);
                self.go("mx_block_perm", Cuda::grid_for(c * h * wd, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
            Op::Resize { dst, src, tmp, c, hin, win, hout, wout } => {
                let (d, s, t) = (self.dp(plan, dst), self.dp(plan, src), self.dp(plan, tmp));
                // The horizontal pass writes the temp, whose plane is `hin*wout`
                // - NOT `hout*wout`. Each pass states its own plane sizes.
                let mut args = Args::new();
                args.ptr(s).ptr(t)
                    .i32(c as i32).i32(hin as i32).i32(win as i32).i32(hout as i32).i32(wout as i32)
                    .i32((hin * win) as i32).i32((hin * wout) as i32).i32(2);
                self.go("mx_resize_axis", Cuda::grid_for(c * hin, 128), (128, 1, 1), plan, &mut args)?;
                let mut args = Args::new();
                args.ptr(t).ptr(d)
                    .i32(c as i32).i32(hin as i32).i32(wout as i32).i32(hout as i32).i32(wout as i32)
                    .i32((hin * wout) as i32).i32((hout * wout) as i32).i32(1);
                self.go("mx_resize_axis", Cuda::grid_for(c * wout, 128), (128, 1, 1), plan, &mut args)?;
            }
            Op::DownS { dst, src, c, h, wd, stride, off, oh, ow } => {
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(d)
                    .i32(c as i32).i32(h as i32).i32(wd as i32)
                    .i32(stride as i32).i32(off as i32).i32(oh as i32).i32(ow as i32);
                self.go("mx_down2", Cuda::grid_for(c * oh * ow, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
        }
        Ok(())
    }

    fn go(&mut self, kernel: &str, grid: (u32, u32, u32), block: (u32, u32, u32), _plan: &Plan, args: &mut Args) -> Result<(), Error> {
        self.kernels += 1;
        self.cuda.launch(kernel, grid, block, args)
    }
}

fn bytemuck_slice(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytemuck_slice_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
}
