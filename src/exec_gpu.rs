//! The GPU executor: one launch (or one small group of them) per op.
//!
//! The op list is the same one the CPU executor walks, so this file is the only
//! place a backend difference can live. Two things are worth reading closely:
//!
//! * `Op::ChanLn` and `Op::ChanScale` need the LayerNorm weights or the CALayer
//!   excitation as DEVICE memory, because the toolkit kernels `lg_channel_layer_norm`
//!   and `lg_channel_scale` read them as pointers. Those vectors are uploaded once into a small pool and
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
}

/// Where the device time went, per op.
pub struct Timeline {
    pub per_op_ms: Vec<f32>,
    pub kernels_per_op: Vec<usize>,
    pub total_ms: f32,
}

/// Bring the CUDA driver up.
///
/// THIS IS THE ONLY THING THAT MAY SEND A RUN TO THE CPU BACKEND. A machine with
/// no NVIDIA driver must still work - that is what one binary for both backends
/// buys - and a driver that will not come up is not an error unless the caller
/// named the GPU. A pass that fails for any other reason is REPORTED, never
/// retried on the CPU; see `Gpu::new` below and the run in `main`.
pub fn driver_ready() -> Result<(), Error> {
    lightgpu::vm::init().map_err(Error)
}

impl Gpu {
    /// The device arena, the weight cache and the small pool.
    ///
    /// THE PASS IS SIZED BEFORE IT IS ALLOCATED, AND REFUSED IF IT WILL NOT FIT.
    /// `plan.arena_len` is the packed arena the plan's own liveness pass
    /// computed, and the arena is a single allocation this engine makes itself,
    /// so the number printed below is the number the driver is about to be asked
    /// for - exact, not a model of what the graph might use. That is what makes
    /// a refusal honest rather than a guess, and refusing is what happens here:
    /// the alternative is `cuMemAlloc failed` half a second into a launch, with
    /// nothing said about what would have fitted.
    pub fn new(plan: &Plan) -> Result<Gpu, Error> {
        let cuda = Cuda::new()?;
        let arena_bytes = plan.arena_bytes();
        let weight_bytes = plan.weight_bytes();
        // A free-VRAM query that fails is NOT a reason to refuse: the driver is
        // already up (the line above brought it up), so this is a courtesy that
        // is occasionally unavailable, and running an unguarded pass beats
        // refusing every pass on a machine where one ioctl moved.
        let free = lightgpu::vm::free_vram().unwrap_or(usize::MAX);
        eprintln!(
            "maxim: arena {} MiB ({} buffers), weights up to {} MiB, {} MiB free",
            arena_bytes / 1048576,
            plan.bufs.len(),
            weight_bytes / 1048576,
            free / 1048576
        );
        if arena_bytes + weight_bytes > free {
            return Err(format!(
                "not enough device memory for a {}x{} pass\n\
                 maxim: the arena needs {} MiB and the weights up to {} MiB; {} MiB is free\n\
                 maxim: the arena is exact - it is the plan's own packed size, not an estimate - so this is a hard limit, not a guess\n\
                 maxim: a smaller image, a lighter checkpoint, or a freer card is what fits",
                plan.shape.w,
                plan.shape.h,
                arena_bytes / 1048576,
                weight_bytes / 1048576,
                free / 1048576
            )
            .into());
        }
        let arena = Buf::new(plan.arena_len).map_err(|e: Error| {
            // THE CHECK ABOVE CAN PASS AND THE ALLOCATION STILL FAIL. Free VRAM
            // is a total, and the driver has to find one CONTIGUOUS run for the
            // arena, so a failure here is reported with the same numbers plus
            // the one fact that distinguishes it - rather than as a bare
            // `cuMemAlloc failed`, which says nothing about the size that was
            // asked for.
            if e.0.contains("OUT_OF_MEMORY") {
                Error(format!(
                    "the arena for {}x{} ({} MiB) did not fit after all, with {} MiB free\n\
                     maxim: {e}\n\
                     maxim: free VRAM is a total, and the arena needs one contiguous run, so a fragmented card can refuse a plan the total says fits",
                    plan.shape.w,
                    plan.shape.h,
                    arena_bytes / 1048576,
                    free / 1048576
                ))
            } else {
                e
            }
        })?;
        Ok(Gpu {
            cuda,
            arena,
            pool: SmallPool::new(),
            weights: WeightCache::new(plan.weights.len()),
            launched: 0,
            kernels: 0,
            timeline: None,
            want_timeline: std::env::var_os("MAXIM_TIMELINE").is_some(),
        })
    }

    /// The GPU's reported model name, for the progress line.
    ///
    /// The toolkit owns the driver, so the name comes from its device query
    /// rather than from a second CUDA call here; `?` becomes "?" because a name
    /// is a courtesy and a run that has already allocated its arena should not
    /// fail over a label.
    pub fn device_name(&self) -> String {
        lightgpu::vm::device().map(|d| d.name).unwrap_or_else(|_| "?".into())
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

    /// Run the plan.
    pub fn run(&mut self, plan: &Plan, host: &[Vec<f32>], input: &[f32], output: &mut [f32]) -> Result<(), Error> {
        // A phase split, because the wall time is what a user sees and the device
        // timeline is what the kernels cost - and the gap between them is the thing
        // worth knowing before optimizing either. `MAXIM_TIME=1` prints it; off by
        // default so a normal run is not perturbed by the clock reads.
        let phases = std::env::var_os("MAXIM_TIME").is_some();
        let t0 = std::time::Instant::now();
        self.begin(plan, input)?;
        let t1 = std::time::Instant::now();
        if self.want_timeline {
            self.run_timed(plan, host)?;
        } else {
            for op in &plan.ops {
                self.op(plan, host, op)?;
            }
        }
        let t2 = std::time::Instant::now();
        self.read_at(plan.offs[plan.output], output)?;
        let t3 = std::time::Instant::now();
        if phases {
            let ms = |a: std::time::Instant, b: std::time::Instant| {
                b.duration_since(a).as_secs_f64() * 1000.0
            };
            eprintln!(
                "time: begin (arena + {} MiB of resident weights) {:.1} ms | op loop {:.1} ms | read-back {:.1} ms | total {:.1} ms",
                self.weights.resident_bytes() / (1024 * 1024),
                ms(t0, t1),
                ms(t1, t2),
                ms(t2, t3),
                ms(t0, t3)
            );
        }
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

    /// One device buffer, copied into `out` - the narrow read-back a run that only
    /// needs the output image should do, rather than pulling the whole arena over
    /// the link.
    pub fn read_into(&self, off: usize, out: &mut [f32]) -> Result<(), Error> {
        self.read_at(off, out)
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

    /// Place the input image, ready for `step_op`.
    ///
    /// The arena is NOT zeroed, and that is a deliberate, measured decision. It
    /// used to be: every run uploaded 942.7 MiB of zeros at the eval size, which
    /// at a 600x400 input is 0.34 s of a 3.31 s wall time. That was pure waste.
    /// The claim "something might read a buffer element before writing it, so
    /// start from zeros" is testable, and it is false for this graph: the
    /// `MAXIM_POISON=1` mode below fills the whole arena with 0xCD - a signalling
    /// NaN as an f32 - and skips the zeroing entirely, and the output is
    /// BYTE-IDENTICAL to the zeroed run at 96x96, 256x256 and 600x400, and with
    /// `--factor 32` (a different padding, so a different fallback-kernel mix).
    /// `tests/packing.rs` runs that comparison.
    ///
    /// Keep the poison mode: it is the evidence, and it is the only way to tell
    /// "this buffer is written before it is read" from "this buffer was zeroed,
    /// so the stale bytes never showed".
    pub fn begin(&mut self, plan: &Plan, input: &[f32]) -> Result<(), Error> {
        if std::env::var_os("MAXIM_POISON").is_some() {
            // The fill goes through the driver directly rather than through a
            // toolkit wrapper: the toolkit has `cuMemsetD8` in `ffi` but exposes
            // no general fill, and this was the only thing that needed one. That
            // keeps the poison mode - the evidence, per the doc above - from
            // making `lightgpu::vm` a file two projects have to edit at once.
            let bytes = self.arena.n * 4;
            let d = lightgpu::ffi::driver().map_err(Error)?;
            lightgpu::ffi::chk(d.cuMemsetD8(self.arena.ptr(), 0xCD, bytes.max(1)), "cuMemsetD8")
                .map_err(Error)?;
        }
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
                // `int`, not `long`: `lg_mul` is the toolkit's `lg_add`-shaped op
                // (see its doc: a single plane, far below 2^31), whereas the
                // `mx_mul` this replaced took a `long`. Reading the length from
                // the wrong slot is silence, not an error, so the width matters.
                args.ptr(x).ptr(y).ptr(d).i32(n as i32);
                self.go("lg_mul", Cuda::grid_for(n, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
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
                // Keeping sigmoid a separate op is what makes the CPU and GPU op
                // lists identical, and the toolkit has one - an earlier comment
                // here claimed it did not because its transformer engines fold it
                // into an attention kernel, which was wrong. `lg_sigmoid` is
                // `1/(1+__expf(-x))`; the `mx_sigmoid` this replaces used `expf`,
                // so this is the one promoted op whose arithmetic changed.
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(d).i32(n as i32);
                self.go("lg_sigmoid", Cuda::grid_for(n, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
            Op::ChanLn { dst, src, scale, bias, c, hw } => {
                let sp = self.pool.get(scale, &host[scale])?;
                let bp = self.pool.get(bias, &host[bias])?;
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                // One thread per spatial position. When this was written the
                // toolkit's `lg_channel_layer_norm` launched one BLOCK per
                // position with a shared-memory halving tree, so this kernel was
                // both faster (4.697 -> 0.468 ms at the eval size) and different
                // in the last bits. The toolkit has since adopted the same
                // per-position shape and the same serial ascending sum, so the two
                // now AGREE on the summation order: the reason to keep a local
                // kernel is that it needs no shared memory and no block geometry
                // at all, not that it disagrees with the toolkit.
                //
                // `c` chooses a COMPILE-TIME instantiation, because that is what
                // lets the pixel's channels live in registers and stops the op
                // reading x twice: see the kernel's own comment (1.46-1.72x, and
                // the shared-memory version of the same idea measured as a
                // loser). The plan only builds c = 32, 64 and 128; `mx_chan_ln`
                // is the runtime-c fallback and is not reached by this model.
                let ck = match c {
                    32 => "mx_chan_ln_c32",
                    64 => "mx_chan_ln_c64",
                    128 => "mx_chan_ln_c128",
                    _ => "mx_chan_ln",
                };
                let mut args = Args::new();
                args.ptr(s).ptr(sp).ptr(bp).ptr(d).i32(c as i32).i32(hw as i32).f32(crate::model::EPS);
                self.go(ck, Cuda::grid_for(hw.max(1), BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
            Op::ChanMean { dst, src, c, hw } => {
                // `lg_channel_mean` is the same op as the deleted
                // `mx_channel_mean` (one block per channel, grid = (c, 1, 1)) but
                // reduces with the toolkit's static 1024-slot scratch instead of
                // one thread per channel with a serial loop, so it takes no
                // `.shared(...)` and it wants a block as wide as the plane
                // allows rather than 128-at-most. The toolkit's summation order
                // is fixed and its CPU twin reproduces it; the result differs
                // from the old kernel only in the last bits.
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(d).i32(c as i32).i32(hw as i32);
                let block = if hw >= 1024 { 1024 } else { 128 };
                self.go("lg_channel_mean", (c.max(1) as u32, 1, 1), (block, 1, 1), plan, &mut args)?;
            }
            Op::ChanScale { dst, src, s: sid, c, hw } => {
                // `sid` is an activation buffer (the sigmoid output), so it is
                // already on the device and needs no upload. `lg_channel_scale`
                // is `mx_channel_scale` with the same `(in, s, out, c, hw)`
                // argument list; in the toolkit it is the `shift = null` case of
                // `lg_channel_affine`.
                let sp = self.dp(plan, sid);
                let (d, x) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(x).ptr(sp).ptr(d).i32(c as i32).i32(hw as i32);
                self.go("lg_channel_scale", Cuda::grid_for(c * hw, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
            // `w` is the CPU twin's biplane; the GPU reads only `wt`.
            Op::Conv1x1 { dst, src, w: _, wt, bias, c_in, c_out, h, wd } => {
                // The 1x1 family holds about a third of the run, and the plan
                // carries the transposed weight this family's kernels need as
                // `wt`.
                let bp = match bias {
                    Some(b) => self.wp(host, b)?,
                    None => 0,
                };
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let wt = wt.expect("the plan was built for the CPU: use --device cpu");
                let wp = self.wp(host, wt)?;
                let mut args = Args::new();
                args.ptr(s).ptr(wp).ptr(bp).ptr(d)
                    .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(wd as i32);
                let plane = (h * wd).max(1) as u32;
                // The kernel's block is 32 pixels x 8 rows of threads, and each
                // thread owns 4 pixels: 128 pixels per block, and 32 or 64 output
                // channels. `BLOCK` (256) is the thread count the other launches
                // use, so the geometry is derived here rather than borrowed from it.
                //
                // WHICH ONE, and it is a `c_out` decision the same way conv3x3's
                // t2/t4 split is: `mx_conv1x1_t8` (64 channels per block) is the
                // fastest kernel in the file above 64 output channels - 2.1-2.4x
                // the scalar form it replaced - while at c_out <= 32 a 64-channel
                // block spreads the same grid over channels that do not exist, and
                // that waste is measurable: 64->32 @448x640 0.5554 against 0.4702 ms,
                // 32->32 @448x640 0.3486 against 0.3232, so `_t4` keeps those two.
                //
                // Both float4 forms require each thread's four pixels to be
                // 16-byte aligned, i.e. `plane % 4 == 0`. A misaligned float4 load
                // does NOT fault, it returns the wrong four floats, so this is a
                // silent-wrongness precondition and it is checked here rather than
                // assumed; the fallback is the scalar kernel, which is always
                // correct. Every shape the plan builds has `plane % 4 == 0`
                // (1120, 4480, 17920, 71680, 286720), so the fallback is unreached.
                let (kernel, oct) = match plane % 4 {
                    0 if c_out > 32 => ("mx_conv1x1_t8", c_out.div_ceil(64)),
                    0 => ("mx_conv1x1_t4", c_out.div_ceil(32)),
                    _ => ("mx_conv1x1_t", c_out.div_ceil(32)),
                };
                let pblocks = plane.div_ceil(128).max(1);
                self.go(kernel, (pblocks, oct.max(1) as u32, 1), (32, 8, 1), plan, &mut args)?;
            }
            // `w` is the CPU twin's copy; the GPU reads only `wt`, the
            // `[tap][ci][oc]` layout (see `weights::conv3x3_t`).
            Op::Conv3x3 { dst, src, w: _, wt, bias, c_in, c_out, h, wd } => {
                let wt = wt.expect("the plan was built for the CPU: use --device cpu");
                let wp = self.wp(host, wt)?;
                let bp = match bias {
                    Some(b) => self.wp(host, b)?,
                    None => 0,
                };
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(wp).ptr(bp).ptr(d).i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(wd as i32);
                // The kernel reads the weight as `[tap][ci][oc]`, which is what makes
                // four output channels one 128-bit shared load instead of four 32-bit
                // ones: measured head to head with the old `[oc][ci][tap]` layout at
                // identical output, 32->32 @448x640 3.406 -> 2.174 ms, 64->64 @224x320
                // 2.245 -> 1.708, 128->128 @112x160 2.616 -> 2.026.
                //
                // The tiled form works on a 64-pixel-wide SEGMENT OF ONE ROW, and
                // the last segment may be short: the grid is
                // (output-channel blocks, rows, segments per row) and each block is
                // told how many columns it owns. A width that is not a multiple of
                // 64 therefore costs one narrow segment per row rather than
                // dropping the whole row to the untiled kernel - which is what it
                // used to do, and which the 600x400 profile showed to be the
                // single largest cost in the run: 17 conv3x3 128->128 @112x160 at
                // ~46 ms each, 785 ms of 3.1 s, because every level below the top
                // has such a width (640 halves to 160 and then 80).
                //
                // The narrow variant is for c_out < 64, so a 32-channel layer does
                // not run with half the block idle.
                //
                // The t2 form owns a 64-column segment; `mx_conv3x3_w4` is the
                // SAME body with 128 columns per block (32x8 threads, 4 pixels
                // each), which halves the number of times the weight tile is
                // restaged - the larger half of this kernel's staging cost. It is
                // worth 1.15-1.25x wherever it does not add padded work and LOSES
                // where it does (the full plane sweep is in the kernel's comment:
                // 0.92x at wd 192 and 160, 0.64x at 64 and 40). Both forms pad
                // the same way at `wd % 128 == 0`, so that is the gate, and it is
                // the only place the two are comparable: 640 is the widest conv3x3
                // in the model and the largest single one at 36.9 ms.
                let wide = c_out < 64 && wd % 128 == 0;
                let segs = wd.div_ceil(if wide { 128 } else { 64 }).max(1);
                let (name, bx, by, loc) = if c_out >= 64 {
                    ("mx_conv3x3_t4", 16u32, 16u32, 64usize)
                } else if wide {
                    ("mx_conv3x3_w4", 32, 8, 32usize)
                } else {
                    ("mx_conv3x3_t2", 32, 8, 32usize)
                };
                // `segs` is passed rather than left for the kernel to recover
                // from `gridDim.y / h`: the segments are folded into `y` where
                // that fits, because a segment on an axis of its own measures 11%
                // slower (see the kernel's comment), and at 2048x2048 the folded
                // count is one block past what a grid dimension allows - so the
                // two layouts coexist and only this side knows which one ran.
                args.i32(0).i32(segs as i32);
                let (gy, gz) = slow_grid(h.max(1) * segs);
                let grid = (c_out.div_ceil(loc).max(1) as u32, gy, gz);
                self.go(name, grid, (bx, by, 1), plan, &mut args)?;
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
                // A register tile, not a shared-memory one: `mx_conv4x4s2` in
                // cuda/maxim.cu keeps 4 output columns x 4 output channels per
                // thread (CD_PX x CD_CO) and reads every operand from global.
                // Measured against the two forms it replaced, all bit-identical:
                // the smem-staged form 1.285x slower at 32->32 @448x640 (3.863
                // vs 3.005 ms), 1.831x at 64->64, 2.532x at 128->128, and the
                // original untiled kernel 10x behind that. Its comment records
                // the full sweep, the losers, and the one deviation (the channel
                // sum is regrouped into CD_CI passes, unchanged by this landing).
                //
                // IT HAS ITS OWN LAUNCH GEOMETRY, and it differs per shape. The
                // rule is as many output columns per block as the register budget
                // holds: blockDim.x * P = 64 columns at 32 channels, 32 columns
                // at 64 and 128, always 4 rows. A block sized by habit instead
                // measures 0.81x at (4,4) and 1.15x at (32,4) for the same P and
                // C, and launching this kernel with a flat grid and a 1-D block
                // leaves threadIdx.y at zero, which writes a fraction of the
                // output and cost 20 dB PSNR once - the standalone harness with
                // its own tiled grid is what localised that to the launch.
                // bx is the number of THREADS in x, and each thread owns
                // CD_PX = 4 output columns, so bx * 4 is the block's column
                // span: 64 columns (bx 16) at 32 output channels, 32 columns
                // (bx 8) at 64 and 128. Writing bx as the column count instead
                // measures 4.40 ms/op against the deleted kernel's 3.86.
                let bx: u32 = if c_out <= 32 { 16 } else { 8 };
                let by: u32 = 4;
                let grid = (
                    ow.div_ceil((bx * 4) as usize).max(1) as u32,
                    oh.div_ceil(by as usize).max(1) as u32,
                    c_out.div_ceil(4).max(1) as u32,
                );
                self.go("mx_conv4x4s2", grid, (bx, by, 1), plan, &mut args)?;
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
                // A register tile of 2 adjacent INPUT positions x 8 output
                // channels per thread (CD_P/CD_C in cuda/maxim.cu), one thread
                // row per input row on blockIdx.z. Measured against the
                // one-input-position-per-thread form it replaced, bit-identical:
                // 1.640x at 32->32 @448x640, 1.755x at 64->64, 1.374x at
                // 128->128. The previous form issued 1.03 loads per FMA against
                // this SM's 4 FMA : 1 load issue ratio, i.e. a load pipe
                // oversubscribed 4x; hoisting the weights across the P positions
                // takes it to 0.31 and the family from 787 to 1153-1289 GFLOP/s.
                //
                // IT HAS ITS OWN LAUNCH GEOMETRY, per shape: a warp must cover
                // several 2P-wide windows of one row for the loads to have
                // anything to hide behind, so block_x is 64 at 32 output
                // channels, 80 at 64 and 16 at 128, with the channel tile on
                // blockIdx.y and the input row on blockIdx.z. block_x = 8 at
                // 32 channels measures 0.590x for the same P and C.
                let bx: u32 = if c_out <= 32 { 64 } else if c_out <= 64 { 80 } else { 16 };
                let gy = (c_out as u32).div_ceil(8).max(1);
                self.go(
                    "mx_convt2x2s2",
                    (wd.div_ceil((bx * 2) as usize).max(1) as u32, gy, h as u32),
                    (bx, 1, 1),
                    plan,
                    &mut args,
                )?;
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
                // A is the axis the Dense replaces, K the axis it reduces. They
                // are equal for this model's gating Dense, but the grid is built
                // from A and the kernel is told both.
                let a = if mode == 0 { outer } else { inner };
                let b = if mode == 0 { c * inner } else { c * outer };
                let grid = (a.div_ceil(64).max(1) as u32, b.div_ceil(64).max(1) as u32, 1);
                let name = if mode == 0 { "mx_gate_mm_t0" } else { "mx_gate_mm_t1" };
                self.go(name, grid, (16, 16, 1), plan, &mut args)?;
            }
            Op::BlockPerm { dst, src, c, h, wd, gh, gw, fh, fw, swap, forward } => {
                let (d, s) = (self.dp(plan, dst), self.dp(plan, src));
                let mut args = Args::new();
                args.ptr(s).ptr(d)
                    .i32(c as i32).i32(h as i32).i32(wd as i32)
                    .i32(gh as i32).i32(gw as i32).i32(fh as i32).i32(fw as i32)
                    .i32(swap as i32).i32(forward as i32);
                // ONE THREAD PER ELEMENT OF THE BLOCKED TENSOR, covering all `c`
                // channels in a loop - not one per (c, g, p) element as this used to
                // launch. The blocked index is the output's contiguous index, so the
                // store is linear and the derivation is two divisions instead of
                // eight, and the channel loop walks the planes instead of asking a
                // fresh thread to find each one. Measured 2.0-2.3x at all 14 shapes
                // this model builds, output identical (see the kernel's own comment).
                let blk = gh * gw * fh * fw;
                self.go("mx_block_perm", Cuda::grid_for(blk, BLOCK), (BLOCK, 1, 1), plan, &mut args)?;
            }
            Op::Resize { dst, src, tmp, c, hin, win, hout, wout } => {
                let (d, s, t) = (self.dp(plan, dst), self.dp(plan, src), self.dp(plan, tmp));
                // The horizontal pass writes the temp, whose plane is `hin*wout`
                // - NOT `hout*wout`. Each pass states its own plane sizes.
                // One thread per output ELEMENT of each pass, so the grid is
                // c*rows*n_out: c*hin*wout for the horizontal pass and
                // c*hout*wout for the vertical one. It used to be c*rows with one
                // thread walking a whole row, which at 448x640 -> 224x320 (kscale
                // 2) cost 13.676 ms against 0.916 ms for this shape - bit-identical
                // output, and the tap weights are now computed once per output
                // element instead of once per row.
                // The kernel picks its own fast axis per direction; the launch has
                // to match it. Horizontal (axis 2): the fast axis is the output
                // COLUMN, so grid.x spans `wout` and grid.y spans (c * hin).
                // Vertical (axis 1): the fast axis is the COLUMN of the temp (wout
                // wide) and the slow one is the output ROW, so grid.x spans `wout`
                // and grid.y spans (c * hout). Block 64x4 in both, which measured
                // best on the heaviest pass and is flat elsewhere.
                const RB: u32 = 64;
                const RC: u32 = 4;
                let mut args = Args::new();
                args.ptr(s).ptr(t)
                    .i32(c as i32).i32(hin as i32).i32(win as i32).i32(hout as i32).i32(wout as i32)
                    .i32((hin * win) as i32).i32((hin * wout) as i32).i32(2);
                let (gy, gz) = slow_grid((c * hin).div_ceil(RC as usize));
                let g = ((wout as u32).div_ceil(RB), gy, gz);
                self.go("mx_resize_axis", g, (RB, RC, 1), plan, &mut args)?;
                let mut args = Args::new();
                args.ptr(t).ptr(d)
                    .i32(c as i32).i32(hin as i32).i32(wout as i32).i32(hout as i32).i32(wout as i32)
                    .i32((hin * wout) as i32).i32((hout * wout) as i32).i32(1);
                let (gy, gz) = slow_grid((c * hout).div_ceil(RC as usize));
                let g = ((wout as u32).div_ceil(RB), gy, gz);
                self.go("mx_resize_axis", g, (RB, RC, 1), plan, &mut args)?;
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

/// A slow grid dimension split across `y` and `z`.
///
/// BOTH ARE CAPPED AT 65535 BLOCKS, and this model exceeds that at 2048x2048:
/// the horizontal resize wants `c * hin / 4` = 65536 blocks on the slow axis, and
/// the level-0 conv3x3 wants `h * segs` = 2048 * 32 = 65536. The failure is loud
/// - `CUDA_ERROR_INVALID_VALUE` at the launch, with the grid printed - but it was
/// unreachable until the arena stopped being the thing that failed first, so it
/// had never run.
///
/// The two kernels that can overflow read the slow index as
/// `blockIdx.z * gridDim.y + blockIdx.y`, which is the number the flat grid gave
/// whenever the flat grid fit (`z` is 0), so no launch geometry changes for a
/// size that already worked.
fn slow_grid(n: usize) -> (u32, u32) {
    const MAX: usize = 65535;
    if n <= MAX {
        (n.max(1) as u32, 1)
    } else {
        (MAX as u32, n.div_ceil(MAX) as u32)
    }
}

fn bytemuck_slice(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytemuck_slice_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
}
