//! `mx_convt2x2s2` against the toolkit's `lg_conv_t2x2`, at the geometries the
//! graph actually runs.
//!
//! WHY THIS EXISTS. The toolkit promoted a stride-2 2x2 transposed convolution
//! (`lg_conv_t2x2`) out of scunet-rs, and this engine has carried its own form of
//! the same operation (`mx_convt2x2s2`) since its decoder was written. The two
//! agree on the OPERATION - weight `[c_in][c_out][2][2]`, output written at
//! `(2y+ky, 2x+kx)` with no tap flip, a nullable bias as the accumulator's
//! initial value - so the only question is which is faster at the shapes this
//! engine runs, and that is a measurement rather than a preference.
//!
//! WHY IT IS AN EXAMPLE RATHER THAN A FLAG. An example can reach
//! `maxim::cuda::Cuda` and load both fatbins directly (see the module doc in
//! `src/cuda.rs`), so it needs no new argument in `src/main.rs` and no new
//! branch in the executor - and it is not part of any release binary.
//!
//! THE TWO ARE BIT-EXACT, AND THAT IS MEASURED RATHER THAN ASSUMED. The loop
//! nesting differs - this engine's iterates `ic` with the four taps inner, the
//! toolkit's iterates `ci` alone and takes its single tap from the output's
//! parity - but each output element sums ONE tap over `ci` in ascending order
//! either way, so the summation order is the same and the two answers are
//! identical to the last bit at every shape below (max |d| = 0). A swap here is a
//! rename, not an arithmetic change, which is what makes it a pure speed
//! question.
//!
//! Run it with the GPU idle:
//!
//!     cargo run --release --example convt2x2
//!
//! The ratios are what matter, not the absolute times: this card is shared and
//! its clock has been seen to swing by more than 10x between windows, so the two
//! kernels are ALTERNATED inside one window and the best round of each is taken.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("convt2x2: this example needs the cuda feature (it is on by default)");
}

#[cfg(feature = "cuda")]
fn main() {
    if let Err(e) = run() {
        eprintln!("convt2x2: {e}");
        std::process::exit(1);
    }
}

#[cfg(feature = "cuda")]
use lightgpu::vm::{Args, DevBuf, Event, Launch};

#[cfg(feature = "cuda")]
const ITERS: usize = 20;
#[cfg(feature = "cuda")]
const ROUNDS: usize = 3;

/// Output channels per thread in `mx_convt2x2s2`, and the widest block width the
/// launcher chooses (`src/exec_gpu.rs`: 64 output channels and below, else 16).
#[cfg(feature = "cuda")]
const CD_C: usize = 8;
#[cfg(feature = "cuda")]
const LG_OC: usize = 8;

#[cfg(feature = "cuda")]
fn run() -> Result<(), String> {
    // (c_in, c_out, input h, input w). The first three are the graph's own
    // 32->32 @448x640, 64->64 @224x320 and 128->128 @112x160; the rest span the
    // c_in values the published checkpoints reach at other input sizes.
    let cases: [(usize, usize, usize, usize); 6] = [
        (32, 32, 448, 640),
        (64, 64, 224, 320),
        (128, 128, 112, 160),
        (32, 32, 128, 128),
        (96, 96, 96, 96),
        (64, 64, 32, 32),
    ];

    let cuda = maxim::cuda::Cuda::new().map_err(|e| e.to_string())?;
    for k in ["mx_convt2x2s2", "lg_conv_t2x2"] {
        if !cuda.project.has(k) && !cuda.toolkit.has(k) {
            return Err(format!("`{k}` is in neither module - is it listed in build.rs?"));
        }
    }
    println!(
        "convt2x2: mx_convt2x2s2 (this engine: {CD_C} channels and 2 positions a thread, ic outer)"
    );
    println!(
        "          lg_conv_t2x2   (toolkit:     {LG_OC} channels a thread, the tap from the output's parity)"
    );
    println!(
        "  {:>15} {:>6} {:>6} {:>10} {:>10} {:>9} {:>11}",
        "out plane", "c_in", "c_out", "mx ms", "toolkit ms", "toolkit/", "max |d|"
    );

    let mut total = [0.0f64; 2];
    for (c_in, c_out, h, wd) in cases {
        let (oh, ow) = (2 * h, 2 * wd);
        let mut s = 2026_0501u32 ^ (c_in as u32) << 8 ^ (h as u32);
        let mut rng = move |n: usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                v.push(((s >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0);
            }
            v
        };
        let x = rng(c_in * h * wd);
        let wt = rng(c_in * c_out * 4);
        let bs = rng(c_out);
        let xd = DevBuf::from_host(&x)?;
        let wdev = DevBuf::from_host(&wt)?;
        let bdev = DevBuf::from_host(&bs)?;
        let out_mx = DevBuf::zeros(c_out * oh * ow * 4)?;
        let out_lg = DevBuf::zeros(c_out * oh * ow * 4)?;

        // The two argument lists are the same eight values in the same order.
        let mut a_mx = Args::new();
        a_mx.ptr(xd.ptr)
            .ptr(wdev.ptr)
            .ptr(bdev.ptr)
            .ptr(out_mx.ptr)
            .i32(c_in as i32)
            .i32(c_out as i32)
            .i32(h as i32)
            .i32(wd as i32);
        let mut a_lg = Args::new();
        a_lg.ptr(xd.ptr)
            .ptr(wdev.ptr)
            .ptr(bdev.ptr)
            .ptr(out_lg.ptr)
            .i32(c_in as i32)
            .i32(c_out as i32)
            .i32(h as i32)
            .i32(wd as i32);

        // The geometries `src/exec_gpu.rs` launches at. The block width is
        // per-shape there because a 2P-wide window per thread needs several warps
        // per row to fill a warp's load slots - a narrow block measures 0.59x for
        // the same tile, which is the effect this column would otherwise hide.
        let bx: u32 = if c_out <= 32 {
            64
        } else if c_out <= 64 {
            80
        } else {
            16
        };
        let mx_grid = (
            wd.div_ceil((bx * 2) as usize).max(1) as u32,
            c_out.div_ceil(CD_C).max(1) as u32,
            h as u32,
        );
        let mx_block = (bx, 1, 1);
        // The toolkit's: one output pixel per position on x, channel tiles on y.
        let lg_grid = (
            (oh * ow).div_ceil(256) as u32,
            c_out.div_ceil(LG_OC).max(1) as u32,
            1,
        );
        let lg_block = (256u32, 1, 1);

        // Warm both, so the first round is not paying for a cold module.
        launch(&cuda, "mx_convt2x2s2", mx_grid, mx_block, &mut a_mx)?;
        launch(&cuda, "lg_conv_t2x2", lg_grid, lg_block, &mut a_lg)?;

        let mut best = [f64::MAX; 2];
        for _ in 0..ROUNDS {
            best[0] = best[0].min(time(&cuda, "mx_convt2x2s2", mx_grid, mx_block, &mut a_mx)?);
            best[1] = best[1].min(time(&cuda, "lg_conv_t2x2", lg_grid, lg_block, &mut a_lg)?);
        }
        total[0] += best[0];
        total[1] += best[1];

        // ONE RUN OF EACH, so the two answers can be compared rather than
        // assumed equal.
        launch(&cuda, "mx_convt2x2s2", mx_grid, mx_block, &mut a_mx)?;
        let mut got = vec![0.0f32; c_out * oh * ow];
        out_mx.download(&mut got)?;
        launch(&cuda, "lg_conv_t2x2", lg_grid, lg_block, &mut a_lg)?;
        let mut alt = vec![0.0f32; c_out * oh * ow];
        out_lg.download(&mut alt)?;
        let mut worst = 0.0f32;
        let mut scale = 0.0f32;
        for (a, b) in got.iter().zip(&alt) {
            worst = worst.max((a - b).abs());
            scale = scale.max(a.abs());
        }

        println!(
            "  {:>6}x{:<8} {:>6} {:>6} {:>10.3} {:>10.3} {:>9.3} {:>11.2e}{}",
            oh,
            ow,
            c_in,
            c_out,
            best[0],
            best[1],
            best[1] / best[0],
            worst,
            if worst > 1e-3 * scale.max(1.0) { "  DISAGREE" } else { "" }
        );
    }
    println!(
        "  {:>15} {:>6} {:>6} {:>10.3} {:>10.3} {:>9.3}",
        "TOTAL",
        "",
        "",
        total[0],
        total[1],
        total[1] / total[0]
    );
    println!("  toolkit/ is ms_toolkit / ms_mx: below 1 the toolkit's kernel is faster, above 1 this");
    println!("  engine's is. The two are bit-exact (the last column is the measured proof), so a");
    println!("  swap would be a rename and this is a pure speed comparison.");
    Ok(())
}

#[cfg(feature = "cuda")]
fn module<'a>(cuda: &'a maxim::cuda::Cuda, name: &str) -> &'a lightgpu::vm::Module {
    if cuda.project.has(name) {
        &cuda.project
    } else {
        &cuda.toolkit
    }
}

#[cfg(feature = "cuda")]
fn launch(
    cuda: &maxim::cuda::Cuda,
    name: &str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    args: &mut Args,
) -> Result<(), String> {
    args.launch(module(cuda, name), name, Launch::new(grid, block))
}

/// The mean milliseconds of `ITERS` launches, bracketed by a CUDA event pair.
///
/// The pair wraps the WHOLE LOOP rather than one launch: every launch here is
/// asynchronous, so a host clock around one measures the cost of the call, and
/// the interesting number is the kernel.
#[cfg(feature = "cuda")]
fn time(
    cuda: &maxim::cuda::Cuda,
    name: &str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    args: &mut Args,
) -> Result<f64, String> {
    let e0 = Event::new()?;
    let e1 = Event::new()?;
    e0.record()?;
    for _ in 0..ITERS {
        launch(cuda, name, grid, block, args)?;
    }
    e1.record()?;
    e1.synchronize()?;
    Ok(e0.elapsed_ms(&e1)? as f64 / ITERS as f64)
}
