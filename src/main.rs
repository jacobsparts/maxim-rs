//! `maxim`: restore one image with a MAXIM checkpoint.
//!
//!     maxim --model maxim-lol.safetensors -i low.png -o high.png
//!     maxim --model ... -i low.png --device cpu        # pure Rust, no driver
//!     maxim --model ... -i low.png --dump out/         # every named activation
//!
//! The engine builds a plan for the padded input shape, runs it on the chosen
//! backend, and crops the result back the way the reference eval script does.

use maxim::config::Config;
use maxim::host::Host;
use maxim::image;
use maxim::model::Builder;
use maxim::weights::Weights;
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "maxim - MAXIM image restoration (denoise, deblur, derain, dehaze, enhance)

USAGE
  maxim --model <weights.safetensors> -i <input.png> [-o <output.png>] [options]

OPTIONS
  -i, --input <path>    input image (PNG)
  -o, --output <path>   output image (PNG); omitted means run and discard
  -m, --model <path>    weights (safetensors, from tools/convert.py)
      --task <name>     enhancement (default), denoising, deblurring, deraining,
                        dehazing - selects the variant the checkpoint was trained
                        with
      --variant <v>     S-1..S-3, M-1..M-3, overriding --task
      --device <d>      gpu (default) or cpu
      --factor <n>      pad the input to a multiple of n (default 64, as the
                        reference eval script does)
      --dump <dir>      write every named activation as .npy for the parity tool
      --profile         report the plan's size, op count and where the GPU time
                        went, per op and per op kind (MAXIM_TIMELINE=1 does the
                        same thing)
      --legacy-ops      use the kernels as they were before the tiling work, so
                        the speedup can be measured (MAXIM_LEGACY_OPS=1 too)
      --verify-gpu      run the plan ONE OP AT A TIME on the CPU and the GPU,
                        comparing each op's destination between them; reports the
                        first op where they disagree. `--device` is ignored.
  -h, --help            this text
"
    );
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut input = None;
    let mut output = None;
    let mut model = None;
    let mut task = "enhancement".to_string();
    let mut variant = None;
    let mut device = if cfg!(feature = "cuda") { "gpu" } else { "cpu" }.to_string();
    let mut factor = 64usize;
    let mut dump: Option<String> = None;
    let mut profile = false;
    let mut verify_gpu = false;
    let mut legacy_ops = std::env::var_os("MAXIM_LEGACY_OPS").is_some();

    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        let next = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_else(|| usage())
        };
        match a {
            "-i" | "--input" => input = Some(next(&mut i)),
            "-o" | "--output" => output = Some(next(&mut i)),
            "-m" | "--model" => model = Some(next(&mut i)),
            "--task" => task = next(&mut i),
            "--variant" => variant = Some(next(&mut i)),
            "--device" => device = next(&mut i),
            "--factor" => factor = next(&mut i).parse().unwrap_or(64),
            "--dump" => dump = Some(next(&mut i)),
            "--profile" | "--timeline" => profile = true,
            // The pre-tiling kernels are still in the binary, so the speedup is a
            // measurement rather than a claim.
            "--legacy-ops" => legacy_ops = true,
            "--verify-gpu" => verify_gpu = true,
            "-h" | "--help" => usage(),
            other => {
                eprintln!("maxim: unknown option {other}");
                usage()
            }
        }
        i += 1;
    }

    if let Err(e) = run(input, output, model, &task, variant, &device, factor, dump, profile, verify_gpu, legacy_ops) {
        eprintln!("maxim: {e}");
        std::process::exit(1)
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    input: Option<String>,
    output: Option<String>,
    model: Option<String>,
    task: &str,
    variant: Option<String>,
    device: &str,
    factor: usize,
    dump: Option<String>,
    profile: bool,
    verify_gpu: bool,
    legacy_ops: bool,
) -> Result<(), maxim::Error> {
    let (Some(input), Some(model)) = (input, model) else {
        usage()
    };
    // `--legacy-ops` chooses between the tiled and the original kernels, which
    // only exist in a GPU build.
    #[cfg(not(feature = "cuda"))]
    let _ = legacy_ops;
    let cfg = match &variant {
        Some(v) => Config::variant(v)?,
        None => Config::for_task(task)?,
    };

    let img = image::read_png(&input)?;
    let padded = image::preprocess(&img, factor);
    println!(
        "input {}x{} -> padded {}x{} (even {}x{}), {} stage(s), features {}",
        img.w, img.h, padded.img.w, padded.img.h, padded.even_w, padded.even_h,
        cfg.num_stages, cfg.features
    );

    let w = Weights::open(&model)?;
    let t0 = Instant::now();
    let plan = Builder::new(&w, cfg.clone()).build(padded.img.h, padded.img.w)?;
    println!(
        "plan: {} ops, {} buffers, arena {:.1} MiB, weights {:.1} MiB, built in {:.2}s",
        plan.ops.len(),
        plan.bufs.len(),
        plan.arena_bytes() as f64 / 1048576.0,
        plan.weight_bytes() as f64 / 1048576.0,
        t0.elapsed().as_secs_f64()
    );

    let mut host = Host::new(plan);
    if std::env::var_os("MAXIM_TRACE").is_some() {
        // The weight blobs are the one thing the per-tensor diff cannot see:
        // printing the first blob's head is the quickest way to tell a bad
        // permutation from a bad index.
        for (i, w) in host.weights.iter().enumerate().take(2) {
            let head: Vec<f32> = w.iter().take(12).copied().collect();
            println!("weight {i}: len {} head {head:?}", w.len());
        }
        // The full dump order is the graph order of the `rec` calls, which is
        // what makes a "first failing tensor" meaningful: it is the earliest
        // point in the graph where the backends disagree.
        for (i, (n, id)) in host.plan.dumps.iter().enumerate() {
            println!("dump {i} {n} -> id {id} shape {:?}", host.plan.bufs[*id]);
        }
    }
    let input_data = host.plan.bufs[host.plan.input].len();
    assert_eq!(input_data, padded.img.data.len());
    host.input_mut().copy_from_slice(&padded.img.data);

    if verify_gpu {
        return verify_gpu_ops(&mut host);
    }

    let t1 = Instant::now();
    match device {
        "cpu" => {
            let (plan, weights) = (&host.plan, &host.weights);
            let arena = &mut host.arena;
            maxim::exec_cpu::run(plan, weights, arena);
        }
        #[cfg(feature = "cuda")]
        "gpu" => {
            let mut gpu = maxim::exec_gpu::Gpu::new(&host.plan)?;
            gpu.set_timeline(profile);
            gpu.set_legacy_ops(legacy_ops);
            let mut out = vec![0.0f32; host.plan.bufs[host.plan.output].len()];
            let o = host.plan.offs[host.plan.input];
            let n = host.plan.bufs[host.plan.input].len();
            let img = host.arena[o..o + n].to_vec();
            gpu.run(&host.plan, &host.weights, &img, &mut out)?;
            println!(
                "gpu: {} launches, arena {:.1} MiB, weights {:.1} MiB, pool {:.1} KiB",
                gpu.ops_launched(),
                gpu.arena_bytes() as f64 / 1048576.0,
                gpu.weight_bytes() as f64 / 1048576.0,
                gpu.pool_bytes() as f64 / 1024.0
            );
            if let Some(tl) = gpu.timeline() {
                report_timeline(&host.plan, tl);
            }
            // Nothing downstream reads the arena unless it is being dumped or
            // cropped, and on a profiled run the copy is the largest thing in
            // the measurement - 942 MiB back over PCIe is ~0.4 s of the total.
            if dump.is_some() || output.is_some() {
                // The whole arena comes back, not just the output: the dump step
                // and the output crop both read it out of `host.arena`, and a
                // named activation is only meaningful if the buffer it lives in
                // was written by THIS run.
                let mut arena = vec![0.0f32; host.plan.arena_len];
                gpu.read_arena(&mut arena)?;
                host.arena = arena;
            }
        }
        #[cfg(not(feature = "cuda"))]
        "gpu" => return Err("this binary was built without the `cuda` feature; use --device cpu".into()),
        other => return Err(format!("unknown device {other} (expected gpu or cpu)").into()),
    }
    let secs = t1.elapsed().as_secs_f64();
    println!("ran in {:.2}s ({:.3} s/op)", secs, secs / host.plan.ops.len() as f64);

    if let Some(dir) = dump {
        std::fs::create_dir_all(&dir)?;
        let names: Vec<(String, usize)> = host.plan.dumps.clone();
        for (name, id) in names {
            let s = host.plan.bufs[id];
            let data = host.take(id);
            let path = format!("{dir}/{}.npy", name.replace('/', "_"));
            image::save_npy(&path, s.c, s.h, s.w, &data)?;
        }
        println!("wrote {} activations to {dir}", host.plan.dumps.len());
    }

    if let Some(out) = output {
        let o = host.plan.offs[host.plan.output];
        let n = host.plan.bufs[host.plan.output].len();
        let pred = image::Image {
            c: 3,
            h: padded.img.h,
            w: padded.img.w,
            data: host.arena[o..o + n].to_vec(),
        };
        let cropped = image::crop_out(&pred, &padded);
        image::write_png(&out, &cropped)?;
        println!("wrote {out}");
    }

    if profile {
        println!("ops: {}", host.plan.ops.len());
    }
    Ok(())
}

/// Where the GPU time went, grouped by op kind.
///
/// The per-op numbers come from a CUDA event around each op, so they are device
/// time and they sum to the run. What matters is the SHARE: the graph is 1840
/// ops, and one kernel family holding a third of the time is a different problem
/// from every op costing the same, which is what the total alone cannot say.
#[cfg(feature = "cuda")]
fn report_timeline(plan: &maxim::model::Plan, tl: &maxim::exec_gpu::Timeline) {
    use std::collections::HashMap;
    let mut by_kind: HashMap<String, (f32, usize, usize)> = HashMap::new();
    for (i, op) in plan.ops.iter().enumerate() {
        // `describe` splits on the first space, and its first word is the op's
        // family: conv1x1, chanln, blockperm, ... This is a report, not a hot
        // loop, so the per-op format! is affordable and the grouping stays in
        // step with the description the rest of the tooling prints.
        let d = maxim::exec_cpu::describe(op);
        let kind = d.split_whitespace().next().unwrap_or("?").to_string();
        let e = by_kind.entry(kind).or_insert((0.0, 0, 0));
        e.0 += tl.per_op_ms[i];
        e.1 += 1;
        e.2 += tl.kernels_per_op[i];
    }
    let mut rows: Vec<_> = by_kind.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "timeline: {:.2}s total, {} ops, {} launches",
        tl.total_ms / 1000.0,
        tl.per_op_ms.len(),
        tl.kernels_per_op.iter().sum::<usize>()
    );
    println!("{:<10} {:>7} {:>9} {:>10} {:>7}  {}", "op", "count", "total ms", "ms/op", "kernels", "share");
    for (kind, (ms, count, kernels)) in rows {
        println!(
            "{:<10} {:>7} {:>9.1} {:>10.3} {:>7}  {:>5.1}%",
            kind,
            count,
            ms,
            ms / count as f32,
            kernels,
            100.0 * ms / tl.total_ms.max(1e-6)
        );
    }
    let mut worst: Vec<(usize, f32)> = tl.per_op_ms.iter().copied().enumerate().collect();
    worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("slowest ops:");
    for (i, ms) in worst.into_iter().take(5) {
        println!("  op {i} {:.2} ms  {}", ms, maxim::exec_cpu::describe(&plan.ops[i]));
    }
}

/// Run the plan one op at a time on both backends and report the first op whose
/// destination disagrees.
///
/// This exists because "the GPU plot looks wrong" is not a diagnostic. The two
/// backends share the op list by construction, so the only place they can differ
/// is a kernel or a launch - and comparing the destination of every op in graph
/// order names the culprit op directly. The CPU side doubles as the reference:
/// it is already known to match torch to ~1e-6 on every named tensor.
///
/// Both arenas are stepped in LOCKSTEP from the same input, so op i sees the
/// same preceding state on both sides; a mismatch therefore reports the op that
/// produced it, not an accumulation.
#[cfg(feature = "cuda")]
fn verify_gpu_ops(host: &mut Host) -> Result<(), maxim::Error> {
    let plan = &host.plan;
    let o = plan.offs[plan.input];
    let n = plan.bufs[plan.input].len();
    let img = host.arena[o..o + n].to_vec();

    let mut gpu = maxim::exec_gpu::Gpu::new(plan)?;
    gpu.begin(plan, &img)?;
    let mut cpu_arena = vec![0.0f32; plan.arena_len];
    cpu_arena[o..o + n].copy_from_slice(&img);

    let mut buf = Vec::new();
    let mut worst_all = 0.0f32;
    let mut bad = 0usize;
    for (i, op) in plan.ops.iter().enumerate() {
        maxim::exec_cpu::step(plan, &host.weights, &mut cpu_arena, i);
        gpu.step_op(plan, &host.weights, i)?;

        // Compare the op's DESTINATION: that is the buffer this op is
        // responsible for. Sources are compared implicitly when the op that
        // wrote them ran.
        let mut ids = Vec::new();
        op.operands(&mut ids);
        let dst = ids[0];
        let len = plan.bufs[dst].len();
        buf.resize(len, 0.0);
        gpu.read_buf(plan, dst, &mut buf)?;
        let c = &cpu_arena[plan.offs[dst]..plan.offs[dst] + len];
        let mut mx = 0.0f32;
        let mut at = 0usize;
        for (j, (a, b)) in c.iter().zip(&buf).enumerate() {
            let d = (a - b).abs();
            if d > mx {
                mx = d;
                at = j;
            }
        }
        // f32 kernels differ in accumulation order, so the bar is generous and
        // relative to the magnitude the buffer actually holds.
        let scale = c.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        if mx > 1e-3 * scale {
            bad += 1;
            if bad <= 20 {
                println!(
                    "op {i}: {} `{}#{}` differs by {mx:.4e} (scale {scale:.3}, element {at})",
                    maxim::exec_cpu::describe(op),
                    plan.labels[dst],
                    dst
                );
            }
        }
        worst_all = worst_all.max(mx / scale);
    }
    println!(
        "--verify-gpu: {} ops, {bad} disagree beyond 1e-3 relative (worst {worst_all:.3e} relative)",
        plan.ops.len()
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn verify_gpu_ops(_host: &mut Host) -> Result<(), maxim::Error> {
    Err("--verify-gpu needs the `cuda` feature".into())
}
