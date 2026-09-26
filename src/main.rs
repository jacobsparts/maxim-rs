//! `maxim`: restore one image with a MAXIM checkpoint.
//!
//!     maxim -m maxim-lol.safetensors -i low.png -o high.png
//!     maxim -m maxim-lol.safetensors -i low.png -o high.png --device cpu
//!
//! The engine builds a plan for the padded input shape, runs it on the chosen
//! backend, and crops the result back the way the reference eval script does.

use maxim::host::Host;
use maxim::image;
use maxim::model::Builder;
use maxim::weights::Weights;
use std::time::Instant;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() -> ! {
    eprintln!(
        "maxim {VERSION} - MAXIM image restoration on lightgpu

USAGE:
    maxim --model <weights.safetensors> -i <in.png> -o <out.png> [options]

OPTIONS:
    -m, --model <path>    converted .safetensors checkpoint (see tools/convert.py)
    -i, --input <path>    input PNG, or - for stdin (default: stdin)
    -o, --output <path>   output PNG, or - for stdout (default: stdout)
        --device <dev>    gpu or cpu (default: gpu when the CUDA driver can be
                          brought up, cpu otherwise; a CPU-only build is always
                          cpu)
        --cpu             same as --device cpu
        --gpu             same as --device gpu, and refuses to fall back
    -q, --quiet           no progress output
    -h, --help            this text
    -V, --version         print the version"
    );
    // THE DEVELOPMENT FLAGS ARE LISTED ONLY BY A BINARY THAT HAS THEM. Help text
    // that advertises an option the parser rejects is worse than no help: a
    // caller reads the list, passes a flag, and gets a refusal it was told would
    // work.
    #[cfg(feature = "dev")]
    eprintln!(
        "
DEVELOPMENT ONLY (this build has `--features dev`; a release build has none of
these, and rejects them by name):
        --dump <dir>      write every named activation as .npy, for
                          tools/compare.py against the reference
        --profile         report where the time went, per op kind and per shape
                          (the same table for both backends; MAXIM_TIMELINE=1
                          does it for the GPU too)
        --verify-gpu      run the plan ONE OP AT A TIME on the CPU and the GPU,
                          comparing each op's destination between them, and
                          report the first op where they disagree
        --factor <n>      pad the input to a multiple of n (default 64, as the
                          reference eval script does)"
    );
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let mut model: Option<String> = None;
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    // A CPU-only build has no GPU backend to default to, so it defaults to the
    // one it can actually run.
    let mut device = if cfg!(feature = "cuda") { "gpu" } else { "cpu" }.to_string();
    // Set only when the caller NAMED the GPU. Without it, a GPU that cannot be
    // brought up is not fatal: the engine falls back to the CPU backend, which
    // is what lets one binary run on a machine with no NVIDIA driver at all.
    let mut force_gpu = false;
    let mut quiet = false;
    // DEV-ONLY STATE, AND ITS ABSENCE IS WHAT KEEPS THE FLAGS OUT. A release
    // build has no flag that sets any of these, so the variables do not exist in
    // it - which also means a later edit cannot re-expose a development flag by
    // wiring it to state that is still being parsed.
    #[cfg(feature = "dev")]
    let mut factor = 64usize;
    #[cfg(feature = "dev")]
    let mut dump: Option<String> = None;
    #[cfg(feature = "dev")]
    let mut profile = false;
    #[cfg(feature = "dev")]
    let mut verify_gpu = false;

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let next = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_else(|| usage())
        };
        match a {
            "-m" | "--model" => model = Some(next(&mut i)),
            "-i" | "--input" => input = Some(next(&mut i)),
            "-o" | "--output" => output = Some(next(&mut i)),
            "--device" => {
                device = next(&mut i);
                force_gpu = device == "gpu";
            }
            "--cpu" => device = "cpu".to_string(),
            "--gpu" => {
                device = "gpu".to_string();
                force_gpu = true;
            }
            "-q" | "--quiet" => quiet = true,
            // THE DEVELOPMENT FLAGS, AND ONLY A DEVELOPMENT BUILD HAS THEM. A
            // release build refuses them BY NAME rather than ignoring them,
            // which matters most for `--dump`: a script that asked for a dump
            // and did not get one must not carry on as if it had.
            #[cfg(feature = "dev")]
            "--dump" => dump = Some(next(&mut i)),
            #[cfg(feature = "dev")]
            "--profile" | "--timeline" => profile = true,
            #[cfg(feature = "dev")]
            "--verify-gpu" => verify_gpu = true,
            #[cfg(feature = "dev")]
            "--factor" => factor = next(&mut i).parse().unwrap_or(64),
            #[cfg(not(feature = "dev"))]
            "--dump" | "--profile" | "--timeline" | "--verify-gpu" | "--factor" => {
                eprintln!("maxim: `{a}` is a development flag and this is a release build");
                eprintln!("maxim: rebuild with `cargo build --release --features dev` for");
                eprintln!("maxim: --dump, --profile, --verify-gpu and --factor");
                std::process::exit(2);
            }
            "-h" | "--help" => usage(),
            "-V" | "--version" => {
                println!("maxim {VERSION}");
                return;
            }
            other => {
                eprintln!("maxim: unknown argument `{other}`");
                usage();
            }
        }
        i += 1;
    }

    let model = model.unwrap_or_else(|| {
        eprintln!("maxim: --model is required (see tools/convert.py)");
        usage()
    });

    #[cfg(feature = "dev")]
    let dev = Dev { factor, dump, profile, verify_gpu };
    #[cfg(not(feature = "dev"))]
    let dev = Dev;

    if let Err(e) = run(&model, input.as_deref(), output.as_deref(), &device, force_gpu, quiet, dev)
    {
        eprintln!("maxim: {e}");
        std::process::exit(1)
    }
}

/// The development-only knobs, so that `run` can be written once. A release
/// build's `Dev` has no fields and `run` reads them as constants, which is what
/// keeps the development paths out of a release binary rather than merely
/// unreachable in it.
#[cfg(feature = "dev")]
struct Dev {
    factor: usize,
    dump: Option<String>,
    profile: bool,
    verify_gpu: bool,
}
#[cfg(not(feature = "dev"))]
struct Dev;

#[allow(clippy::too_many_arguments)]
fn run(
    model: &str,
    input: Option<&str>,
    output: Option<&str>,
    device: &str,
    force_gpu: bool,
    quiet: bool,
    dev: Dev,
) -> Result<(), maxim::Error> {
    #[cfg(feature = "dev")]
    let (factor, dump, profile, verify_gpu) =
        (dev.factor, dev.dump.clone(), dev.profile, dev.verify_gpu);
    // A RELEASE BUILD'S `Dev` IS A UNIT STRUCT AND THESE ARE CONSTANTS. That is
    // the point of the split: the development paths below are not "unreachable
    // at run time", they are not compiled, so a release binary has no code that
    // could act on a dump directory or a pad multiple even if it were handed
    // one. `factor` is the only one of the four that has a life outside a
    // development build - the reference eval script pads by 64 and so does this
    // engine, unconditionally, and a caller who could change it would be able to
    // produce an image that does not match the published pipeline.
    #[cfg(not(feature = "dev"))]
    let (factor, _dump, profile, _verify_gpu) = (64usize, None::<String>, false, false);
    // The unit `Dev` is all a release build has left of the development flags.
    #[cfg(not(feature = "dev"))]
    let _ = dev;

    // THE CHECKPOINT SAYS WHAT IT IS. The variant is a property of the file, not
    // a flag: `Weights::open` derives it from the weights, so a mismatched
    // `--variant` cannot build a graph that reads parameters which are not there.
    let w = Weights::open(model)?;
    let cfg = w.config.clone();

    // The image. `-` or absent means stdin, so the engine drops into a shell
    // pipeline without a temporary file.
    let img = match input {
        None | Some("-") => {
            use std::io::Read;
            let mut buf = Vec::new();
            std::io::stdin()
                .read_to_end(&mut buf)
                .map_err(|e| format!("read stdin: {e}"))?;
            image::read_png_stream(&buf[..], "stdin")?
        }
        Some(p) => image::read_png(p)?,
    };
    let padded = image::preprocess(&img, factor);
    if !quiet {
        eprintln!(
            "maxim {VERSION}: {}x{} -> padded {}x{} (even {}x{}), {} stage(s), features {}",
            img.w, img.h, padded.img.w, padded.img.h, padded.even_w, padded.even_h,
            cfg.num_stages, cfg.features
        );
    }

    let t0 = Instant::now();
    // A RELEASE BUILD DOES NOT CARRY THE DUMP BUFFERS. Naming an activation pins
    // it to the end of the run, because that is when `--dump` snapshots it - and
    // 43 names held to the end of an 1840-op graph is not a detail: it is the
    // difference between an arena of 7871 MiB and one of 5411 MiB at 2048x2048,
    // 2.01x the graph's true live set against 1.15x. `--dump` is a development
    // flag that a release binary refuses BY NAME, so a release build was paying
    // that memory for a feature it does not contain. `tests/packing.rs` is what
    // makes the swap safe: it runs the same graph with and without the names and
    // demands a bit-identical result.
    let b = Builder::new(&w, cfg);
    #[cfg(feature = "dev")]
    let b = if dump.is_some() { b } else { b.without_dumps() };
    #[cfg(not(feature = "dev"))]
    let b = b.without_dumps();
    let plan = b.build(padded.img.h, padded.img.w)?;
    if !quiet {
        eprintln!(
            "maxim: plan {} ops, {} buffers, arena {:.1} MiB, weights {:.1} MiB, built in {:.2}s",
            plan.ops.len(),
            plan.bufs.len(),
            plan.arena_bytes() as f64 / 1048576.0,
            plan.weight_bytes() as f64 / 1048576.0,
            t0.elapsed().as_secs_f64()
        );
    }

    // ------------------------------------------------------------- the backend
    //
    // WHICH BACKEND RUNS IS DECIDED BEFORE THE HOST ARENA IS SIZED, and the order
    // is load-bearing. A GPU run reads exactly one thing out of the host arena -
    // the input image - and the device executes off its own copy, so allocating
    // the other 942.7 MiB would be memory and page-fault time spent so that
    // nothing can read it. A CPU run executes off the whole arena. Sizing the
    // host arena from the device NAME alone, before knowing whether the GPU can
    // actually be brought up, is how a fallback turns into a panic instead of a
    // slow run: the CPU backend then reads past the end of a buffer that was
    // never allocated.
    //
    // ONLY THE DRIVER FAILING TO COME UP SENDS A RUN TO THE CPU. That case is the
    // whole reason one binary carries both backends - a machine with no NVIDIA
    // driver must still work - and it is not an error unless the caller named the
    // GPU. A pass that fails for any other reason is REPORTED; see the run below.
    // A CPU-only build has no GPU branch at all, so naming the GPU is answered
    // with that fact rather than by running the CPU engine and reporting the
    // performance of something the caller did not ask for.
    #[cfg(not(feature = "cuda"))]
    {
        let _ = force_gpu;
        if device == "gpu" {
            return Err("this build has no cuda feature; use --device cpu".into());
        }
    }
    // A CPU-only build cannot reach the GPU, so its answer is the one that needs
    // no driver probe. The two bindings are the same name because everything
    // below asks the same question of it.
    #[cfg(not(feature = "cuda"))]
    let use_cpu = true;
    #[cfg(feature = "cuda")]
    let use_cpu = if device != "gpu" {
        true
    } else {
        match maxim::exec_gpu::driver_ready() {
            Ok(()) => false,
            Err(e) if force_gpu => return Err(e),
            Err(e) => {
                // NOT PROGRESS OUTPUT, SO `--quiet` DOES NOT SILENCE IT. Which
                // backend ran is a property of the result, like a warning, and a
                // caller that asked for quiet to keep its logs small still needs
                // to know it got the CPU.
                eprintln!("maxim: cuda: {e}");
                eprintln!("maxim: falling back to the CPU backend (--gpu forces the GPU)");
                true
            }
        }
    };

    // The full host arena is needed wherever the host actually holds the
    // activations: the CPU backend, the per-op verify walk, and a dump (which
    // reads named activations out of it). Only a development build can reach the
    // two paths that ask for the full length.
    let want_full_arena = {
        #[cfg(feature = "dev")]
        {
            verify_gpu || dump.is_some()
        }
        #[cfg(not(feature = "dev"))]
        {
            false
        }
    };
    let host_arena_len = if use_cpu || want_full_arena {
        plan.arena_len
    } else {
        plan.offs[plan.input] + plan.bufs[plan.input].len()
    };
    if use_cpu {
        // THE CPU BACKEND IS THE LAST ONE, SO IT REFUSES RATHER THAN FAILING.
        // There is nothing below it to fall back to, and a pass that runs a
        // machine out of RAM does not fail - it swaps. So the footprint is
        // computed and compared against what the machine has BEFORE the arena is
        // allocated, the same way the GPU side compares its plan against free
        // VRAM; a refusal therefore costs nothing.
        maxim::memguard::check(&plan)?;
    }
    let mut host = Host::with_arena_len(plan, host_arena_len);
    #[cfg(feature = "dev")]
    if std::env::var_os("MAXIM_TRACE").is_some() {
        // The weight blobs are the one thing the per-tensor diff cannot see:
        // printing the first blob's head is the quickest way to tell a bad
        // permutation from a bad index.
        for (i, w) in host.weights.iter().enumerate().take(2) {
            let head: Vec<f32> = w.iter().take(12).copied().collect();
            eprintln!("weight {i}: len {} head {head:?}", w.len());
        }
        // The full dump order is the graph order of the `rec` calls, which is
        // what makes a "first failing tensor" meaningful: it is the earliest
        // point in the graph where the backends disagree.
        for (i, (n, id)) in host.plan.dumps.iter().enumerate() {
            eprintln!("dump {i} {n} -> id {id} shape {:?}", host.plan.bufs[*id]);
        }
    }
    let input_data = host.plan.bufs[host.plan.input].len();
    assert_eq!(input_data, padded.img.data.len());
    host.input_mut().copy_from_slice(&padded.img.data);

    #[cfg(feature = "dev")]
    if verify_gpu {
        return verify_gpu_ops(&mut host);
    }

    let t1 = Instant::now();
    // The CPU backend's per-op census, filled in when `--profile` is on. Only a
    // development build can turn the flag on, so the whole measuring loop is
    // gone from a release binary rather than merely unreachable in it.
    #[cfg(feature = "dev")]
    let mut cpu_timeline: Option<(Vec<f32>, f32)> = None;
    #[cfg(feature = "dev")]
    if use_cpu && profile {
        // Per-op HOST timing, the same census the GPU backend gets from CUDA
        // events. `step` is the call the op-by-op verify walk already uses, so
        // the profile runs the plan the way that path does instead of through a
        // second implementation of the loop.
        let n = host.plan.ops.len();
        let mut ms = vec![0.0f32; n];
        let t = Instant::now();
        for i in 0..n {
            let t0 = Instant::now();
            maxim::exec_cpu::step(&host.plan, &host.weights, &mut host.arena, i);
            ms[i] = t0.elapsed().as_secs_f32() * 1000.0;
        }
        cpu_timeline = Some((ms, t.elapsed().as_secs_f32() * 1000.0));
    } else if use_cpu {
        maxim::exec_cpu::run(&host.plan, &host.weights, &mut host.arena);
    }
    #[cfg(not(feature = "dev"))]
    if use_cpu {
        maxim::exec_cpu::run(&host.plan, &host.weights, &mut host.arena);
    }

    let out_data: Vec<f32> = if use_cpu {
        let (o, n) = (host.plan.offs[host.plan.output], host.plan.bufs[host.plan.output].len());
        host.arena[o..o + n].to_vec()
    } else {
        let want_arena = {
            #[cfg(feature = "dev")]
            {
                dump.is_some()
            }
            #[cfg(not(feature = "dev"))]
            {
                false
            }
        };
        // A FAILED PASS IS NOT RETRIED ON THE CPU, AND THAT IS DELIBERATE. It
        // was, until this commit. A large image wants more VRAM than the card
        // has, so a memory failure switched to the CPU backend and finished the
        // job - which hid the reason, and the reason is that the image does not
        // fit, which the caller can act on. It also moved the failure somewhere
        // worse: the CPU twin holds the whole arena in host RAM, and a machine
        // short of memory swaps rather than erroring. It did not even work, in
        // the end - the host arena was sized for a GPU run, so the CPU backend
        // read past its end and panicked. `exec_gpu::Gpu::new` now sizes the
        // pass and refuses one that cannot fit, with the numbers, before it
        // allocates anything at all.
        run_gpu(&mut host, profile, quiet, want_arena)?
    };

    let secs = t1.elapsed().as_secs_f64();
    if !quiet {
        eprintln!("maxim: ran in {:.2}s ({:.3} s/op)", secs, secs / host.plan.ops.len() as f64);
    }

    #[cfg(feature = "dev")]
    if let Some(dir) = &dump {
        std::fs::create_dir_all(dir)?;
        let names: Vec<(String, usize)> = host.plan.dumps.clone();
        for (name, id) in names {
            let s = host.plan.bufs[id];
            let data = host.take(id);
            let path = format!("{dir}/{}.npy", name.replace('/', "_"));
            image::save_npy(&path, s.c, s.h, s.w, &data)?;
        }
        eprintln!("maxim: wrote {} activations to {dir}", host.plan.dumps.len());
    }

    #[cfg(feature = "dev")]
    if let Some((ms, total)) = &cpu_timeline {
        report_cpu_timeline(&host.plan, ms, *total);
    }

    // The reference crops the padded prediction back to the original size; `-`
    // or absent means stdout, so the result can be piped straight to a viewer.
    let pred = image::Image { c: 3, h: padded.img.h, w: padded.img.w, data: out_data };
    let cropped = image::crop_out(&pred, &padded);
    match output {
        None | Some("-") => {
            use std::io::Write;
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            image::write_png_stream(&mut lock, &cropped, "stdout")?;
            lock.flush().map_err(|e| format!("stdout: {e}"))?;
        }
        Some(p) => {
            image::write_png(p, &cropped)?;
            if !quiet {
                eprintln!("maxim: wrote {p}");
            }
        }
    }
    Ok(())
}

/// Run the plan on the GPU and bring the result back.
///
/// `want_arena` is what `--dump` needs: the whole host arena, so that every
/// named activation can be snapshotted. A plain run reads ONE small tensor
/// instead - the output `-o` crops from - because how much comes back is not a
/// small difference: the whole arena is 942.7 MiB at the eval size, about 0.4 s
/// over this link, against 2.9 MiB for the output.
#[cfg(feature = "cuda")]
fn run_gpu(host: &mut Host, profile: bool, quiet: bool, want_arena: bool) -> Result<Vec<f32>, maxim::Error> {
    let mut gpu = maxim::exec_gpu::Gpu::new(&host.plan)?;
    if profile {
        gpu.set_timeline(true);
    }
    if !quiet {
        eprintln!("maxim: device {}", gpu.device_name());
    }
    let o = host.plan.offs[host.plan.input];
    let n = host.plan.bufs[host.plan.input].len();
    let img = host.arena[o..o + n].to_vec();
    let mut out = vec![0.0f32; host.plan.bufs[host.plan.output].len()];
    gpu.run(&host.plan, &host.weights, &img, &mut out)?;
    if !quiet {
        eprintln!(
            "maxim: {} launches, arena {:.1} MiB, weights {:.1} MiB, pool {:.1} KiB",
            gpu.ops_launched(),
            gpu.arena_bytes() as f64 / 1048576.0,
            gpu.weight_bytes() as f64 / 1048576.0,
            gpu.pool_bytes() as f64 / 1024.0
        );
    }
    #[cfg(feature = "dev")]
    if let Some(tl) = gpu.timeline() {
        report_timeline(&host.plan, tl);
    }
    if want_arena {
        let mut arena = vec![0.0f32; host.plan.arena_len];
        gpu.read_arena(&mut arena)?;
        host.arena = arena;
        let (o, n) = (host.plan.offs[host.plan.output], host.plan.bufs[host.plan.output].len());
        Ok(host.arena[o..o + n].to_vec())
    } else {
        gpu.read_into(host.plan.offs[host.plan.output], &mut out)?;
        Ok(out)
    }
}

/// A CPU-only build reaches this only if the caller named the GPU, which `run`
/// answers before it gets here.
#[cfg(not(feature = "cuda"))]
fn run_gpu(_host: &mut Host, _profile: bool, _quiet: bool, _want_arena: bool) -> Result<Vec<f32>, maxim::Error> {
    Err("this build has no cuda feature; use --device cpu".into())
}

/// Where the CPU time went, in the same shape as the GPU report.
///
/// The GPU's numbers come from a CUDA event around each op; these come from
/// `Instant::now()` around the same op index, so they are HOST time and include
/// whatever the pool costs to wake. The two reports are deliberately the same
/// table, in the same order, computed by the same grouping, so a family that is
/// expensive on one backend and cheap on the other is visible as a row that
/// moved rather than as a number someone has to remember.
///
/// `per_op_ms` is the only difference from the GPU form - there is no launch
/// count to report, because a CPU op IS one call - and the table omits that
/// column rather than printing a column of ones.
#[cfg(feature = "dev")]
fn report_cpu_timeline(plan: &maxim::model::Plan, per_op_ms: &[f32], total_ms: f32) {
    use std::collections::HashMap;
    let mut by_kind: HashMap<String, (f32, usize)> = HashMap::new();
    for (i, op) in plan.ops.iter().enumerate() {
        // Same rule as the GPU report: `describe`'s first word is the family.
        let d = maxim::exec_cpu::describe(op);
        let kind = d.split_whitespace().next().unwrap_or("?").to_string();
        let e = by_kind.entry(kind).or_insert((0.0, 0));
        e.0 += per_op_ms[i];
        e.1 += 1;
    }
    let mut rows: Vec<_> = by_kind.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap_or(std::cmp::Ordering::Equal));
    println!("timeline: {:.2}s total, {} ops, cpu", total_ms / 1000.0, per_op_ms.len());
    println!("{:<10} {:>7} {:>9} {:>10}  {}", "op", "count", "total ms", "ms/op", "share");
    for (kind, (ms, count)) in rows {
        println!(
            "{:<10} {:>7} {:>9.1} {:>10.3}  {:>5.1}%",
            kind,
            count,
            ms,
            ms / count as f32,
            100.0 * ms / total_ms.max(1e-6)
        );
    }
    let mut by_shape: HashMap<String, (f32, usize)> = HashMap::new();
    for (i, op) in plan.ops.iter().enumerate() {
        let e = by_shape.entry(maxim::exec_cpu::describe(op)).or_insert((0.0, 0));
        e.0 += per_op_ms[i];
        e.1 += 1;
    }
    let mut rows2: Vec<_> = by_shape.into_iter().collect();
    rows2.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap_or(std::cmp::Ordering::Equal));
    println!("by shape (kind, then the shape or flags it describes):");
    let mut printed = std::collections::HashSet::new();
    for (k, (ms, n)) in rows2.iter().take(12) {
        println!("  {ms:9.1} ms {n:5}x  {k}");
        printed.insert(k.clone());
    }
    println!("  -- every other shape with 16 or more ops (aggregate, per-op):");
    let mut many: Vec<_> = rows2.iter().filter(|(k, (_, n))| *n >= 16 && !printed.contains(k.as_str())).collect();
    many.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap_or(std::cmp::Ordering::Equal));
    for (k, (ms, n)) in many {
        println!("  {ms:9.1} ms {n:5}x  {k}   ({:.3} ms/op)", ms / *n as f32);
    }
    let mut worst: Vec<(usize, f32)> = per_op_ms.iter().copied().enumerate().collect();
    worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("slowest ops:");
    for (i, ms) in worst.into_iter().take(5) {
        println!("  op {i} {:.2} ms  {}", ms, maxim::exec_cpu::describe(&plan.ops[i]));
    }
}

/// Where the GPU time went, grouped by op kind.
///
/// The per-op numbers come from a CUDA event around each op, so they are device
/// time and they sum to the run. What matters is the SHARE: the graph is 1840
/// ops, and one kernel family holding a third of the time is a different problem
/// from every op costing the same, which is what the total alone cannot say.
#[cfg(all(feature = "dev", feature = "cuda"))]
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
    // A census by op kind and shape: the aggregate share says WHICH family is
    // expensive, and this says which shapes inside it, which is what a kernel
    // decision turns on (a 64-wide tile versus a fallback, a 128-channel layer
    // versus a 32-channel one).
    let mut by_shape: std::collections::HashMap<String, (f32, usize)> = std::collections::HashMap::new();
    for (i, op) in plan.ops.iter().enumerate() {
        let e = by_shape.entry(maxim::exec_cpu::describe(op)).or_insert((0.0, 0));
        e.0 += tl.per_op_ms[i];
        e.1 += 1;
    }
    let mut rows2: Vec<_> = by_shape.into_iter().collect();
    rows2.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap_or(std::cmp::Ordering::Equal));
    // Print the 12 most expensive shapes AND every shape with enough ops that a
    // per-op launch cost could dominate it. The second half is the point: a shape
    // at 0.2 ms/op looks free next to a 4 ms conv, but 224 of them are 45 ms of
    // the run, and the top-12 cut alone hides them (this is how the 120 `copy`
    // ops, which dispatch NO kernel at all, stayed invisible for a whole session).
    println!("by shape (kind, then the shape or flags it describes):");
    let mut printed = std::collections::HashSet::new();
    for (k, (ms, n)) in rows2.iter().take(12) {
        println!("  {ms:9.1} ms {n:5}x  {k}");
        printed.insert(k.clone());
    }
    println!("  -- every other shape with 16 or more ops (aggregate, per-op):");
    let mut many: Vec<_> = rows2.iter().filter(|(k, (_, n))| *n >= 16 && !printed.contains(k.as_str())).collect();
    many.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap_or(std::cmp::Ordering::Equal));
    for (k, (ms, n)) in many {
        println!("  {ms:9.1} ms {n:5}x  {k}   ({:.3} ms/op)", ms / *n as f32);
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
/// order names the culprit op directly. The CPU backend's own reading of the
/// graph is independently known to match torch to ~1e-6 on every named tensor,
/// so the diff is against trustworthy numbers rather than against itself.
///
/// Both arenas are stepped in LOCKSTEP from the same input, so op i sees the
/// same preceding state on both sides; a mismatch therefore reports the op that
/// produced it, not an accumulation.
#[cfg(all(feature = "dev", feature = "cuda"))]
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

#[cfg(all(feature = "dev", not(feature = "cuda")))]
fn verify_gpu_ops(_host: &mut Host) -> Result<(), maxim::Error> {
    Err("--verify-gpu needs the `cuda` feature".into())
}
