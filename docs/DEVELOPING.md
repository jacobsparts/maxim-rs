# Developing

Everything here is for someone editing a kernel or checking a change to the
engine. None of it is needed to run it, and none of the flags below exist in a
release binary - see Development flags.

## The shape of the engine

One artifact is central: a **plan**. Building it resolves every parameter name,
transposes each kernel into the layout a lightgpu kernel wants, and records every
buffer's size and live range in ONE arena. Both backends then walk that same list
of ops, so `--device cpu` and `--device gpu` cannot drift apart by more than
floating point, and a new op is added in three places that the compiler checks
against each other: the `Op` enum, `exec_cpu`, `exec_gpu`.

The arena is a single allocation and aliasing buffers are just overlapping
offsets, so a split or a concatenation costs nothing and 2089 buffers cost no
allocation traffic. A GPU run allocates only enough host arena to hold the input
unless a dump or the verify walk needs the whole thing (`Host::with_arena_len`).

## Development flags

A release binary has none of these, and refuses them by name rather than ignoring
them - a script that asked for a dump must not carry on as if it had one. Build
one with:

```sh
cargo build --release --features dev
```

* `--dump <dir>` writes every named activation as `.npy`; `tools/compare.py`
  diffs two dump directories.
* `--profile` prints the per-op census: by op kind, by shape, and the slowest
  ops. On the GPU the numbers are device time from a CUDA event around each op;
  on the CPU they are host time around the same op index.
* `--verify-gpu` runs the plan one op at a time on BOTH backends from the same
  input and reports every op whose destination disagrees beyond 1e-3 relative.
  "The plot is wrong somewhere" is not a diagnostic; an op index is.
* `--factor <n>` pads to a multiple of `n` instead of 64.

Environment hooks, read only by a `dev` build: `MAXIM_TRACE` (weight blobs and
the dump order), `MAXIM_SERIAL_UPTO` / `MAXIM_SERIAL_FROM` (bisect a divergence
by forcing the pool on or off around an op index), `MAXIM_TIMELINE`, `MAXIM_TIME`
(GPU phase timings), `MAXIM_POISON` (fill the arena with `0xCD` before the run,
which is how the arena zeroing that used to cost 0.4 s was proved unnecessary),
`MAXIM_DEEP_DUMPS` (also name the per-block-internal activations).

## Checking a change

The engine's output is compared against goldens it did not produce, and the
hashes are the check: run the three size below and compare against these strings.
A change to a CPU kernel that moves them is wrong even if the picture looks right.

| input | golden MD5 of the CPU output |
|---|---|
| 96x96 gradient | `fd22ee7612d63b00a54fd375d68688ab` |
| 600x400 LOL eval frame | `9cb65100ad661e20690a334d59a89a4d` |

The GPU output is a different hash on purpose (the kernels accumulate in a
different order) and is not golden; the backends are held to 1e-3 relative by
`--verify-gpu` and to the same picture by `cargo test --release`.

`tools/reference.py` is a PyTorch transcription of the same model, kept as a
development tool. `tools/eval.py` runs a whole evaluation set through both and
reports PSNR against the ground truth.

## Where the time goes

At 600x400, the CPU census is dominated by three op families, and all three were
written the same naive way - one thread per output element - before their kernels
were rewritten to tile over the output and sweep the sources sequentially:

| family | share of the CPU run |
|---|---|
| gated matmul (`gatemm`) | 28% |
| 3x3 conv | 22% |
| 1x1 conv | 20% |
| everything else | 30% |

The CPU path is the no-GPU fallback and is held to the same standard as the GPU,
so it is optimised rather than left slow. What is closed, and should not be
retried: the 3x3 conv is against the scalar-FP ceiling (206-216 GFLOP/s over 24
threads) and LLVM already emits packed operations for its inner loop, so hand
vectorisation is a 0.97-0.99x no-op; ISA flags do nothing (`target-cpu=x86-64-v3`
is SLOWER); DRAM saturates at about 60 GB/s with only four cores, so the plan's
memory traffic is not the limit; and a single-thread microbenchmark is NOT
predictive at 24 threads - three separate attempts were made on that basis and
all three lost in situ.

Measure with `--profile` in situ rather than with an isolated kernel: the machine
this was tuned on carries other load, and a census that attributes time to an op
index cannot lie about what the run spent.

## Accuracy, in detail

Parity is not correctness. Both backends walk the same op list, so a mistake in
the plan - a transposed weight, a swapped axis, a block never applied - is
consistent between them and invisible to a backend comparison. Hence three
separate checks, each catching something the others cannot.

**Parity with the reference, per activation.** `tools/compare.py` between the
engine's dump and `reference.py`'s, on both backends:

| input | tensors | worst max abs | worst PSNR | backend agreement (`--verify-gpu`) |
| --- | --- | --- | --- | --- |
| 128x128 crop | 43 of 43, 0 failed | 9.3e-6 | 121 dB | 1840 ops, 0 beyond 1e-3 relative, worst 6.4e-5 |
| 256x256 crop | 43 of 43, 0 failed | 1.2e-5 | 118 dB | 1840 ops, 0 beyond 1e-3 relative, worst 3.0e-4 |
| 640x448 (a real eval image) | - | - | - | 1840 ops, 0 beyond 1e-3 relative, worst 1.5e-4 |

The 640x448 row is there because it is the size the model actually runs on, and
because the square powers of two above hid a real bug: the tiled gating matmul's
first version loaded its tile with `float4`s, which is only legal where the
address is 16-byte aligned - true for every square size, false for the grid cell
counts a 640x448 image produces, where it faulted at launch.

The backend disagreement grows with tensor size, which is what f32 accumulation
order does when there are more terms to sum in a different sequence, not a
widening algorithmic gap: the bar is 1e-3 relative to each buffer's own
magnitude, and no size comes close.

**Correctness, not just self-consistency.** `tools/eval.py` measures against the
ground truth, so the plan has to be the right graph. On all 15 LOL images the
engine scores a mean PSNR of 23.466 and `reference.py` 23.466; the published
figure for the checkpoint is 23.43. A difference in the mean points at an engine
bug, while agreement with a mean that missed 23.43 would point at the checkpoint
or the preprocessing instead. `tools/eval.py` returns non-zero on either.

**Known infidelities**, all far below 8-bit output precision and deliberately not
chased:

* The 1x1 conv accumulates over input channels where the reference's implicit
  GEMM accumulates in its own blocked order, and the gating matmul's per-element
  order differs from an einsum's. Tiling a kernel also changes the order the
  terms are summed in - the transposed 1x1 kernel keeps four running sums where
  the original kept one - which is what the 1e-3 relative bar exists to absorb.
  The 3x3 conv's order (`dy`, `dx`, `ci`) was deliberately preserved.
* The multi-scale input pyramid uses jax's nearest-neighbour rule - output `i`
  takes `floor((i + 0.5) * m / n)`, so a 2x downsample keeps the ODD rows - where
  torch's `interpolate(mode="nearest")` keeps the even ones. The engine
  implements jax's rule, since jax trained the weights.

## Layout

```
src/model.rs     the graph: one method per block of the reference, plus the packer
src/exec_cpu.rs  the CPU executor and its tiled kernels
src/exec_gpu.rs  the GPU executor: one launch per op
src/config.rs    the variant table and the checkpoint -> variant derivation
src/weights.rs   safetensors reader and the kernel-layout transposes
src/image.rs     PNG I/O, the reference padding and the crop back
cuda/maxim.cu    this engine's kernel family
tools/           conversion, the PyTorch reference, the diff and the eval set
```

## The plan is packed

Buffers are allocated by live range, not by name: a buffer's slot is reused once
its previous occupant is dead. `tests/packing.rs` checks that the packed plan is
the size it claims, that no op overlaps its own inputs, and that recording a dump
(the only thing that changes what is live to the end of the run) does not change
the result.

## Parity tooling

```
tools/convert.py    Flax .npz -> .safetensors (--task labels the header)
tools/reference.py  PyTorch transcription of the model, --dump/--trace/--seed
tools/compare.py    diff two dump directories, --tol, --all
tools/eval.py       a whole eval set through both, PSNR against ground truth
```

`tools/compare.py` against a `reference.py` dump gives 43 of 43 named tensors
matching, worst max abs 9.3e-6.

## Build variants

| command | what it is |
|---|---|
| `cargo build --release` | CPU + CUDA, one binary |
| `cargo build --release --no-default-features` | CPU only |
| `cargo build --release --features dev` | CPU + CUDA and the development flags |

`cargo build --release --no-default-features` OVERWRITES `target/release/maxim`,
so copy the GPU binary aside before building it if you need both.
