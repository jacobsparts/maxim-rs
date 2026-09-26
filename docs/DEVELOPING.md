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
hashes are the check: run the sizes below and compare against these strings.
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

## Sizes and memory

The whole feature map is resident at once on whichever backend runs, so the
footprint follows the input and the stage count and is known before anything
runs - it is a plan total, and every run prints it:

| input | variant | plan | arena | weights |
|---|---|---|---|---|
| 256x256 | S-2 | 1840 ops, 2089 buffers | 215.5 MiB | 86.7 MiB |
| 256x256 | S-3 | 2885 ops, 3276 buffers | 291.0 MiB | 135.2 MiB |
| 640x448 (600x400 padded) | S-2 | 1840 ops | 942.7 MiB | 86.7 MiB |
| 704x768 (669x760 padded) | S-3 | 2885 ops | 2400.6 MiB | 135.2 MiB |

`weights` is the resident bytes after the loader's layout transposes, which is
why it is larger than the file: a checkpoint is 54.1 MiB (S-2) or 84.7 MiB (S-3)
of tensor data, and the transposed copies of the convolutions are what the
kernels read. There is no tiling mode to trade memory for time, so a large image
needs a correspondingly large card; the CPU path allocates the same arena on the
host, and it is only the GPU's copy that a plain GPU run can skip.

The input is padded up to a multiple of 64 and the result cropped back, so the
output has the input's dimensions. That is what the upstream evaluation script
does, and it is not cosmetic: each stage downsamples, and the gMLP blocks need
the feature map to be a multiple of their 8-to-16 pixel block size.

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

## Performance

MAXIM-LOL, 600x400 input padded to 640x448, on a GTX 1080 (Pascal, sm_61) with an
i7-13700K (24 hardware threads):

| | this engine | PyTorch reference |
|---|---|---|
| GPU | **0.72 s** | - |
| CPU | **5.8 s**, 50.4 s single-threaded | 7.7 s wall at 16 threads |

The whole CPU path was taken from 24.37 s to 5.8 s over twelve landings. The GPU
figure is steady-state: the card parks at 139 MHz between runs, so the first run
after an idle period takes about 20 s while it ramps to its boost clock, which is
worth knowing before believing a cold measurement. The CPU figures move with
machine load, and this machine carries a lot of it.

The CPU path is the no-GPU fallback and is held to the same standard as the GPU,
not treated as a slow correctness check: it walks the same plan, and each op's
own channels go across a rayon pool.

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

**Against the authors' own outputs**, which is the check neither of the two
above is. Upstream publishes the PNGs their JAX implementation produced, next to
a per-image PSNR table, so the engine can be compared with the thing it is a
reimplementation of:

| checkpoint | set | images | engine mean PSNR | their mean PSNR | theirs, +0.5 LSB | engine vs their PNGs |
| --- | --- | --- | --- | --- | --- | --- |
| LOL (S-2, enhancement) | eval15 | 15 | 23.4664 | 23.4346 | 23.4868 | 47.9-53.3 dB |
| RESIDE-Indoor (S-2, dehazing) | the whole test set | 500 | 37.9286 | 38.1133 | 37.9721 | 50.7-51.6 dB |
| RealBlur-R (S-3, deblurring) | the whole test set | 980 | 37.3867 | 37.1131 | 37.3702 | - |

The third column is the point of the table. `their mean PSNR` is what the authors'
own PNGs score against the ground truth, and it reproduces their published
per-image tables exactly - 23.4346 for LOL, 38.1133 for RESIDE-Indoor. But those
PNGs are floored, so they are on average half a level low, and `theirs, +0.5 LSB`
adds that half level back before scoring. That column is the one to compare with:
the engine is within 0.02-0.04 dB of it on all three, which is what a faithful
reimplementation of the same graph and the same weights should be. Comparing the
raw column instead would charge the engine for a rounding choice made in
upstream's save function.

Why the correction is needed at all: both of their scripts write with
`(np.clip(x, 0., 1.) * 255.).astype(uint8)`, which TRUNCATES, while this engine
(and `reference.py`) round half up. Roughly half the pixels therefore differ by
one 8-bit level, which makes the two pictures a ~50 dB match rather than an
identical one. Rounding is kept: half a level on average is far below the model's
own error, and moving to truncation would change every golden hash for no gain.

The S-3 path is covered by the RealBlur-R row, which is also the largest check
here: 980 images, all of them, against the authors' numbers for the same set. The
five S-3 checkpoints (denoising and deblurring) share a graph, so that row is
about the architecture rather than about one file.

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
