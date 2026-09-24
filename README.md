# maxim

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).
The family also includes [rmbg-rs](https://github.com/jacobsparts/rmbg-rs),
[locate-anything-rs](https://github.com/jacobsparts/locate-anything-rs),
[realesrgan-rs](https://github.com/jacobsparts/realesrgan-rs) and
[lama-inpaint-rs](https://github.com/jacobsparts/lama-inpaint-rs); they
share the [lightgpu toolkit](https://github.com/jacobsparts/lightgpu).
[pixeldeck](https://github.com/jacobsparts/pixeldeck) is a local web app for
cleaning up product photos that drives all of these engines.

MAXIM (Multi-Axis MLP) image restoration as a single self-contained binary. No
Python at inference, no JAX, no PyTorch, no CUDA toolkit needed to run it.

```
maxim --model maxim-lol.safetensors -i dark.png -o bright.png
```

## Why this exists

The reference MAXIM is a Flax/JAX program: a JAX install, a CUDA-matched `jaxlib`,
a pickle-only `.npz` checkpoint, and a model definition spread across a config
dictionary, a module tree and an eval script. This is the same network written as
an engine:

* **One self-contained binary**: 1.92 MB with the CUDA backend (0.96 MB with it
  compiled out). The only direct dependencies are `png` and the toolkit below -
  no image framework, no BLAS, no BLAS-shaped wrapper.
* **The checkpoint is read, not executed.** `tools/convert.py` copies the
  `opt/target/*` leaves - the weights, about a quarter of the file, the rest is
  optimizer state - into the standard `.safetensors` container once. At run time
  they are memory-mapped and shape-checked against the architecture constants.
* **A CPU backend that is actually usable.** `--device cpu` is the same graph in
  plain Rust with no driver and no GPU, because the plan is one shared op list
  and the two executors are two implementations of it (see below).
* **Byte-level agreement with the reference.** On the LOL evaluation set the
  engine's output differs from the reference implementation's by a mean of
  0.0001 of an 8-bit level, and both score a mean PSNR of 23.466 against the
  ground truth - the published figure for this checkpoint is 23.43.

## Build

```sh
cargo build --release                 # both backends (needs nvcc for the GPU one)
cargo build --release --no-default-features   # CPU only: no nvcc, no CUDA headers
```

The GPU build embeds fatbins compiled by `nvcc` at build time and `dlopen`s
`libcuda.so.1` at run time, so nothing links against CUDA and the resulting binary
runs on a machine with no toolkit installed. A CPU-only build has no NVIDIA
dependency at all and defaults to `--device cpu`.

## Get the weights

The checkpoints are published by the MAXIM authors on Google Cloud Storage, one
directory per task and dataset -
[`ckpt/Enhancement/LOL`](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Enhancement/LOL)
for the one this engine defaults to (MAXIM-2S, PSNR 23.43). Download the `.npz`,
then convert it:

```sh
python3 tools/convert.py --npz checkpoint.npz --out ../models/maxim-lol.safetensors
```

The converter needs `numpy`; the engine needs neither `numpy` nor JAX. Weight
names keep their Flax form (`stage_0_encoder_block_0/Conv_0/kernel`) because that
is what the model code reads; the engine's loader is what knows a `kernel` under
a `ConvTranspose_0` is `[kh, kw, c_in, c_out]` and must be flipped, rather than
baking a transformation into the file where it cannot be seen or undone.

## Run

```sh
maxim --model maxim-lol.safetensors -i dark.png -o bright.png
maxim --model maxim-lol.safetensors -i dark.png -o bright.png --device cpu
maxim --model maxim-denoise.safetensors -i noisy.png -o clean.png --task denoising
```

| flag | meaning |
| --- | --- |
| `-m, --model` | converted `.safetensors` checkpoint |
| `-i, --input` / `-o, --output` | PNG in / PNG out |
| `--task` | `enhancement` (default), `denoising`, `deblurring`, `deraining`, `dehazing` - picks the variant the checkpoint was trained with |
| `--variant` | `S-1`..`S-3`, `M-1`..`M-3`, overriding `--task` |
| `--device` | `gpu` (default) or `cpu` |
| `--factor` | pad the input to a multiple of this (default 64, as the reference eval script does) |

`--dump <dir>` writes every named activation as `.npy` for
`tools/compare.py`, which is the parity tool described below.

The input is padded to a multiple of `--factor` and the result cropped back, so
the output has the input's dimensions. The padding is not cosmetic: with
`depth = 3` the network downsamples by 8 at every stage boundary and the gMLP
block layout requires the feature map to be a multiple of the 8-to-16-pixel block
size, so an unpadded input either fails to build or silently addresses a
sub-block feature map.

## How it is put together

The engine is organised around one artifact: a `Plan`, the straight-line sequence
of 1840 ops a *fixed input shape* runs. Building it

* resolves every parameter name against the checkpoint and reports the ones it
  did not use,
* transposes each kernel into the layout a lightgpu kernel wants,
* computes every buffer's size and its live range, and packs all 2089 of them
  into one arena, reusing storage as soon as a buffer's last reader has run.

Both backends then walk that same list. `--device cpu` and `--device gpu` cannot
drift apart by more than floating point, and a new op has to be added in three
places that the compiler checks against each other (the op enum, `exec_cpu` and
`exec_gpu`) rather than in two implementations that have to be kept in step by
hand.

That property is what makes the parity tooling cheap, and it is the reason this
engine could be validated at all:

* `--dump <dir>` writes 43 named activations; `tools/compare.py` diffs two dump
  directories. Running it against `tools/reference.py` (a PyTorch transcription
  of the same model, kept in the repo as a development tool) gives 43 of 43
  tensors matching, most above 100 dB and the worst at 121 dB, max abs 9.3e-6.
* `--verify-gpu` runs the plan **one op at a time on both backends** and compares
  each op's destination between them. It is the difference between "the whole
  model is wrong" and "op 384 is wrong": the two comparisons above say nothing
  about where a divergence starts, and the one bug this path had turned out to be
  a single kernel's channel stride - reported as op 384 of 1840, the first resize
  in the graph, by a tool that named it in one run. It reports 1840 ops, 0
  disagreements beyond 1e-3 relative, worst 6.4e-5 - i.e. f32 accumulation order.
* `MAXIM_TRACE=1` prints one line per op with its operand labels, and
  `MAXIM_DEEP_DUMPS=1` (on the engine and on `reference.py`) turns on ~750
  per-block-internal activations, which localise a divergence to a single line of
  a block. They are behind a flag because a named buffer has to stay live to the
  end of the run for the dump to see it, which costs a large multiple of the
  arena.

## What makes it fast: lightgpu

Both backends sit on [lightgpu](https://github.com/jacobsparts/lightgpu), a small
library of hand-written CUDA kernels plus the driver plumbing to launch them -
not a wrapper around cuDNN or cuBLAS. MAXIM calls the toolkit's `lg_add`,
`lg_channel_layer_norm`, `lg_gelu_erf` and `lg_lrelu` and ships the rest of what
it needs in `cuda/maxim.cu`: the padded stride-2 4x4 conv and its transposed twin,
the space/block permutation, the two-pass bilinear resize, the channel mean and
scale, and a handful of elementwise ops. An `Op::Copy` is a device-to-device
memcpy rather than a launch, which is faster for the row copies the model does; the
toolkit's `lg_copy` is therefore compiled out, and the lists name only what an op can
launch. The two toolkit kernels the tiled forms replaced are the exception: every
kernel in the list is still linked and still launchable under `--legacy-ops`, because
a before-and-after from one binary is worth more than a few kilobytes.

Three op families are the whole cost of this model, and all three were written the
same naive way: **one thread per output element**, which re-reads every operand for
every output. Each now has a tiled kernel that keeps the reuse in registers and
shared memory, and each original is still linked, so the difference is measured
rather than asserted (`--legacy-ops` runs the originals):

| op | original | tiled | at 512x512 |
| --- | --- | --- | --- |
| 1x1 conv (444 ops) | `lg_conv1x1` | `mx_conv1x1_t` | 5.39 -> 1.56 ms/op |
| gating matmul (112 ops) | `mx_gate_mm` | `mx_gate_mm_t0`/`_t1` | 18.8 -> 0.89 ms/op |
| 3x3 conv (66 ops) | `lg_conv3x3s1p1` | `mx_conv3x3_t4`/`_t2` | 28.8 -> 3.42 ms/op |

The 1x1 conv put one thread on each output and walked its input column one
`plane`-strided float at a time, so it moved about `c_in` times the traffic the op
needs and ran at 199 GFLOP/s of this card's 8.9 TFLOP/s. Four output channels per
thread, over a transposed `[c_in][c_out]` weight, makes one input load feed four
outputs and brings it to the memory roofline instead - which is where it now sits
(288 GB/s of ~320 GB/s available). An ablation settles where the gain came from: the
same kernel with one channel per thread measures 5.35 ms/op, i.e. all of the 3.4x
is the tiling, none of it the transposed layout. That layout is still what makes a
multi-channel inner loop possible, and the GPU plan carries the transposed twin of
every 1x1 weight for it (74.1 MiB of weights rather than 54.1).

The gating matmul is an implicit GEMM: both of the gMLP's modes reduce one spatial
axis of a `[c][outer][inner]` tensor, so the same body covers them, with the tile
FILLED differently per mode and read through one index helper. A 64x64 output tile
with a 16-deep K chunk means one activation load feeds four outputs and one weight
load four more, in place of one of each per output.

The 3x3 conv tiles a 64-pixel segment of one row plus a one-column halo per input
channel, so a 3x3 window is three offsets into the same shared tile: about 47 MACs
per load instead of 1. Its accumulation order (`dy`, `dx`, `ci`) is the original's,
which is why the tiled form agrees with the CPU more closely than the original did
(worst relative disagreement 6.4e-5, down from 8.1e-5).

Kernels are compiled per consumer: `build.rs` gives `nvcc` an explicit `--entries`
list for each fatbin, so the binary embeds only the kernels this engine can call.
A name missing from that list is pruned from the fatbin and then fails at launch
rather than at build time, so the build script checks the list and the source
against each other in both directions - a typo fails the build, and so does a
kernel defined but not listed.

The graph is still 1840 small ops rather than a handful of large GEMMs, and with the
three families tiled the remaining time is spread thin rather than concentrated: at
640x448 the largest share is the 3x3 conv at 30%, the 1x1 conv is 24% and every
elementwise op in the graph together is 3.5%. `--profile` prints that breakdown,
per op and per family, from CUDA events around each op. The arena is one allocation,
so the 2089 buffers cost no allocation traffic at all.

## Performance

Measured on a GTX 1080 (sm_61) and an i7-13700K, for one 600x400 image from the LOL
eval set (padded to 640x448, 1840 ops):

| | time | per op | arena |
| --- | --- | --- | --- |
| `--device gpu` | 3.6 s | 2.0 ms | 942.7 MiB |
| `--device gpu --legacy-ops` | 9.0 s | 4.9 ms | 942.7 MiB |
| `--device cpu` | 155 s | 84 ms | 942.7 MiB |

The legacy row is the same binary with the kernels as they were before the tiling
work in the previous section. That is the honest way to state a 2.5x: both numbers
come from one build, so neither has to be taken on trust.

The GPU figure is steady-state. The card parks at 139 MHz between runs and the first
run after an idle period takes about 20 s while it ramps to its 1835 MHz boost
clock, so a single measurement of a run this short has to be read with that in mind.
The CPU figure is one core at 99% CPU - the CPU executor is serial, one op at a
time, and it walks the same op list rather than a fused graph, so it pays both the
per-op dispatch and a cold 942 MiB arena.

For scale, `tools/reference.py` - a PyTorch transcription of the same model, kept in
the repo as a development tool, and pure CPU, with no `.cuda()` in it - takes 7.7 s
wall at 1443% CPU (16 threads) on the same image. The engine and the reference are
now in the same range, arriving from opposite directions: torch wins by fusing and
batching 1840 ops into a few large kernels, and this engine pays a launch per op but
has kernels tiled for its own shapes.

Single-image time is close to linear in the input's area:

| input | `--device gpu` | with `--legacy-ops` |
| --- | --- | --- |
| 128x128 | 0.27 s | 0.43 s |
| 256x256 | 0.64 s | 1.83 s |
| 384x384 | 1.64 s | 4.58 s |
| 512x512 | 2.60 s | 8.38 s |

The arena is a function of the padded input size, not of the model: a 128x128 crop
needs 53.9 MiB, and 600x400 is the largest size the eval set contains. The feature
map is not tiled, so the whole of it lives in the arena and a large image needs a
card that can hold it. The weights are 74.1 MiB on the GPU plan and 54.1 MiB on the
CPU-only one, the difference being the transposed 1x1 weights that only the GPU
kernel reads.

## Accuracy

The engine is checked in three ways, each of which catches something the others
cannot.

**Parity with the reference, per activation.** `tools/compare.py` between the
engine's dump and `reference.py`'s, on two input sizes, on both backends:

| input | tensors | worst max abs | worst PSNR | backend agreement (`--verify-gpu`) |
| --- | --- | --- | --- | --- |
| 128x128 crop | 43 of 43, 0 failed | 9.3e-6 | 121 dB | 1840 ops, 0 beyond 1e-3 relative, worst 6.4e-5 |
| 256x256 crop | 43 of 43, 0 failed | 1.2e-5 | 118 dB | 1840 ops, 0 beyond 1e-3 relative, worst 3.0e-4 |
| 640x448 (a real eval image) | - | - | - | 1840 ops, 0 beyond 1e-3 relative, worst 1.5e-4 |

The 640x448 row is there because it is the size the model actually runs on, and
because the square powers of two above hid a real bug: the tiled gating matmul's
first version loaded its tile with `float4`s, which is only legal where the
address is 16-byte aligned - true for every square size, false for the grid cell
counts a 640x448 image produces, where it faulted at launch. It is now scalar and
guarded, and the fills are a few percent of that kernel's work.

Both dump sizes are 43 of 43 with nothing below 100 dB. The backend disagreement grows
with tensor size - 6.4e-5 at 128x128, 3.0e-4 at 256x256 - which is what f32
accumulation order does when there are more terms to sum in a different sequence,
not a widening algorithmic gap: the bar is 1e-3 relative to each buffer's own
magnitude, and no size comes close.

**Parity is not correctness.** Both backends walk the same op list, so a mistake
in the plan - a transposed weight, a swapped axis, a block that is never applied -
is *consistent* between them and invisible to the comparison above. That is what
the eval set is for: it measures the output against the ground truth, so the plan
has to be the right graph and not merely a self-consistent one.

```
python3 tools/eval.py                     # all 15 images, engine only
python3 tools/eval.py --reference         # also runs tools/reference.py on the same images
```

On all 15 images of the LOL eval set:

| | mean PSNR | per-image engine vs reference |
| --- | --- | --- |
| engine | **23.466** | mean abs difference 0.0000-0.0002 of a level |
| `tools/reference.py` | 23.466 | |
| published (this checkpoint) | 23.43 | |

Every per-image PSNR agrees to three decimals, and the two images that are 13.7
and 15.6 dB (the reference fails on them too) fail identically. A difference in
the *mean* points at an engine bug; agreement with a mean that missed 23.43 would
point at the checkpoint or at the preprocessing instead. `tools/eval.py` returns
non-zero on either.

**Known infidelities**, all far below 8-bit output precision and deliberately not
chased:

* The 1x1 conv accumulates over input channels where the reference's implicit GEMM
  accumulates in its own blocked order, and the gating matmul's per-element order
  differs from an einsum's. These show up as 1e-6-level differences, not as a PSNR
  gap. Tiling a kernel also changes the order the terms are summed in - the transposed
  1x1 kernel keeps four running sums where the original kept one - which is exactly
  what the 1e-3 relative bar exists to absorb. The 3x3 conv's order (`dy`, `dx`, `ci`)
  was deliberately preserved, and its agreement improved as a result.
* The multi-scale input pyramid uses jax's nearest-neighbour rule - output `i`
  takes `floor((i + 0.5) * m / n)`, so a 2x downsample keeps the ODD rows - where
  torch's `interpolate(mode="nearest")` keeps the even ones. The engine implements
  jax's rule, since jax is what trained the weights.

## Layout

```
src/model.rs     the graph: one method per block of the reference, plus the packer
src/exec_cpu.rs  the CPU executor and its shared-memory kernels
src/exec_gpu.rs  the GPU executor: one launch per op
src/config.rs    the variant table and the task -> variant mapping
src/weights.rs   safetensors reader and the kernel-layout transposes
src/image.rs     PNG in/out, padding and the crop back
src/host.rs      the host-side arena, weights and named activations
src/main.rs      the CLI, --dump, --profile, --legacy-ops and --verify-gpu
cuda/maxim.cu    this engine's own kernels (sixteen, incl. the tiled forms)
tools/reference.py  a PyTorch transcription of the same model (the reference)
tools/compare.py    diff two --dump directories
tools/eval.py       PSNR against the LOL ground truth, optionally vs the reference
tools/convert.py    Flax .npz -> .safetensors
tests/packing.rs    the arena packer's invariants
tests/end_to_end.rs the model restores an image, and both backends agree on it
```

`tests/packing.rs` is the odd one out and worth keeping: it asserts that the same
graph built with and without named activations produces **bit-identical** output,
and that no op's destination overlaps a source it reads. Both are properties of
the packer that fail silently - one produced a spectacular false alarm where every
activation looked wrong while the arithmetic was provably right - and neither can
be seen in an output comparison, because the bug changes the *plan*, not a kernel.

`tests/end_to_end.rs` covers the other blind spot. Every parity check above can
pass while the model is wrong in a way both backends agree on, so it asserts the
thing a user cares about instead: a low-light input at under 0.2 mean level comes
out more than 0.2 brighter and not saturated. It also asserts the CPU and GPU
outputs agree, which is the only check that would catch a backend that is right on
every op and wrong overall. Both tests skip rather than fail without the weights
and the dataset, so a fresh clone can still run the suite.
