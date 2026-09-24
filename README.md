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

* **One self-contained binary**: 1.52 MB with the CUDA backend (0.96 MB with it
  compiled out). The direct dependencies are `png`, `rayon` and `libc`, plus the
  toolkit below.
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
  tensors matching, most above 100 dB and the worst at 121 dB, max abs 9.1e-6.
* `--verify-gpu` runs the plan **one op at a time on both backends** and compares
  each op's destination between them. It is the difference between "the GPU plot
  is wrong" and "op 384 is wrong": the whole-model diff says nothing about where
  the disagreement starts, and a planted bug that cost 14 dB turned out to be a
  single kernel's stride. It reports 1840 ops, 0 disagreements beyond 1e-3
  relative, worst 8.1e-5 - i.e. f32 accumulation order.
* `MAXIM_TRACE=1` prints one line per op with its operand labels, and
  `MAXIM_DEEP_DUMPS=1` (on the engine and on `reference.py`) turns on ~750
  per-block-internal activations, which localise a divergence to a single line of
  a block. They are behind a flag because a named buffer has to stay live to the
  end of the run for the dump to see it, which costs a large multiple of the
  arena.

## What makes it fast: lightgpu

Both backends sit on [lightgpu](https://github.com/jacobsparts/lightgpu), a small
library of hand-written CUDA kernels plus the driver plumbing to launch them -
not a wrapper around cuDNN or cuBLAS. MAXIM calls six of the toolkit's kernels
(`lg_conv1x1`, `lg_conv3x3s1p1`, `lg_add`, `lg_channel_layer_norm`,
`lg_gelu_erf`, `lg_lrelu`) and ships the eleven it needs that the toolkit does
not have in `cuda/maxim.cu`: the padded stride-2 4x4 conv and its transposed twin,
the blocked-layout gating matmul, the space/block permutation, the two-pass
bilinear resize, the channel mean and scale, and a handful of elementwise ops.
An `Op::Copy` is a device-to-device memcpy rather than a launch, which is faster
for the row copies the model does; the toolkit's `lg_copy` is therefore compiled
out. A kernel with no caller is bytes in the binary, so the lists name only what
an op actually launches.

Kernels are compiled per consumer: `build.rs` gives `nvcc` an explicit `--entries`
list for each fatbin, so the binary embeds only the kernels this engine can call.
A name missing from that list is pruned from the fatbin and then fails at launch
rather than at build time, so the build script checks the list and the source
against each other in both directions - a typo fails the build, and so does a
kernel defined but not listed.

The graph is not the bottleneck for this model, the kernel count is: MAXIM is
1840 small ops rather than a handful of large GEMMs, and at 5 ms per op the run is
dominated by the launch/sync path. The arena is one allocation, so the 2089
buffers cost no allocation traffic at all.

## Performance

Measured on a GTX 1080 (sm_61) and 24 CPU threads, for one 600x400 image from the
LOL eval set (padded to 640x448, 1840 ops):

| | time | per op | arena |
| --- | --- | --- | --- |
| `--device gpu` | 9.4 s | 5 ms | 942.7 MiB |
| `--device cpu` | 160 s | 87 ms | 942.7 MiB |

The GPU figure is steady-state: the card parks at 139 MHz between runs, and the
first run after an idle period takes about 20 s while it ramps to its 1835 MHz
boost clock. Take any single measurement of a run this short with that in mind.

The arena is a function of the padded input size, not of the model: a 128x128 crop
needs 53.9 MiB, and 600x400 is the largest size the eval set contains. There is
no tiling yet; the whole feature map lives in the arena, so a large image needs a
card that can hold it. The weights are 54.1 MiB on either backend.

## Accuracy

The engine is checked in three ways, each of which catches something the others
cannot.

**Parity with the reference, per activation.** `tools/compare.py` between the
engine's dump and `reference.py`'s, on two input sizes, on both backends:

| input | tensors | worst max abs | worst PSNR | backend agreement (`--verify-gpu`) |
| --- | --- | --- | --- | --- |
| 128x128 crop | 43 of 43, 0 failed | 9.1e-6 | 121 dB | 1840 ops, 0 beyond 1e-3 relative, worst 8.1e-5 |
| 256x256 crop | 43 of 43, 0 failed | 1.1e-5 | 118 dB | 1840 ops, 0 beyond 1e-3 relative, worst 3.0e-4 |

Both sizes are 43 of 43 with nothing below 100 dB. The backend disagreement grows
with tensor size - 8.1e-5 at 128x128, 3.0e-4 at 256x256 - which is what f32
accumulation order does when there are more terms to sum in a different sequence,
not a widening algorithmic gap: the bar is 1e-3 relative to each buffer's own
magnitude, and neither size comes close.

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

* `lg_conv1x1` and `lg_conv3x3s1p1` accumulate in `(oc, ic)` order where the
  reference's implicit GEMM accumulates in its own blocked order, and
  `mx_gate_mm`'s per-element order differs from an einsum's. These show up as
  1e-6-level differences, not as a PSNR gap.
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
src/main.rs      the CLI, --dump, --profile and --verify-gpu
cuda/maxim.cu    this engine's eleven kernels
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
