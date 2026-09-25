# maxim-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).
The family also includes [rmbg-rs](https://github.com/jacobsparts/rmbg-rs),
[realesrgan-rs](https://github.com/jacobsparts/realesrgan-rs),
[nafnet-rs](https://github.com/jacobsparts/nafnet-rs),
[lama-inpaint-rs](https://github.com/jacobsparts/lama-inpaint-rs) and
[locate-anything-rs](https://github.com/jacobsparts/locate-anything-rs), all
built on the [lightgpu toolkit](https://github.com/jacobsparts/lightgpu);
[pixeldeck](https://github.com/jacobsparts/pixeldeck) is a local web app for
cleaning up product photos that drives them all.

[MAXIM](https://github.com/google-research/maxim) (Multi-Axis MLP) image
restoration as a single self-contained binary - low-light enhancement, denoising,
deblurring, deraining and dehazing. Feed it a PNG, get back a restored PNG. No
Python, JAX, PyTorch, ONNX Runtime or CUDA toolkit needed at runtime.

```
maxim -m maxim-lol.safetensors -i dark.png -o bright.png
```

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. The GPU is used when a CUDA driver is available and the
  CPU path otherwise, so one binary covers a machine with no NVIDIA driver at
  all; `--device cpu|gpu` overrides that choice.
* 2.90 MiB binary, statically linked except `libc` and `libgcc_s`.
  `libcuda.so.1` is `dlopen`ed, so no driver is required on disk. (The CPU-only
  build is 1.30 MiB.)
* The checkpoint is data, not code: `tools/convert.py` copies the weights out of
  the Flax `.npz` into the standard `.safetensors` container once, and at run
  time they are memory-mapped. The converted file also carries its own
  architecture in its header, so there is no variant flag to get wrong.
* **Both backends are faster than the PyTorch reference** on the machine this was
  built on, and the reference is only reachable at all through a full JAX or
  PyTorch install.

## Download

Prebuilt binaries and the converted checkpoint are attached to the
[releases](https://github.com/jacobsparts/maxim-rs/releases). Both binaries run
on the CPU; they differ only in whether CUDA support is compiled in.

| asset | contents | notes |
|---|---|---|
| `maxim-linux-x86_64` | CPU + CUDA, auto-selected | x86-64 Linux with glibc ≥ 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); falls back to the CPU path when no NVIDIA driver is present. GPU path needs a compute capability 6.1+ GPU |
| `maxim-linux-x86_64-cpu-only` | CPU only | same, with nothing NVIDIA-related included - `--device gpu` is refused |
| `maxim-lol.safetensors` | the enhancer | see Choosing a checkpoint |

```sh
./maxim-linux-x86_64 -m maxim-lol.safetensors -i dark.png -o bright.png
```

## Build

```sh
cargo build --release
# CPU only, no CUDA toolkit or driver needed at build time either:
cargo build --release --no-default-features
```

The default build needs `nvcc` (set `NVCC=` if it is not on `PATH`) and produces
one binary with both backends. The `--no-default-features` build contains only
the CPU path, which reports `this build has no cuda feature; use --device cpu` if
asked for the GPU rather than failing obscurely. The kernels cover `sm_61`,
`sm_75`, `sm_80` and compute capability 8.0 PTX, so the GPU path runs on Pascal
(GTX 10-series) through Ampere, and on anything newer via the PTX.

`lightgpu` is a normal Cargo dependency on
[its repository](https://github.com/jacobsparts/lightgpu), so a clone of this
project builds on its own.

## Choosing a checkpoint

Which model you run is decided by the file you pass to `-m`. MAXIM was trained
separately for each task, so the weights are not interchangeable: an enhancement
model run on a noisy frame, or a denoising model on an underexposed one, is off
its training distribution and can make the picture worse rather than better.

The released file is the **enhancement** model trained on **LOL** (low-light),
the published MAXIM-2S configuration - 32 features, 2 stages, 14.2 M parameters.
Its own header records what it is, so the engine needs no flag to know how to
read it.

To convert another checkpoint from the official releases, download the Flax
`.npz` and run:

```sh
python3 tools/convert.py --npz checkpoint.npz --task denoising --out ../models/maxim-denoise.safetensors
```

The converter needs `numpy`; the engine needs neither `numpy` nor JAX. The
`--task` is only a label written into the converted header - the architecture
(variant) is derived from the weights themselves. Weight names keep their Flax
form (`stage_0_encoder_block_0/Conv_0/kernel`) because that is what the model code
reads; the engine's loader is what knows a `kernel` under a `ConvTranspose_0` is
`[kh, kw, c_in, c_out]` and must be flipped, rather than baking a transformation
into the file where it cannot be seen or undone.

## Usage

```sh
maxim -m maxim-lol.safetensors -i dark.png -o bright.png
maxim -m maxim-lol.safetensors -i dark.png -o bright.png --device cpu
# pipeline use - both streams default to stdin/stdout, and `-` names them too
cat dark.png | maxim -m maxim-lol.safetensors > bright.png
```

```
-m, --model <path>    converted .safetensors checkpoint (see tools/convert.py)
-i, --input <path>    input PNG, or - for stdin (default: stdin)
-o, --output <path>   output PNG, or - for stdout (default: stdout)
    --device <dev>    gpu or cpu (default: gpu when the CUDA driver can be
                      brought up, cpu otherwise; a CPU-only build is always cpu)
    --cpu             same as --device cpu
    --gpu             same as --device gpu, and refuses to fall back
-q, --quiet           no progress output
-h, --help            this text
-V, --version         print the version
```

Nothing else is in a release binary.

The input is padded up to a multiple of 64 and the result cropped back, so the
output has the input's dimensions. That padding is what the upstream evaluation
script does, and it is not cosmetic: each stage downsamples, and the gMLP blocks
need the feature map to be a multiple of their 8-to-16 pixel block size.

## Large images

The whole feature map is resident on the device at once, so VRAM grows with the
input: a 600x400 image needs a 942.7 MiB arena, and the requirement is dominated
by the largest stage. It is a plan total and it is predictable, not a matter of
luck - but it also means a very large image needs a correspondingly large card,
and there is no tiling mode to trade memory for time.

## Performance

MAXIM-LOL (the released enhancer), 600x400 input padded to 640x448, on a
GTX 1080 (Pascal, sm_61) with an i7-13700K (24 hardware threads):

| | this engine | PyTorch reference |
|---|---|---|
| GPU | **0.72 s** | - |
| CPU | **5.8 s**, 50.4 s single-threaded | 7.7 s wall at 16 threads |

The GPU figure is steady-state; the card parks at 139 MHz between runs, so the
first run after an idle period takes about 20 s while it ramps to its boost
clock. The CPU figures move with machine load.

The CPU path is the fallback for a machine with no GPU, and it is held to the
same standard rather than treated as a slow correctness check: it walks the same
1840-op plan, and each op's own channels go across a rayon pool.

## Accuracy

Both backends walk one op list, so they cannot drift apart by more than floating
point, and the engine is checked against numbers it did not produce:

* On all 15 images of the LOL evaluation set (`tools/eval.py`) the owner's output
  scores a mean PSNR of **23.466** against the ground truth, and the PyTorch
  reference scores 23.466 on the same images. The published figure for this
  checkpoint is 23.43.
* `tools/compare.py` diffs the engine's per-activation dump against
  `tools/reference.py`'s: 43 of 43 named tensors agree, worst max abs 9.3e-6.
* `cargo test` restores a real low-light image and asserts it comes out brighter
  without saturating, and that both backends produce the same picture.

The parity tooling, the golden hashes and the measured per-op costs are in
[docs/DEVELOPING.md](docs/DEVELOPING.md); none of it is needed to run the engine.

## Licence and attribution

The Rust and CUDA code in this repository is licensed under the MIT license; see
[LICENSE](LICENSE).

This is an independent reimplementation of the MAXIM architecture, which is by
[google-research](https://github.com/google-research/maxim) and Apache-2.0
licensed (© 2022 Google LLC). `tools/reference.py` is a PyTorch transcription of
their network and is therefore a derived work, not covered by this repository's
copyright. The **checkpoint** is their work as well: the converted
`maxim-lol.safetensors` attached to the releases is a format conversion of the
official LOL enhancement `.npz`, redistributed under the same Apache-2.0 terms.
The original `.npz` is not redistributed here.

The upstream PSNR figure quoted above is from the MAXIM paper and repository and
is reproduced as the authors report it, not measured here.
