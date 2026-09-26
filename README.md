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
* 2.82 MiB binary, statically linked except `libc`, `libm` and `libgcc_s`.
  `libcuda.so.1` is `dlopen`ed, so no driver is required on disk. (The CPU-only
  build is 1.24 MiB.)
* All eleven checkpoints MAXIM was trained are converted and attached, so every
  task works out of the box - and each file carries its own architecture in its
  header, so there is no variant flag to get wrong. See Choosing a checkpoint.
* Fast on both backends: 0.72 s on a GTX 1080 and 5.8 s on 24 CPU threads for a
  600x400 image, against 7.7 s for the PyTorch reference.

## Download

Prebuilt binaries and the converted checkpoints are attached to the
[releases](https://github.com/jacobsparts/maxim-rs/releases). Both binaries run
on the CPU; they differ only in whether CUDA support is compiled in.

| asset | contents | notes |
|---|---|---|
| `maxim-linux-x86_64` | CPU + CUDA, auto-selected | x86-64 Linux with glibc ≥ 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); falls back to the CPU path when no NVIDIA driver is present. GPU path needs a compute capability 6.1+ GPU |
| `maxim-linux-x86_64-cpu-only` | CPU only | same, with nothing NVIDIA-related included - `--device gpu` is refused |
| 11 `maxim-*.safetensors` checkpoints | one per task and dataset | see Choosing a checkpoint |

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
separately for each task and each dataset, so the weights are not
interchangeable: an enhancement model run on a noisy frame, or a denoising model
on an underexposed one, is off its training distribution and can make the
picture worse rather than better. Every checkpoint upstream publishes is
attached, converted; each one records its own architecture in its header, so the
engine needs no flag to know how to read it.

| checkpoint | task | trained on | params | upstream PSNR |
|---|---|---|---|---|
| `maxim-lol.safetensors` | enhancement | LOL - low light | 14.2 M | 23.43 |
| `maxim-fivek.safetensors` | enhancement | FiveK - retouching | 14.2 M | 26.15 |
| `maxim-sidd.safetensors` | denoising | SIDD - real camera noise (also DND) | 22.2 M | 39.96 |
| `maxim-gopro.safetensors` | deblurring | GoPro - motion blur (also HIDE) | 22.2 M | 32.86 |
| `maxim-reds.safetensors` | deblurring | REDS - compressed video | 22.2 M | 28.93 |
| `maxim-realblur-r.safetensors` | deblurring | RealBlur-R - raw camera | 22.2 M | 39.45 |
| `maxim-realblur-j.safetensors` | deblurring | RealBlur-J - JPEG camera | 22.2 M | 32.84 |
| `maxim-rain13k.safetensors` | deraining | Rain13k - rain streaks | 14.2 M | 33.24 |
| `maxim-raindrop.safetensors` | deraining | Raindrop - drops on glass | 14.2 M | 31.87 |
| `maxim-sots-indoor.safetensors` | dehazing | RESIDE-Indoor (SOTS-Indoor) | 14.2 M | 38.11 |
| `maxim-sots-outdoor.safetensors` | dehazing | RESIDE-Outdoor (SOTS-Outdoor) | 14.2 M | 34.19 |

The PSNR figures are the upstream authors' own, not measured here. Two of their
thirteen published rows share a checkpoint with another row - the SIDD weights
are what they report for DND, and the GoPro weights are what they report for
HIDE - so eleven files cover all of them.

Pick by what the picture is, not by which file is newest. Denoising and
deblurring are the larger three-stage models (84.7 MiB, about 1.5x the CPU time
of the others on the same image); everything else is two-stage and 54.1 MiB.

To convert a checkpoint yourself, download the Flax `.npz` and run
`python3 tools/convert.py --npz checkpoint.npz --task denoising --out out.safetensors`.
The converter needs `numpy`; the engine needs neither `numpy` nor JAX. The
docstring of `tools/convert.py` has the loop that downloads and converts all
eleven.

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

## Accuracy

Both backends walk one op list, so they cannot drift apart by more than floating
point, and the output is checked against the authors' own published results on
whole official test sets - LOL (23.466 dB against their 23.4346), all 980
RealBlur-R images (37.387 against 37.113) and all 500 RESIDE-Indoor images (37.929
against 38.113). The engine lands within 0.02-0.04 dB of the official JAX
implementation on all three (worst 0.044 dB); how that is measured, and why the
published PNGs themselves score slightly differently, is in
[docs/DEVELOPING.md](docs/DEVELOPING.md).

## Licence and attribution

The Rust and CUDA code in this repository is licensed under the MIT license; see
[LICENSE](LICENSE).

This is an independent reimplementation of the MAXIM architecture, which is by
[google-research](https://github.com/google-research/maxim) and Apache-2.0
licensed (© 2022 Google LLC). `tools/reference.py` is a PyTorch transcription of
their network and is therefore a derived work, not covered by this repository's
copyright. The **checkpoints** are their work as well: each converted
`maxim-*.safetensors` attached to the releases is a format conversion of the
corresponding official `.npz`, redistributed under the same Apache-2.0 terms.
The original `.npz` files are not redistributed here.

The upstream PSNR figures quoted above are from the MAXIM paper and repository
and are reproduced as the authors report them, not measured here.
