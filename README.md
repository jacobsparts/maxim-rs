# maxim-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).

MAXIM image restoration in one self-contained binary: low-light enhancement,
denoising, deblurring, deraining and dehazing. Feed it a PNG, get back a restored
PNG. No Python, JAX, PyTorch, ONNX Runtime or CUDA toolkit needed.

```
maxim -m maxim-lol.safetensors -i dark.png -o bright.png
```

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. The GPU is used when a CUDA driver is available and the
  CPU path otherwise, so one binary covers a machine with no NVIDIA driver at
  all; `--device cpu|gpu` overrides that choice.
* 2.82 MiB binary, statically linked except `libc`, `libm` and `libgcc_s`.
  `libcuda.so.1` is `dlopen`ed, so no driver is required on disk.
* All eleven checkpoints MAXIM was trained are converted and attached, so every
  task works out of the box - and each file carries its own architecture in its
  header, so there is no variant flag to get wrong.
* Fast on both backends: 0.72 s on a GTX 1080 and 5.8 s on 24 CPU threads for a
  600x400 image, against 7.7 s for the PyTorch reference.
* Sizes the pass before it starts it: the whole feature map is resident at once,
  so the footprint is arithmetic on the input size, and a pass that will not fit
  is refused with the numbers instead of failing half-way through.

The engine lands within 0.05 dB of the official JAX implementation on whole
official test sets (LOL, all 980 RealBlur-R images, all 500 RESIDE-Indoor images).

## Download

Prebuilt binary and the converted checkpoints are attached to the
[release](https://github.com/jacobsparts/maxim-rs/releases).

| asset | what it is |
|---|---|
| `maxim-linux-x86_64` | the engine: x86-64 Linux with glibc >= 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); falls back to the CPU path when no NVIDIA driver is present, the GPU path needs a compute capability 6.1+ GPU |
| 11 `maxim-*.safetensors` checkpoints | one per task and dataset; see Models |

```sh
chmod +x maxim-linux-x86_64      # a download does not carry the executable bit
./maxim-linux-x86_64 -m maxim-lol.safetensors -i dark.png -o bright.png
```

## Models

Which model you run is decided by the file you pass to `-m`. MAXIM was trained
separately for each task and each dataset, so the weights are not
interchangeable: an enhancement model run on a noisy frame, or a denoising model
on an underexposed one, is off its training distribution and can make the
picture worse rather than better.

| checkpoint | task | trained on | params |
|---|---|---|---|
| `maxim-lol.safetensors` | enhancement | LOL - low light | 14.2 M |
| `maxim-fivek.safetensors` | enhancement | FiveK - retouching | 14.2 M |
| `maxim-sidd.safetensors` | denoising | SIDD - real camera noise (also DND) | 22.2 M |
| `maxim-gopro.safetensors` | deblurring | GoPro - motion blur (also HIDE) | 22.2 M |
| `maxim-reds.safetensors` | deblurring | REDS - compressed video | 22.2 M |
| `maxim-realblur-r.safetensors` | deblurring | RealBlur-R - raw camera | 22.2 M |
| `maxim-realblur-j.safetensors` | deblurring | RealBlur-J - JPEG camera | 22.2 M |
| `maxim-rain13k.safetensors` | deraining | Rain13k - rain streaks | 14.2 M |
| `maxim-raindrop.safetensors` | deraining | Raindrop - drops on glass | 14.2 M |
| `maxim-sots-indoor.safetensors` | dehazing | RESIDE-Indoor (SOTS-Indoor) | 14.2 M |
| `maxim-sots-outdoor.safetensors` | dehazing | RESIDE-Outdoor (SOTS-Outdoor) | 14.2 M |

Denoising and deblurring are the larger three-stage models (84.7 MiB, about 1.5x
the CPU time of the others on the same image); everything else is two-stage and
54.1 MiB.

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
                      brought up, cpu otherwise)
    --cpu             same as --device cpu
    --gpu             same as --device gpu, and refuses to fall back
-q, --quiet           no progress output
-h, --help            this text
-V, --version         print the version
```

The input is padded up to a multiple of 64 and the result cropped back, so the
output has the input's dimensions.

## Memory

The whole feature map is resident at once on whichever backend runs, so there is
no tiling mode and no trade of memory for time: the footprint follows the input
and the stage count, and it is a plan total that every run prints. The arena is
ONE allocation, packed by buffer live range, so the number printed is exactly
what the driver is about to be asked for; when that does not fit, the pass is
refused before anything is allocated:

```console
$ maxim -m maxim-lol.safetensors -i 4096.png -o out.png
maxim: arena 29436 MiB (2089 buffers), weights up to 86 MiB, 8003 MiB free
maxim: not enough device memory for a 4096x4096 pass
```

There is deliberately no fallback from the GPU to the CPU on a memory failure -
a silent switch to a multi-gigabyte host allocation is worse than an error, and
`--gpu` turns even the driver-missing case into an error. The CPU backend holds
the same arena in host memory and is guarded the same way.

For scale, on an 8 GB card: 640x480 needs 539 MiB of arena, 1280x853 1869 MiB,
2048x1362 5059 MiB and 2048x2048 7359 MiB. The two-stage checkpoints
(enhancement, deraining, dehazing) need about half of what the three-stage ones
(denoising, deblurring) do at the same size.

## Licence and attribution

The Rust and CUDA code in this repository is licensed under the MIT license; see
[LICENSE](LICENSE). This is an independent reimplementation of the MAXIM
architecture, which is by
[google-research](https://github.com/google-research/maxim) and Apache-2.0
licensed (© 2022 Google LLC). `tools/reference.py` is a PyTorch transcription of
their network and is therefore a derived work. The **checkpoints** are their work
as well: each converted `maxim-*.safetensors` attached to the releases is a
format conversion of the corresponding official `.npz`, redistributed under the
same Apache-2.0 terms. The original `.npz` files are not redistributed here.
