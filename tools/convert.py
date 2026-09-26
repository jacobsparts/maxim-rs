"""Convert MAXIM's Flax `.npz` checkpoint to a `safetensors` file.

Upstream publishes one checkpoint per task and dataset, and which of the six
variants each one is follows from the task alone, so they all convert with the
same command - which is copied here rather than described, because it is also
how the eleven attached to the release were produced and checked:

    base=https://storage.googleapis.com/gresearch/maxim/ckpt
    while read dir npz out task; do
        curl -sS -o $npz.npz "$base/$dir/checkpoint.npz"
        python3 tools/convert.py --npz $npz.npz --task $task --out $out
    done <<'EOF'
    Denoising/SIDD         denoising-sidd       maxim-sidd.safetensors         denoising
    Deblurring/GoPro       deblurring-gopro     maxim-gopro.safetensors        deblurring
    Deblurring/REDS        deblurring-reds      maxim-reds.safetensors         deblurring
    Deblurring/RealBlur_R  deblurring-realblur-r maxim-realblur-r.safetensors  deblurring
    Deblurring/RealBlur_J  deblurring-realblur-j maxim-realblur-j.safetensors  deblurring
    Deraining/Rain13k      deraining-rain13k    maxim-rain13k.safetensors      deraining
    Deraining/Raindrop     deraining-raindrop   maxim-raindrop.safetensors     deraining
    Dehazing/SOTS-Indoor   dehazing-indoor      maxim-sots-indoor.safetensors  dehazing
    Dehazing/SOTS-Outdoor  dehazing-outdoor     maxim-sots-outdoor.safetensors dehazing
    Enhancement/LOL        enhancement-lol      maxim-lol.safetensors          enhancement
    Enhancement/FiveK      enhancement-fivek    maxim-fivek.safetensors        enhancement
    EOF

The .npz names are the ones this loop downloads to, and the header records the
basename, so re-running it from scratch reproduces the published files byte for
byte. (The `numpy <version>` that used to be in that string did not: it was a
fact about the machine, not about the model.)

Denoising/ and Denoising/SIDD are the same weights, and GoPro serves HIDE and
SIDD serves DND, so eleven files cover all fourteen published task rows. Each is
54.1 MiB (S-2) or 84.7 MiB (S-3) once the optimizer state is dropped.

Only the `opt/target/*` leaves are weights; the rest of the file is optimizer
state (~75% of the bytes) and is dropped. The names keep their flax form
(`stage_0_encoder_block_0/Conv_0/kernel`) because that is what the model code
reads; the engine's weight loader is what knows a `kernel` under a
`ConvTranspose_0` is stored [kh, kw, c_in, c_out] and must be flipped, rather
than baking a transformation in here where it cannot be seen.

The header's `__metadata__` records what the engine could otherwise only guess:
the task the checkpoint was trained for and the variant. The engine does NOT
depend on it - it derives the architecture from the weights themselves, because
a file converted before this existed must still load - but a self-describing
checkpoint is what `realesrgan-rs` and `nafnet-rs` publish, and it is what tells
a human which of the six published models they are holding.

    python3 tools/convert.py --npz ../models/checkpoint.npz --task enhancement \
        --out ../models/maxim-lol.safetensors
"""

import argparse
import json
import os
import sys

import numpy as np

# safetensors dtype names, by numpy kind
DTYPES = {
    np.dtype("float32"): "F32",
    np.dtype("float16"): "F16",
    np.dtype("float64"): "F64",
    np.dtype("int32"): "I32",
}


def variant_of(tensors):
    """The published variant these weights are.

    Feature width is the last dimension of the first encoder conv; the stage
    count is how many `stage_N_*` groups the checkpoint has. Both are facts about
    the bytes, so writing them down cannot make the file wrong - which is also
    why the engine derives them itself rather than trusting this string.
    """
    by_name = dict(tensors)
    features = by_name["stage_0_encoder_block_0/Conv_0/kernel"].shape[-1]
    stages = 1
    for n in range(1, 8):
        if any(k.startswith("stage_%d_" % n) for k in by_name):
            stages = n + 1
    if features not in (32, 64) or not 1 <= stages <= 3:
        return "unknown"
    return "%s-%d" % ("S" if features == 32 else "M", stages)


def convert(npz_path, out_path, task="enhancement"):
    basename = os.path.basename(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        keys = sorted(k[len("opt/target/"):] for k in z.keys() if k.startswith("opt/target/"))
        if not keys:
            sys.exit("no opt/target/* leaves: is this a MAXIM checkpoint?")
        tensors = []
        total = 0
        for k in keys:
            a = np.asarray(z["opt/target/" + k])
            if a.dtype not in DTYPES:
                a = a.astype(np.float32)
            if not a.flags["C_CONTIGUOUS"]:
                a = np.ascontiguousarray(a)
            tensors.append((k, a))
            total += a.nbytes

    header = {}
    header["__metadata__"] = {
        "format": "maxim",
        "task": task,
        "variant": variant_of(tensors),
        "tensor_count": str(len(tensors)),
        # The source file, as realesrgan-rs records it. Deliberately NOT the
        # numpy version that happened to be installed, which was in this string
        # before: a build string makes the same weights convert to a different
        # file on another machine, and the point of this script is that the
        # published checkpoints are reproduced by re-running it on the .npz.
        "source": "MAXIM Flax checkpoint %s, opt/target/* leaves" % basename,
    }
    offset = 0
    with open(out_path, "wb") as f:
        f.write(b"\0" * 8)                       # placeholder for the header length
        for k, a in tensors:
            header[k] = {
                "dtype": DTYPES[a.dtype],
                "shape": list(a.shape),
                "data_offsets": [offset, offset + a.nbytes],
            }
            offset += a.nbytes
        blob = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
        # the tensor data starts at 8 + len(blob), padded to a multiple of 8
        pad = (-(8 + len(blob))) % 8
        blob += b" " * pad
        f.seek(0)
        f.write(len(blob).to_bytes(8, "little"))
        f.write(blob)
        for _, a in tensors:
            f.write(a.tobytes())
    print("wrote %s: %d tensors, %.1f MiB" % (out_path, len(tensors), total / 2**20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--task",
        default="enhancement",
        choices=["enhancement", "denoising", "deblurring", "deraining", "dehazing"],
        help="the task the checkpoint was trained for, recorded in the header; "
        "the engine does not read it",
    )
    args = ap.parse_args()
    convert(args.npz, args.out, args.task)


if __name__ == "__main__":
    main()
