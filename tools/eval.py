"""Score the engine on the LOL eval15 set.

This is the end-to-end check the unit tests cannot be: it runs the built binary
over all 15 pairs, measures PSNR against the ground truth, and compares the mean
with the published MAXIM number for this checkpoint (23.43). Parity tests prove
the engine reproduces the graph; this proves the graph is the *right* graph, so a
mistake that is consistent between the CPU and the GPU - a transposed weight, a
missing block - cannot hide in it.

With `--reference` it also runs `tools/reference.py` on the same images, which
is what makes a disagreement attributable: the reference is a separate
implementation of the same model, so if the engine's PSNR and the reference's
disagree per image, the engine is wrong; if they agree but both miss 23.43, the
checkpoint or the preprocessing is.

    python3 tools/eval.py --data /tmp/lol/eval15 --reference

`--data` is a directory holding `low/` and `high/` (or a zip of one).
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
from PIL import Image

PUBLISHED = 23.43


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")


def prepare(data, work):
    """A directory with low/ and high/, extracting the zip if that is what we got."""
    if os.path.isdir(data) and os.path.isdir(os.path.join(data, "low")):
        return data
    if not os.path.isfile(data):
        sys.exit("%s is neither an eval directory nor a zip" % data)
    with zipfile.ZipFile(data) as z:
        # The archive may hold the set at the top level (eval15/...) or nested
        # under a directory; find the member that ends in low/<name>.png and take
        # everything up to and including that eval15/ (or low/'s parent).
        names = [n for n in z.namelist() if n.endswith(".png") and "__MACOSX" not in n]
        low = [n for n in names if "/low/" in n or n.startswith("low/")]
        if not low:
            sys.exit("no low/*.png inside %s" % data)
        root = low[0].rsplit("/low/", 1)[0]
        if root.endswith("low"):
            root = os.path.dirname(root)
        keep = [n for n in names if n.startswith(root + "/" if root else "")]
        z.extractall(work, members=keep)
    return os.path.join(work, root) if root else work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../models/LOLdataset_train_test.zip",
                    help="a directory holding low/ and high/, or a zip of one")
    ap.add_argument("--bin", default="./target/release/maxim")
    ap.add_argument("--model", default="../models/maxim-lol.safetensors")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--reference", action="store_true", help="also run tools/reference.py")
    ap.add_argument("--ckpt", default="../models/checkpoint.npz")
    ap.add_argument("--out-dir", default=None, help="keep the outputs here instead of a temp dir")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="maxim-eval-")
    os.makedirs(out_dir, exist_ok=True)
    data = prepare(args.data, tempfile.mkdtemp(prefix="lol-"))

    lows = sorted(glob.glob(os.path.join(data, "low", "*.png")))
    if args.limit:
        lows = lows[: args.limit]
    if not lows:
        sys.exit("no images under %s/low" % data)

    rows = []
    for low in lows:
        name = os.path.splitext(os.path.basename(low))[0]
        high = os.path.join(data, "high", name + ".png")
        gt = read_rgb(high)

        eng_path = os.path.join(out_dir, "engine_%s.png" % name)
        subprocess.run([args.bin, "--model", args.model, "-i", low,
                        "-o", eng_path, "--device", args.device],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # The engine pads the input to a multiple of 64 and crops back, so the
        # output is the ground truth's size; asserting it here catches a change
        # in that contract before the PSNR is computed on mismatched arrays.
        eng = read_rgb(eng_path)
        if eng.shape != gt.shape:
            sys.exit("engine output %s is %s, the ground truth is %s"
                     % (name, eng.shape, gt.shape))

        ref = None
        if args.reference:
            ref_path = os.path.join(out_dir, "reference_%s.png" % name)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "reference.py"),
                            "--ckpt", args.ckpt, "--variant", "S-2", "--image", low,
                            "--out", ref_path],
                           check=True, stdout=subprocess.DEVNULL)
            ref = read_rgb(ref_path)

        rows.append((name, psnr(eng, gt), None if ref is None else psnr(ref, gt),
                     None if ref is None else float(np.abs(eng - ref).mean())))

    print("%-8s %10s %12s %14s" % ("image", "engine", "reference", "mean |eng-ref|"))
    for name, e, r, d in rows:
        print("%-8s %10.3f %12s %14s"
              % (name, e, "-" if r is None else "%.3f" % r, "-" if d is None else "%.4f" % d))
    mean = sum(r[1] for r in rows) / len(rows)
    print("\n%d images   engine mean PSNR %.3f   published %.2f" % (len(rows), mean, PUBLISHED))
    if args.reference:
        rmean = sum(r[2] for r in rows) / len(rows)
        delta = abs(mean - rmean)
        print("reference mean PSNR %.3f   engine-reference delta %.4f" % (rmean, delta))
        if delta > 0.01:
            print("the engine and the reference disagree on the MEAN: that is an "
                  "engine bug, not a checkpoint or preprocessing difference")
            return 1
    # The published figure is the mean over the whole eval15 set, so a subset
    # (--limit) has no expected value to compare against - the first three
    # images are all below the mean, and warning about that would be noise.
    if args.limit or len(rows) != 15:
        print("(a subset: the published %.2f is a mean over all 15 images)" % PUBLISHED)
    elif abs(mean - PUBLISHED) > 0.5:
        print("mean PSNR is more than 0.5 dB from the published %.2f" % PUBLISHED)
        return 1
    print("outputs kept in %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
