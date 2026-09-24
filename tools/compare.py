#!/usr/bin/env python3
"""Diff two activation dumps, tensor by tensor.

    tools/compare.py /tmp/refd /tmp/cpud

Both sides are directories of `.npy` files written by `reference.py --dump`
and by `maxim --dump`. Names are matched by file stem, so the reference's
`stage0_enc0_bridge` and the engine's `stage0_enc0_bridge` line up.

For each tensor this reports

  * the maximum absolute difference,
  * the RMS difference (the number that says whether the two are the same
    arithmetic or merely the same shape),
  * the PSNR treating the value as an image in [0, 1], which is how the
    published results are quoted.

A tensor whose RMS is above `--tol` is called out. This is a tool for finding
the FIRST tensor that diverges, so the output is ordered by stage and level
rather than alphabetically.
"""

import argparse
import os
import sys

import numpy as np


def psnr(a, b):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    mse = float((d * d).mean())
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def key(name):
    """Order by stage, then by the numeric parts of the name."""
    import re

    parts = name.split("_")
    k = []
    for p in parts:
        m = re.match(r"^(.*?)(\d+)$", p)
        if m:
            k.append((m.group(1), int(m.group(2))))
        else:
            k.append((p, -1))
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("mine")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="RMS difference above which a tensor is a failure")
    ap.add_argument("--all", action="store_true", help="show every tensor")
    args = ap.parse_args()

    ref = {f[:-4]: os.path.join(args.ref, f) for f in os.listdir(args.ref) if f.endswith(".npy")}
    mine = {f[:-4]: os.path.join(args.mine, f) for f in os.listdir(args.mine) if f.endswith(".npy")}
    common = sorted(set(ref) & set(mine), key=key)
    if not common:
        sys.exit("compare: no names in common (ref has %d, mine has %d)" % (len(ref), len(mine)))

    print("%-22s %10s %12s %10s %8s" % ("tensor", "shape", "max abs", "rms", "psnr"))
    failures = 0
    for name in common:
        a = np.load(ref[name])
        b = np.load(mine[name])
        # The reference dumps NCHW batches, the engine [c][h][w]: same tensor,
        # two conventions, so a leading singleton is dropped on either side.
        a = a.squeeze()
        b = b.squeeze()
        if a.shape != b.shape:
            print("%-22s SHAPE %s vs %s" % (name, a.shape, b.shape))
            failures += 1
            continue
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        rms = float(np.sqrt((d * d).mean()))
        p = psnr(a, b)
        bad = not np.isfinite(rms) or rms > args.tol
        if bad or args.all:
            print("%-22s %10s %12.3e %10.3e %8.2f%s"
                  % (name, str(a.shape), float(d.max()), rms, p, "  <-- FAIL" if bad else ""))
        failures += bad
        if bad:
            # The first failure is the interesting one; the rest are downstream.
            print("first failure: %s at %s" % (name, ref[name]))
            break
    only_ref = sorted(set(ref) - set(mine), key=key)
    only_mine = sorted(set(mine) - set(ref), key=key)
    if only_ref:
        print("not compared (only in reference): %s" % ", ".join(only_ref))
    if only_mine:
        print("not compared (only in engine): %s" % ", ".join(only_mine))
    print("%d of %d tensors compared, %d failed" % (len(common), len(common), failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
