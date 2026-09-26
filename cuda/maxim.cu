// MAXIM's own kernels.
//
// These are the ops the lightgpu toolkit does not already have: the two strided
// convs flax's `Conv_down`/`ConvT_up` need, the gated-MLP matmul, the bilinear
// resize with jax's antialiasing rule, the block/grid permutation, and the gated
// multiply `y = u * (v + 1)`.
//
// Everything here is NCHW, except the gating matmuls and `mx_block_perm`, which
// work on the *blocked* layout MAXIM's gMLP layers are written in.
//
// `mx_mul` used to be here too; it is the toolkit's `lg_mul` now (see the
// promotion note at the bottom).
//
// Several of these are the TILED form of an op this engine used to run one
// output element per thread: `mx_conv1x1_t` replaces the toolkit's `lg_conv1x1`,
// `mx_conv3x3_t2`/`_t4` replace its `lg_conv3x3s1p1`, `mx_gate_mm_t0`/`_t1`
// replace this file's own `mx_gate_mm`, and `mx_conv4x4s2` replaces the untiled
// version of itself. The originals have been deleted, so the
// numbers below are what the old kernels measured while they were still here,
// from a one-off A/B against the tiled build - each tiled form shares one load
// across several outputs where the original re-read every operand per output.

// 3x3 pad-1 conv, tiled: this engine's replacement for the toolkit's
// `lg_conv3x3s1p1`, which this engine no longer launches at ANY width - the
// segment's `ow` (below) makes the tile cover a short last segment, so the only
// thing `--factor` changes is how many segments a row has. The toolkit kernel is
// still listed in the build for the other engines that use it, not for this one.
//
// That kernel gives each output element its own thread, so per output it issued
// 9*c_in loads of the input and 9*c_in of the weight - about 1 MAC per 8 bytes
// moved - and at 512x512 this family was the largest share of the run (1.90 s of
// a 4.32 s run with the untiled kernels, ~1.3 TFLOP/s against this card's 8.9).
// Here a 64-pixel block
// of outputs is computed from a shared-memory tile, so one input load feeds 64
// outputs and one weight load 64 more: ~47 MACs per load.
//
// The pixel block is a SEGMENT OF ONE ROW, which is what makes the halo cheap: 64
// outputs need 66 input columns per (ci, row), so a 3x3 window is three offsets
// into the same tile rather than three separate loads.
//
// The LAST segment of a row may be partial. `ow` is how many output columns the
// segment actually owns; the store stops there and the tile load, which already
// bounds-checks every read against `wd`, is untouched. This is what keeps a row
// whose width is not a multiple of 64 on the tiled path: the plan's pad-to-64
// gives the top level 640, but halving on the way down gives 160 and 80, and
// before `ow` existed the host dropped those whole rows to the toolkit kernel
// `lg_conv3x3s1p1`. That was the largest single cost in the 600x400 profile -
// 17 conv3x3 128->128 @112x160 at ~46 ms each, 785 ms of a 3.1 s run - because
// every level below the first has such a width.
//
// `PXT` (pixels per thread, 4 or 2) picks the block shape and hence the number of
// output channels per block: 4 -> 16x16 threads and 64 channels, 2 -> 32x8 and 32.
// The host picks by `c_out` so that a 32-channel layer does not run with half the
// block idle.
//
// Accumulation order is dy, dx, ci - the same as the toolkit kernel and the CPU
// twin, which makes the tiled form agree with both bit-for-bit rather than
// merely within tolerance.
constexpr int C3_CI = 8;   // input channels per K chunk

// THE WEIGHT LAYOUT, and it is the whole reason this kernel is fast.
//
// The weight arrives as `[tap][ci][oc]` (k*k, then c_in, then c_out - flax's own
// order; see `weights::conv3x3_t`), not the `[oc][ci][tap]` the toolkit's
// `lg_conv3x3s1p1` wants. That is deliberate, and it was measured: with the old
// layout the four weights a thread needs for its four output channels sit 512
// floats apart in the shared tile, so every 48 multiply-accumulates spent 14
// shared-memory loads (2 for the input column, 12 for the weights) and on Pascal
// a shared load costs about four FMAs of issue, which is where the throughput went.
// With `[tap][ci][oc]` those four weights are CONTIGUOUS, so they are one 128-bit
// load: 5 loads per 48 FMAs instead of 14. The staging read gets faster too, since
// consecutive threads now read consecutive `oc` of one `(tap, ci)` row.
//
// Measured head to head against the old layout, both launched with their own
// geometry, output identical to the element against a host reference (the same
// worst difference and the same mismatch count from both, so the change is a layout
// change and not an arithmetic one):
//     32->32  @448x640   3.406 -> 2.174 ms   1551 -> 2431 GFLOP/s
//     64->64  @224x320   2.245 -> 1.708 ms   2354 -> 3095 GFLOP/s
//     128->128 @112x160  2.616 -> 2.026 ms   2020 -> 2609 GFLOP/s
//
// The accumulation order is unchanged either way: dy, then ci, then dx, with the
// four output channels accumulated in ascending oc - which is what makes the two
// layouts agree bit for bit rather than merely closely.
template <int TX, int TY, int PXT>
__device__ __forceinline__ void c3_body(
    const float *__restrict__ in, const float *__restrict__ wt,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int x0, int segs)
{
    constexpr int LOC = TY * 4;             // output channels per block
    constexpr int PX = TX * PXT;            // output pixels per block
    constexpr int XW = PX + 2;              // input columns: 64 outputs + halo
    constexpr int XP = XW + 1;              // padded row stride (odd)
    // The weight tile's row stride, [tap][ci] rows of LOC output channels. Padded
    // to a multiple of 4 so each thread's four weights are 16-byte aligned, which
    // is what the float4 load below requires - and a MISALIGNED float4 load does
    // not fault, it returns the wrong four floats.
    constexpr int W = LOC + 4;

    const int oc0 = blockIdx.x * LOC;
    // A segment is 64 output columns of ONE row, and the last one may be short:
    // `x0` and `ow` come from the host, which enumerates the segments per row, so
    // a width that is not a multiple of 64 costs one narrow segment per row
    // rather than the whole row falling back to the untiled kernel.
    // The grid is (output-channel blocks, rows * segments, 1) - segments along y,
    // NOT on z. Measured on the GTX 1080 at 128->128 @112x160, launching the same
    // work as (oc, rows, segments) costs 4.43 ms against 3.98 ms for this layout,
    // a reproducible 11%, because the z axis orders the blocks so that the
    // 64-column segments of a row are no longer adjacent in launch order (the L2
    // reads a row's neighbours together). A division is much cheaper than that, so
    // the segment is recovered here rather than given its own axis.
    // THE SLOW INDEX IS (z, y) FLATTENED, because one dimension is not enough:
    // gridDim.y is capped at 65535 and the level-0 conv3x3 of a 2048x2048 image
    // wants h * segs = 2048 * 32 = 65536 blocks, one past the limit. `segs` comes
    // from the host rather than from `gridDim.y / h` for the same reason the flat
    // layout is kept wherever it fits: the fold above measures 11% slower when
    // segments go on an axis of their own, so the two layouts coexist and only
    // the host knows which one it launched.
    //
    // The split is a ceiling, so `y0` can land past the last row; those blocks
    // must do nothing rather than write past the plane. With the flat layout
    // (z == 0) the guard is never true, so nothing about a size that already
    // worked changes.
    const int slow = blockIdx.z * gridDim.y + blockIdx.y;
    const int per_row = segs;               // segments per row
    const int y0 = slow / per_row;
    const int seg = slow % per_row;
    if (y0 >= h) return;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const int sx0 = x0 + seg * PX;
    // How many columns THIS segment owns. Derived rather than passed: a host
    // argument cannot express a different value per block, and getting it wrong is
    // silent - it drops columns (measured: passing a flat `ow` lost exactly the 32
    // trailing columns of every row at 96 and 160 wide, 65536 elements at 16x96).
    const int ow = (sx0 + PX <= wd) ? PX : (wd - sx0);
    const size_t plane = (size_t)h * wd;

    // [tap][ci][oc], row-padded so a thread's four weights land on one aligned
    // float4 and so the strided reads below do not put two threads in one bank.
    __shared__ float sw[C3_CI * 9 * W];
    // [ci][dy][column], where column 0 is input x0 - 1.
    __shared__ float sx[C3_CI * 3 * XP];

    float acc[4][PXT];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float b = (bias && oc0 + ty * 4 + i < c_out) ? bias[oc0 + ty * 4 + i] : 0.0f;
        #pragma unroll
        for (int j = 0; j < PXT; ++j) acc[i][j] = b;
    }

    for (int ci0 = 0; ci0 < c_in; ci0 += C3_CI) {
        // Weight tile, one element per thread per pass. Consecutive threads take
        // consecutive `oc` (contiguous in the global weight AND in this tile), so
        // both the load and the store coalesce.
        for (int i = tid; i < C3_CI * 9 * LOC; i += 256) {
            const int tt = i / LOC;
            const int oc = i % LOC;
            const int tap = tt / C3_CI, ci = tt % C3_CI;
            const int goc = oc0 + oc, gci = ci0 + ci;
            float v = 0.0f;
            if (goc < c_out && gci < c_in)
                v = wt[((size_t)tap * c_in + gci) * c_out + goc];
            sw[tt * W + oc] = v;
        }
        // Input tile, halo included; out-of-range reads are zeros, which is the
        // same arithmetic as the toolkit kernel's `continue` on a bounds test.
        for (int i = tid; i < C3_CI * 3 * XW; i += 256) {
            const int ci = i / (3 * XW);
            const int r = i % (3 * XW);
            const int dy = r / XW, col = r % XW;
            const int iy = y0 + dy - 1, ix = sx0 + col - 1;
            const int gci = ci0 + ci;
            float v = 0.0f;
            if (gci < c_in && iy >= 0 && iy < h && ix >= 0 && ix < wd)
                v = in[(size_t)gci * plane + (size_t)iy * wd + ix];
            sx[(ci * 3 + dy) * XP + col] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int dy = 0; dy < 3; ++dy) {
            #pragma unroll
            for (int ci = 0; ci < C3_CI; ++ci) {
                // The PXT + 2 inputs this thread's pixels need, for every one of
                // the three taps: one load per column instead of one per tap.
                float xv[PXT + 2];
                #pragma unroll
                for (int j = 0; j < PXT + 2; ++j)
                    xv[j] = sx[(ci * 3 + dy) * XP + tx * PXT + j];
                #pragma unroll
                for (int dx = 0; dx < 3; ++dx) {
                    // ONE 128-bit load for the four output channels this thread
                    // owns - the entire point of the [tap][ci][oc] layout.
                    const float4 wv4 = *reinterpret_cast<const float4 *>(
                        &sw[((dy * 3 + dx) * C3_CI + ci) * W + ty * 4]);
                    const float wv[4] = {wv4.x, wv4.y, wv4.z, wv4.w};
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        #pragma unroll
                        for (int j = 0; j < PXT; ++j) acc[i][j] += wv[i] * xv[dx + j];
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (oc0 + ty * 4 + i >= c_out) continue;
        const int col = tx * PXT;
        const size_t row = (size_t)(oc0 + ty * 4 + i) * plane + (size_t)y0 * wd + sx0 + col;
        #pragma unroll
        for (int j = 0; j < PXT; ++j)
            if (col + j < ow) out[row + j] = acc[i][j];
    }
}

// 4 pixels per thread: 64 output channels per block, 16x16 threads.
//
// `__launch_bounds__(256, 3)` is worth 1.75x and it is the whole story of this
// kernel's occupancy. Without it ptxas spends 142 registers on the body, which at
// 256 threads a block is 36352 of the SM's 65536 and therefore allows exactly ONE
// block per SM: 8 warps, 12.5% occupancy, and the measured 1.07-1.27 TFLOP/s was
// 13% of this card's fp32 peak, i.e. throughput tracking occupancy. Asking for
// three blocks (16 warps, 37.5%) takes it to 80 registers with 136 bytes of
// spill, which the extra warps more than pay for; four blocks (64 registers, 176+
// bytes of spill) measurably loses. Measured with a host reference, identical
// output element for element at every setting:
//   64->64  @224x320   3.857 -> 2.201 ms (1370 -> 2401 GFLOP/s)
//   128->128 @112x160  4.448 -> 2.540 ms (1188 -> 2080 GFLOP/s)
extern "C" __global__ void __launch_bounds__(256, 3) mx_conv3x3_t4(
    const float *__restrict__ in, const float *__restrict__ wt,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int x0, int segs)
{
    c3_body<16, 16, 4>(in, wt, bias, out, c_in, c_out, h, wd, x0, segs);
}

// 2 pixels per thread: 32 channels per block, 32x8 threads, for c_out < 64.
//
// THREE blocks per SM, and the "two, not three" this comment used to argue for is
// now stale: it was written for a form that wanted 142 registers and spilled 192
// bytes at a three-block bound, whereas this one compiles to 89 registers and its
// input tile is 16800 bytes, so three blocks cost neither. The geometry below is
// byte-for-byte the form that was measured; the ONLY change is the launch bound,
// and it is worth 1.28-1.34x at every shape of this family (interleaved
// round-robin best-of-five in one harness, 0 differing at all of them):
//     32->32 @448x640   2.813 -> 2.101 ms   1878 -> 2515 GFLOP/s  1.34x
//     32->32 @224x320   0.709 -> 0.530      1863 -> 2494          1.34x
//     32->32 @112x160   0.225 -> 0.168      1468 -> 1970          1.34x
//     32->32 @56x80     0.073 -> 0.057      1132 -> 1453          1.28x
extern "C" __global__ void __launch_bounds__(256, 3) mx_conv3x3_t2(
    const float *__restrict__ in, const float *__restrict__ wt,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int x0, int segs)
{
    c3_body<32, 8, 2>(in, wt, bias, out, c_in, c_out, h, wd, x0, segs);
}

// A WIDE SEGMENT: 128 output columns per block instead of 64, and the geometry is
// the same 32x8 threads with 4 pixels each, so the only thing that changes is how
// many pixels one block owns.
//
// The weight tile is restaged by EVERY block and its traffic is
// 9*c_in*c_out*h*segs floats, so halving `segs` halves it - which is the larger
// half of this kernel's staging cost (zeroing just that read, with the layout and
// all the index arithmetic intact, took 32->32 @448x640 3.45 -> 2.20 ms). What it
// costs is a wider input tile (CI * 3 * (PX + 3) floats) and a padded last segment
// that can be up to 127 columns wide instead of 63.
//
// The two effects trade off per plane, and this is the whole plane sweep at
// 32->32 @448x640-oid shapes, interleaved round-robin best-of-five, every entry
// bit-identical to `mx_conv3x3_t2` (the timing is ms, and the last column is this
// kernel against `mx_conv3x3_t2`):
//     wd   t2(m3)  wide(m4)   ratio   padded columns (64 vs 128)   wide/t2
//     640  2.1844  1.7413    1.25x   640  640                       1.25x
//     512  1.6550  1.4147    1.17x   512  512                       1.17x
//     384  1.2531  1.0583    1.18x   384  384                       1.18x
//     320  1.0452  1.0268    1.02x   320  384                       1.02x
//     256  0.8346  0.7122    1.17x   256  256                       1.17x
//     224  0.8297  0.7030    1.18x   224  256                       1.18x
//     192  0.6285  0.6815    0.92x   192  256                       0.92x
//     160  0.6246  0.6661    0.94x   160  256                       0.94x
//     128  0.4211  0.3676    1.15x   128  128                       1.15x
//      96  0.4152  0.3533    1.18x    96  128                       1.18x
//      80  0.4035  0.3425    1.18x    80  128                       1.18x
//      64  0.2158  0.3366    0.64x    64  128                       0.64x
//      40  0.2125  0.3220    0.66x    64  128                       0.66x
// So the outcome is decided by how much PADDED work the wide form adds, not by any
// single width: equal padded counts are a 1.15-1.25x win, a 1.33x padded ratio is a
// 0.92-0.94x loss, and the narrowest planes are the worst case of all because the
// wide grid still runs a whole 128-column block for 40 columns of image. The host
// therefore picks by padded width - and note that 640 and 320 are the only two of
// those it ever sees.
extern "C" __global__ void __launch_bounds__(256, 4) mx_conv3x3_w4(
    const float *__restrict__ in, const float *__restrict__ wt,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int x0, int segs)
{
    c3_body<32, 8, 4>(in, wt, bias, out, c_in, c_out, h, wd, x0, segs);
}

// 1x1 conv over a TRANSPOSED weight, `[c_in][c_out]` - this engine's own
// replacement for the toolkit's `lg_conv1x1`.
//
// The profile made the reason concrete: at 512x512 the gMLP's 1x1 convs hold 30%
// of the run, and `lg_conv1x1` was moving about 26x the traffic the op needs. It
// puts one thread on each output element, so a thread walks its input column one
// `plane`-strided float at a time and the block touches a 32-byte sector per tap
// to use 4 of its 8 floats - and because the weight is `[c_out][c_in]`, the
// alternative of giving each thread several output channels would stride c_in
// through the weight on every step. Both halves of that are fixed by the
// transposed weight the plan carries as `wt`: consecutive outputs become
// consecutive, and the input column is then shared by several outputs.
//
// WHAT THE SHIPPED FORM IS, and it took five measured variants to get here. The
// block owns 128 PIXELS (32 threads x C1_PXT) of C1_TILE or TILE output channels
// and stages BOTH operands in shared memory:
//   * the input as a 32 x 128 tile of the C1_CI channels of this K chunk, so the
//     TY threads that share a pixel column read it ONCE between them instead of
//     TY times. This is the big one: 1.45-1.62x by itself.
//   * the weight as a `[ci][oc]` tile with row stride LOC+4, so a thread's TILE
//     weights are CONTIGUOUS and one 128-bit LDS.128 replaces TILE scalar LDS -
//     the same trick, for the same reason, as conv3x3's `[tap][ci][oc]`. Worth
//     another 4-8% on top of the input tile. The +4 padding must be a multiple of
//     4 because a MISALIGNED float4 load returns the wrong four floats, it does
//     not fault; `LOC` is a multiple of 4 for both instantiations, so every
//     thread's base is aligned by construction.
//
// THE MEASUREMENTS, all interleaved round-robin best-of-five in one harness (the
// run-to-run spread on this card is ~40%, so the round-robin is what makes them
// evidence), and ALL of them bit-identical to the kernel they replace - 0
// differing elements against the shipped kernel's own output at all ten shapes,
// because every variant preserves the accumulation order exactly: ic0 chunks
// ascending, ci ascending inside a chunk, TILE outputs in oc order. Gains are
// against the deleted `mx_conv1x1_t` (scalar input loads, `[oc][ci]` weight tile)
// and the shape that won is the one below:
//
//                       TILE=4                  TILE=8
//   64->32  @448x640    0.8067 -> 0.4702 (1.72x)  0.5554 (1.45x)
//   32->64  @448x640    0.9706 -> 0.6345 (1.53x)  0.5920 (1.64x)
//   128->64 @224x320    0.6938 -> 0.4045 (1.72x)  0.2952 (2.35x)
//   32->32  @448x640    0.4904 -> 0.3232 (1.52x)  0.3486 (1.41x)
//   64->128 @224x320    0.7893 -> 0.4711 (1.68x)  0.3650 (2.16x)
//   128->256@112x160    0.6931 -> 0.4051 (1.71x)  0.2942 (2.36x)
//   256->128@112x160    0.6534 -> 0.3673 (1.78x)  0.2688 (2.43x)
//   64->64  @224x320    0.4094 -> 0.2416 (1.69x)  0.1946 (2.10x)
//   128->128@112x160    0.3594 -> 0.2126 (1.69x)  0.1585 (2.27x)
//   128->128@56x80      0.1015 -> 0.0579 (1.75x)  0.0555 (1.83x)
//
// So the host picks by `c_out`, exactly as conv3x3's t2/t4 split does: TILE=8
// (mx_conv1x1_t8, 64 channels per block) above 64 output channels, TILE=4
// (mx_conv1x1_t4, 32) at and below, where a 64-channel block would waste half the
// grid - that waste is visible as the two shapes TILE=8 loses (64->32 and 32->32).
// ptxas: 58 registers / 20992 B for TILE=4, 75 / 25088 for TILE=8, both under
// `__launch_bounds__(256, 3)`; the smem stays under the 48 KB default so neither
// needs an opt-in.
//
// THE VARIANTS THAT LOST, kept here because each one is a dead end someone would
// otherwise re-try:
//   * a C1_CI of 64 instead of 32 is 2.0-2.3x SLOWER at every shape (64->32
//     @448x640 0.79 -> 1.74 ms). Fewer barriers is not worth it at any size, so
//     the barrier count was never this kernel's cost.
//   * dropping the shared memory entirely and taking the TILE weights from global
//     as one `__ldg` broadcast float4 per (ci, threadIdx.y) - which also deletes
//     BOTH barriers - is 1.07-1.13x faster than the old kernel (40 registers, 0
//     smem) and still slower than the staged forms at all ten shapes, and it is
//     the only variant that LOSES at the smallest shape (1.079 vs 0.0994 ms). The
//     weight is 4-32 KB and L1-resident, but the block still issues eight
//     redundant LDG.128 per pixel column.
//   * the input tile WITHOUT the transposed weight tile is 1.44-1.62x, i.e. the
//     weight layout is worth a further 4-8% and the input tile is worth the rest.
//
// `wd` is the contiguous axis, so the input tile's rows are coalesced reads.
constexpr int C1_PXT = 4;   // pixels per thread
constexpr int C1_TILE = 4;  // output channels per thread in the SCALAR fallback
constexpr int C1_TILES = 8; // output channels per thread in mx_conv1x1_t8
// 32, not 8. The staging loop loads LOC*C1_CI weights with TX*TY threads and then
// pays TWO __syncthreads, so C1_CI sets how many MACs amortise a barrier: at 8 that
// is 128 MACs per barrier per thread and the barriers cost ~10% of the op (measured
// standalone at 128->256 @112x160: CI=8 2.752 ms, CI=16 2.538, CI=32 2.492, CI=64
// 2.621 - 64 loses to shared-memory pressure). The accumulation is over `ic0` chunks
// in ascending order either way and `ci` ascends inside a chunk, so the sum is the
// same sequence of additions and the result is bit-identical: verified 0 differing
// elements of 4587520 against a host reference at CI=8, 16, 32 and 64.
constexpr int C1_CI = 32;   // input channels per K chunk

// The tiled body. `TILE` is the output channels per thread, and it is the only
// thing that differs between the two instantiations below.
template <int TILE>
__device__ __forceinline__ void c1_body(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd)
{
    constexpr int TX = 32, TY = 256 / TX;
    constexpr int PX = TX * C1_PXT;       // 128 pixels per block
    constexpr int LOC = TY * TILE;        // output channels per block
    constexpr int WSW = LOC + 4;          // weight row stride, 16-byte aligned
    const size_t plane = (size_t)h * wd;
    const int px0 = blockIdx.x * PX;
    const int p0 = px0 + threadIdx.x * C1_PXT;
    const int oc0 = blockIdx.y * LOC;
    const int tid = threadIdx.y * TX + threadIdx.x;
    __shared__ float sw[C1_CI * WSW];
    __shared__ float sx[C1_CI][PX];
    float acc[TILE][C1_PXT];
    #pragma unroll
    for (int i = 0; i < TILE; ++i) {
        const int oc = oc0 + threadIdx.y * TILE + i;
        const float b = (oc < c_out && bias) ? bias[oc] : 0.0f;
        #pragma unroll
        for (int j = 0; j < C1_PXT; ++j) acc[i][j] = b;
    }
    for (int ic0 = 0; ic0 < c_in; ic0 += C1_CI) {
        // Weight tile, [ci][oc]: consecutive threads take consecutive `oc`, which
        // is contiguous in the global weight AND in this tile, so both the load
        // and the store coalesce.
        for (int i = tid; i < C1_CI * LOC; i += TX * TY) {
            const int ci = i / LOC, oc = i % LOC;
            const int goc = oc0 + oc, gci = ic0 + ci;
            float v = 0.0f;
            if (goc < c_out && gci < c_in) v = w[(size_t)gci * c_out + goc];
            sw[ci * WSW + oc] = v;
        }
        // Input tile, one float4 per thread per pass (32 channels x 32 float4s =
        // 1024, i.e. exactly four passes of 256 threads). Out-of-range reads are
        // zero, which is the same arithmetic as the bounds test in the fallback.
        // A float4 load needs its address 16-byte aligned, so the vector path is
        // taken only when the whole quad is inside the plane; the last partial
        // block of a plane falls to per-element guarded scalar reads.
        #pragma unroll
        for (int i = tid; i < C1_CI * (PX / 4); i += TX * TY) {
            const int ci = i / (PX / 4), j = (i % (PX / 4)) * 4;
            const int q = px0 + j;
            float4 v;
            if (q + 4 <= (int)plane) {
                v = *reinterpret_cast<const float4 *>(&in[(size_t)(ic0 + ci) * plane + q]);
            } else {
                v = make_float4(
                    (q + 0 < (int)plane) ? in[(size_t)(ic0 + ci) * plane + q + 0] : 0.0f,
                    (q + 1 < (int)plane) ? in[(size_t)(ic0 + ci) * plane + q + 1] : 0.0f,
                    (q + 2 < (int)plane) ? in[(size_t)(ic0 + ci) * plane + q + 2] : 0.0f,
                    (q + 3 < (int)plane) ? in[(size_t)(ic0 + ci) * plane + q + 3] : 0.0f);
            }
            *reinterpret_cast<float4 *>(&sx[ci][j]) = v;
        }
        __syncthreads();
        #pragma unroll
        for (int ci = 0; ci < C1_CI; ++ci) {
            // The thread's own four pixels: one LDS.128, no bank conflict (each
            // thread reads a different 16-byte slot of the row).
            const float4 xv4 = *reinterpret_cast<const float4 *>(&sx[ci][threadIdx.x * C1_PXT]);
            const float cur[C1_PXT] = {xv4.x, xv4.y, xv4.z, xv4.w};
            // ...and its own TILE weights, as TILE/4 aligned LDS.128s.
            float wv[TILE];
            #pragma unroll
            for (int t = 0; t < TILE; t += 4) {
                const float4 w4 = *reinterpret_cast<const float4 *>(
                    &sw[ci * WSW + threadIdx.y * TILE + t]);
                wv[t + 0] = w4.x; wv[t + 1] = w4.y; wv[t + 2] = w4.z; wv[t + 3] = w4.w;
            }
            #pragma unroll
            for (int i = 0; i < TILE; ++i) {
                #pragma unroll
                for (int j = 0; j < C1_PXT; ++j) acc[i][j] += wv[i] * cur[j];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TILE; ++i) {
        const int oc = oc0 + threadIdx.y * TILE + i;
        if (oc >= c_out) continue;
        // Every store is bounds-checked, the first included. The last block of a
        // plane can own fewer than PX pixels, and an unguarded `p0` write lands in
        // the NEXT CHANNEL's plane - measured, not theoretical: without this the
        // whole engine went nondeterministic (two identical 96x96 runs differed in
        // 6651 of 9216 pixels) because that stray write races other blocks in the
        // same launch.
        #pragma unroll
        for (int j = 0; j < C1_PXT; ++j)
            if (p0 + j < (int)plane) out[(size_t)oc * plane + p0 + j] = acc[i][j];
    }
}

// 4 pixels x 4 output channels per thread: 128 pixels and 32 channels per block.
//
// This is the shape for c_out <= 32. It needs `plane % 4 == 0` for its float4
// input load, which the host checks rather than assumes - a misaligned float4 load
// returns the wrong four floats and does not fault, so the fallback below is not
// optional. Given that gate its guarded quad path is unreachable by construction
// and is kept anyway, because "the host checks it" is a property of the caller.
extern "C" __global__ void __launch_bounds__(256, 3) mx_conv1x1_t4(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd)
{
    c1_body<C1_TILE>(in, w, bias, out, c_in, c_out, h, wd);
}

// 4 pixels x 8 output channels per thread: 128 pixels and 64 channels per block.
//
// The shape for c_out >= 64, and the fastest kernel in this file at the model's
// own sizes - 2.1-2.4x the deleted `mx_conv1x1_t` (see the table above). Same
// precondition as `_t4` and chosen by the host under the same `plane % 4 == 0`
// test.
extern "C" __global__ void __launch_bounds__(256, 3) mx_conv1x1_t8(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd)
{
    c1_body<C1_TILES>(in, w, bias, out, c_in, c_out, h, wd);
}

// The SCALAR fallback, for a plane length the float4 forms cannot take
// (`plane % 4 != 0`). Its loads are per element and bounds-checked, so it is
// always correct; it is the kernel the tiled forms were measured against and it
// is 1.4-2.4x slower. It reads the same `[c_in][c_out]` weight with `[oc][ci]`
// staging, because this form's thread does not own four consecutive oc.
extern "C" __global__ void mx_conv1x1_t(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd)
{
    constexpr int TX = 32, TY = 8;
    constexpr int PX = TX * C1_PXT;     // 128 pixels per block
    constexpr int LOC = TY * C1_TILE;   // 32 output channels per block
    const size_t plane = (size_t)h * wd;
    const int p0 = blockIdx.x * PX + threadIdx.x * C1_PXT;
    const int oc0 = blockIdx.y * LOC;

    // [oc][ci] for the chunk, row-padded so the strided reads below spread banks.
    __shared__ float sw[LOC * (C1_CI + 1)];

    float acc[C1_TILE][C1_PXT];
    #pragma unroll
    for (int i = 0; i < C1_TILE; ++i) {
        const int oc = oc0 + threadIdx.y * C1_TILE + i;
        const float b = (oc < c_out && bias) ? bias[oc] : 0.0f;
        #pragma unroll
        for (int j = 0; j < C1_PXT; ++j) acc[i][j] = b;
    }

    for (int ic0 = 0; ic0 < c_in; ic0 += C1_CI) {
        // One element per thread per pass: 32*8 weights against 256 threads.
        for (int i = threadIdx.y * TX + threadIdx.x; i < LOC * C1_CI; i += TX * TY) {
            const int oc = i / C1_CI, ci = i % C1_CI;
            const int goc = oc0 + oc, gci = ic0 + ci;
            float v = 0.0f;
            if (goc < c_out && gci < c_in) v = w[(size_t)gci * c_out + goc];
            sw[oc * (C1_CI + 1) + ci] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int ci = 0; ci < C1_CI; ++ci) {
            // Bounds-checked per element: a block at the right edge of the plane
            // owns fewer than PX pixels, and reading past `plane` would leave the
            // channel. The pixels a thread cannot own contribute nothing, and its
            // stores are guarded below.
            float xv[C1_PXT];
            #pragma unroll
            for (int j = 0; j < C1_PXT; ++j) {
                const int p = p0 + j;
                xv[j] = (p < (int)plane) ? in[(size_t)(ic0 + ci) * plane + p] : 0.0f;
            }
            #pragma unroll
            for (int i = 0; i < C1_TILE; ++i) {
                const float wv = sw[(threadIdx.y * C1_TILE + i) * (C1_CI + 1) + ci];
                #pragma unroll
                for (int j = 0; j < C1_PXT; ++j) acc[i][j] += wv * xv[j];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < C1_TILE; ++i) {
        const int oc = oc0 + threadIdx.y * C1_TILE + i;
        if (oc >= c_out) continue;
        #pragma unroll
        for (int j = 0; j < C1_PXT; ++j)
            if (p0 + j < (int)plane) out[(size_t)oc * plane + p0 + j] = acc[i][j];
    }
}

// 4x4 stride 2 with flax's padding='SAME' - `Conv_down`, the UNet encoder's
// downsample. This is the register-tile form and it is now the ONLY one: both
// predecessors have been deleted and their numbers are kept below.
//
// NOT the toolkit's `lg_conv4x4s4`, which is a stride-4 patch embed with no
// padding: this halves the resolution and keeps ceil(h/2) rows, and it pads by
// (1,1) at the sizes this model uses (pad_top = pad_left = 1 here, since the
// caller has already made the shape a multiple of 64 and hence even).
//
// THE HISTORY, in reverse order. The original kernel gave each output element
// its own thread and summed over ci INNERMOST, so its 512 input loads per output
// each strided a whole input plane and reused nothing: 39.717 ms at 32->32
// @448x640, 59 GFLOP/s, about 118 GB/s of a card that does 256. Giving a thread
// four outputs instead of one measured 39.7 -> 36.96 ms (so the dependency chain
// was NOT the cost) and splitting the channel sum into eight independent partial
// accumulators measured 39.781 ms (nothing at all); the access pattern was the
// whole cost, and staging a 32x8 output tile in 40640 B of shared memory fixed
// it, 39.717 -> 3.832 ms, 59 -> 613 GFLOP/s, and in situ 177.9 -> 25.0 ms. That
// staged kernel is the second thing deleted here.
//
// WHAT REPLACED IT: the staged tile ran 2 blocks/SM (512 threads, 25% of the
// SM's warps) and the tile was buying less than the occupancy cost. Two things
// settled it. First, the bank conflict: the staged weight read
// `sw[(i*CD_CI + ci)*16 + ky*4 + kx]` has the four `i` addresses strided by
// CD_CI*16 = 128 floats, which is 0 mod 32 banks, so a thread's four weights came
// from ONE bank - the same mistake conv3x3 had before its transposed-weight
// landing. Restaging the weight as [ci][tap][oc] so the four are one aligned
// LDS.128 (`cdv2`) is BIT-IDENTICAL and exactly 1.000x, so the conflict was real
// and cost nothing. Second, and decisive: the same 32x8 geometry with the shared
// memory removed ENTIRELY and inputs and weights read from global (40 registers,
// no smem, still bit-identical) is 1.107x FASTER (3.455 vs 3.824 ms). The loads
// were being hidden by the tile at the price of the warps needed to hide them.
//
// So the family was swept as a search over work-per-thread and occupancy with the
// accumulation order fixed - `P` adjacent output columns x `C` output channels
// per thread, in the shipped order (bias, then passes of 8 input channels, then
// ky, then kx), flagging 36-70 generated entry points, all compared BITWISE
// against this kernel. Winner `P=4, C=4`, plain [oc][ci][ky][kx] weight read from
// global, no shared memory at all:
//
//     32->32   @448x640   3.863 -> 3.005 ms   608 -> 782 GFLOP/s   1.285x
//     64->64   @224x320   3.830 -> 2.092 ms   613 -> 1123 GFLOP/s  1.831x
//     128->128 @112x160   4.021 -> 1.588 ms   584 -> 1479 GFLOP/s  2.532x
//
// LAUNCH GEOMETRY, which differs per shape and is now chosen in exec_gpu.rs:
// block(16,4) grid(5,56,8) at 32->32, block(8,4) grid(5,28,16) at 64->64,
// block(8,4) grid(3,14,32) at 128->128. The rule behind it is "as many output
// columns per block as the register budget allows": blockDim.x * P = 64 columns
// (128 threads) at 32 channels and 32 columns (64 threads) at 64/128, with
// blockDim.y = 4 rows. ptxas spends 72-80 registers and there is no spill.
// Block shapes, not just P and C, matter - the same P/C at block(4,4) is 0.81x
// and at block(32,4) 1.15x - which is why the geometry is part of the result.
//
// LOSERS, all bit-identical, all measured interleaved at 32->32 @448x640 against
// this form as 1.00x: the smem-staged form it replaces 1.00x, the transposed
// in-smem form (`cdv2`) 1.000x, P=1 C=4 0.908x, P=2 C=4 0.896x, P=1 C=1 0.506x,
// C=16 0.84-0.99x, P=12 0.548x, P=32 0.196x, P=4 C=4 at block(4,4) 0.811x. The
// WEIGHT-LAYOUT axis is dead in this family: plain weight 1.285x beats the
// float4-per-ky form 1.187x and the transposed [ky][kx][ci][oc] form 1.132x, so
// unlike conv3x3 and conv1x1 this op needs no transposed twin and weights.rs is
// untouched by it.
//
// THE ONE THING THIS KERNEL DOES NOT DO is reproduce the ORIGINAL untiled form
// bit for bit, and the reason is structural rather than sloppy: the channel sum
// runs in passes of 8, so the additions come out as (pass), then ky, then kx,
// then the 8 channels inside the pass, where the deleted kernel summed ky, kx,
// then all channels. Same terms, same order within a pass, different grouping -
// against a host reference in the OLD order it reads 4.2e-05 worst on 58403 of
// 2293760 elements where the old kernel read 1.1e-05 on 30. That does NOT
// propagate to anything visible: measured PNG to PNG against the old kernel's
// output, 96x96 is byte-identical and 256x256 / 600x400 come out at 98.06 and
// 98.25 dB - one LSB on 0.001% of pixels, which is the final quantisation step
// rather than a difference in the image. THIS form is bit-identical to the
// staged form it replaces (0 differing of 2293760 at 32->32, and at every other
// shape), so that deviation is unchanged by this landing.
//
// Holding all c_in resident so the passes collapse into one would make it
// bit-exact with the original, but at 32 channels that needs 46 KB of shared
// memory for an 8x8 output tile, i.e. a tile too small to reuse anything - which
// is the trade this family refuses in both directions now.
constexpr int CD_PX = 4;      // output columns per thread
constexpr int CD_CO = 4;      // output channels per thread
constexpr int CD_CI = 8;      // input channels per K pass

extern "C" __global__ void mx_conv4x4s2(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int oh, int ow, int pad_top, int pad_left)
{
    const size_t in_plane = (size_t)h * wd;
    const int oc0 = blockIdx.z * CD_CO;
    const int oxb = (blockIdx.x * (int)blockDim.x + threadIdx.x) * CD_PX;
    const int oy = blockIdx.y * (int)blockDim.y + threadIdx.y;
    if (oy >= oh || oxb >= ow) return;

    float acc[CD_PX][CD_CO];
    #pragma unroll
    for (int p = 0; p < CD_PX; ++p)
        #pragma unroll
        for (int i = 0; i < CD_CO; ++i) {
            const int goc = oc0 + i;
            acc[p][i] = (bias && goc < c_out) ? bias[goc] : 0.0f;
        }

    // Four adjacent outputs share input columns, so one row of the input is read
    // once per ky for 2*CD_PX+2 values instead of 4*CD_PX - which is where the
    // loads per FMA come from, taken from 0.4 to about 0.32 by the reuse and to
    // 0.2 by the fact that no per-thread value is ever read twice.
    const int ixbase = 2 * oxb - pad_left;
    for (int ci0 = 0; ci0 < c_in; ci0 += CD_CI) {
        #pragma unroll
        for (int ciw = 0; ciw < CD_CI; ++ciw) {
            const int gci = ci0 + ciw;
            if (gci >= c_in) continue;
            #pragma unroll
            for (int ky = 0; ky < 4; ++ky) {
                const int iy = oy * 2 + ky - pad_top;
                float xv[2 * CD_PX + 2];
                const bool rowok = (iy >= 0 && iy < h);
                #pragma unroll
                for (int c = 0; c < 2 * CD_PX + 2; ++c) {
                    const int ix = ixbase + c;
                    xv[c] = (rowok && ix >= 0 && ix < wd)
                                ? in[(size_t)gci * in_plane + (size_t)iy * wd + ix] : 0.0f;
                }
                #pragma unroll
                for (int i = 0; i < CD_CO; ++i) {
                    const int goc = oc0 + i;
                    if (goc >= c_out) continue;
                    #pragma unroll
                    for (int kx = 0; kx < 4; ++kx) {
                        const float wv = w[((size_t)goc * c_in + gci) * 16 + ky * 4 + kx];
                        #pragma unroll
                        for (int p = 0; p < CD_PX; ++p) acc[p][i] += wv * xv[2 * p + kx];
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int p = 0; p < CD_PX; ++p) {
        const int ox = oxb + p;
        if (ox >= ow) continue;
        #pragma unroll
        for (int i = 0; i < CD_CO; ++i) {
            const int goc = oc0 + i;
            if (goc < c_out) out[((size_t)goc * oh + oy) * ow + ox] = acc[p][i];
        }
    }
}

// 2x2 stride 2 transposed conv - `ConvT_up` (UNetDecoderBlock, CrossGatingBlock).
//
// A scatter, so the weight is `[c_in][c_out][2][2]` and spatially REVERSED at
// load time (see `weights::convt2x2`): flax's ConvTranspose is the adjoint of its
// Conv, while a scatter reproduces torch's conv_transpose2d. Getting the flip
// wrong costs 65 dB (28 dB instead of 93 dB on a decoder block).
//
// out[oc][2y+ky][2x+kx] = bias[oc] + sum_ic in[ic][y][x] * w[ic][oc][ky][kx]
//
// One thread per INPUT position with CT_OC output channels in registers, where
// CT_OC is now a per-call constant (CD_C) rather than a #define. The form before
// that looped `oc` outermost and `ic` innermost, so it re-read the input value
// for every (oc, ic) pair and the weight with a stride of c_out*4 per ic - one
// MAC per two loads, which is the same shape of mistake `mx_conv1x1_t` was built
// to fix. Holding the input value once per `ic` and using it CT_OC times measures
// 8x: 128->64 @112x160 goes 12.023 -> 1.492 ms and 64->32 @224x320 11.682 ->
// 1.535 ms (97.7 -> 787 GFLOP/s), bit-identical at all three shapes the model
// uses because the accumulation order is untouched: `ic` ascending inside one
// (ky, kx) tap, and the four taps in (ky, kx) order.
//
// THAT version is what this file rewrote once, and it was still load-bound: per
// (ic, oc) it issues 1 input load + 4 weight loads against 4 FMAs, i.e. 1.03
// loads per FMA, where this SM issues 4 FMA warp instructions per cycle against
// one 32-bit load - the load pipe is oversubscribed 4x, which is why the family
// sat at 787 GFLOP/s (9.6% of peak). The fix is the same one that took convdown
// from 613 to 782 GFLOP/s: MORE WORK PER THREAD ALONG BOTH AXES.
//
//     P  = adjacent INPUT positions per thread (CD_P). Their 2P outputs are the
//          2P consecutive output columns 2x..2x+2P-1, and the WEIGHTS are shared
//          across them, which is what takes the weight loads out of the inner
//          loop. Loads per FMA: (P + 4C) / (4*P*C).
//     C  = output channels per thread (CD_C).
//
//     32->32   @448x640 out   0.835 -> 0.509 ms   703 -> 1153 GFLOP/s   1.640x
//     64->64   @224x320 out   0.799 -> 0.455 ms   735 -> 1289 GFLOP/s   1.755x
//     128->128 @112x160 out   0.696 -> 0.507 ms   844 -> 1159 GFLOP/s   1.374x
//
// measured interleaved, best of 7, against the CT_OC=8 form with the shipped
// shape as 1.00x, and BIT-IDENTICAL in every case (0 differing of 9175040 at
// 32->32; this kernel copies the shipped form's accumulation order exactly, so it
// is bit-exact rather than merely close).
//
// LOSERS, same sweep, same shapes: P=1 C=16 0.903-0.951x, P=2 C=16 0.689-0.903x,
// P=4 C=8 1.011-1.078x, P=4 C=4 1.275-1.382x, P=8 C=4 1.255-1.668x, and for P=2
// C=8 a narrow block_x is what costs: 0.590x at bx=8 against 1.64x at bx=64-96,
// because a 2P-wide window per thread needs several warps per row to fill a warp's
// load slots. The geometry is per shape and is chosen in exec_gpu.rs.
//
// The four tap accumulators are written out separately (acc[ky][kx][i][p]) rather
// than as one 2x2 loop so the compiler can keep all 4*C*P in registers.
constexpr int CD_P = 2;   // adjacent input positions per thread
constexpr int CD_C = 8;   // output channels per thread

extern "C" __global__ void mx_convt2x2s2(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd)
{
    const size_t in_plane = (size_t)h * wd;
    const int oh = h * 2, ow = wd * 2;
    const size_t out_plane = (size_t)oh * ow;
    const int oc0 = blockIdx.y * CD_C;
    const int xb = (blockIdx.x * (int)blockDim.x + threadIdx.x) * CD_P;
    const int y = blockIdx.z;
    if (y >= h || xb >= wd) return;

    float acc[4][CD_C][CD_P];
    #pragma unroll
    for (int k = 0; k < 4; ++k)
        #pragma unroll
        for (int i = 0; i < CD_C; ++i)
            #pragma unroll
            for (int p = 0; p < CD_P; ++p) {
                const int oc = oc0 + i;
                acc[k][i][p] = (bias && oc < c_out) ? bias[oc] : 0.0f;
            }

    for (int ic = 0; ic < c_in; ++ic) {
        float v[CD_P];
        #pragma unroll
        for (int p = 0; p < CD_P; ++p)
            v[p] = in[(size_t)ic * in_plane + (size_t)y * wd + xb + p];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int ky = k / 2, kx = k % 2;
            #pragma unroll
            for (int i = 0; i < CD_C; ++i) {
                const int oc = oc0 + i;
                if (oc >= c_out) continue;
                const float wv = w[(((size_t)ic * c_out + oc) * 2 + ky) * 2 + kx];
                #pragma unroll
                for (int p = 0; p < CD_P; ++p) acc[k][i][p] += wv * v[p];
            }
        }
    }

    #pragma unroll
    for (int ky = 0; ky < 2; ++ky)
        #pragma unroll
        for (int kx = 0; kx < 2; ++kx)
            #pragma unroll
            for (int i = 0; i < CD_C; ++i) {
                const int oc = oc0 + i;
                if (oc >= c_out) continue;
                #pragma unroll
                for (int p = 0; p < CD_P; ++p) {
                    const int x = xb + p;
                    if (x < wd)
                        out[(size_t)oc * out_plane + (size_t)(y * 2 + ky) * ow + (x * 2 + kx)] =
                            acc[ky * 2 + kx][i][p];
                }
            }
}

// The gated-MLP matmul as an implicit GEMM over a shared-memory tile. The
// version this replaced gave each output element its own thread, so every output
// re-read its whole input row and its whole weight row: at 512x512 that family
// was the largest single share of a 5.97 s run (2.13 s) at ~101 GFLOP/s,
// about 1% of this card's fp32 peak.
//
// Both modes are the same GEMM, which is why one body covers them:
//
//   mode 1: K = inner (contiguous in the activation), B = c*outer
//   mode 0: K = outer, B = c*inner - and there the activation's contiguous axis
//           is B, not K
//
// so `in` is the same matrix in the other storage order: [B][K] row-major in
// mode 1, and [K][B] (that is, [B][K] transposed) in mode 0. The weight is [A][K]
// with K contiguous in both, where A is the axis the Dense replaces. The tile is
// therefore FILLED differently per mode and READ back through one index helper,
// so the compute loop - and hence the accumulation order over K - is the same for
// both. A == K in this model (the gating Dense is square and replaces the axis it
// reduces), but the two are kept apart because that is what the indexing needs.
//
// Tile: 64 (A) x 64 (B), K in chunks of 16, 256 threads each accumulating a 4x4
// register block. Strides are padded (65 and 17, both odd) so that neither the
// stride-4 tile writes nor the stride-1 tile reads put two threads in one bank.
constexpr int MMA = 64;   // A tile
constexpr int MMB = 64;   // B tile
constexpr int MMK = 16;   // K chunk

// The (b, k) slot of the activation tile, per storage order.
template <int MODE0>
__device__ __forceinline__ int mm_sx(int b, int k) {
    return MODE0 ? (k * (MMB + 1) + b) : (b * (MMK + 1) + k);
}

template <int MODE0>
__device__ __forceinline__ void mm_body(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c, int outer, int inner)
{
    const int A = MODE0 ? outer : inner;
    const int K = MODE0 ? outer : inner;
    const int B = MODE0 ? c * inner : c * outer;

    const int a0 = blockIdx.x * MMA + threadIdx.y * 4;
    const int b0 = blockIdx.y * MMB + threadIdx.x * 4;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const size_t plane = (size_t)outer * inner;

    __shared__ float sx[MODE0 ? MMK * (MMB + 1) : MMB * (MMK + 1)];
    __shared__ float sw[MMA * (MMK + 1)];

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = (bias && a0 + i < A) ? bias[a0 + i] : 0.0f;
    }

    for (int k0 = 0; k0 < K; k0 += MMK) {
        // Fill both tiles. One float4 per thread, which is exactly the 256 a 64x16
        // patch needs: no per-thread loop, and every warp reads either 128
        // consecutive bytes (a whole row of a K-contiguous tile) or two rows of 64
        // (the transposed one) - never a stride long enough to split a transaction.
        //
        // An element outside the matrix is written as ZERO, not skipped: with K
        // chunked, a skipped slot would be read back holding the previous chunk's
        // value. The guard is therefore on the value, not the store.
        {
            const int n = tid;
            float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            // Scalar, guarded loads rather than one float4: a float4 needs its
            // address to be 16-byte aligned, and the address here is
            // `(r/inner)*plane + k*inner + r%inner`, which is only aligned when
            // `inner` is a multiple of 4. The 640x448 eval size has grid cells
            // that are not (the square power-of-two test sizes all are, which is
            // why this read as a working kernel until it met a real image), and a
            // misaligned vector load is an address error at launch rather than a
            // wrong answer. The fill is a couple of percent of this kernel's
            // work, so a vector load here is not worth that failure mode; the
            // WEIGHT tile below keeps its float4 only where the arithmetic proves
            // the address is aligned.
            if (MODE0) {
                // 16 K rows of 16 float4s each, one float4 per thread: row =
                // n / 16, column = n % 16. (Dividing by the ROW LENGTH in floats
                // - MMB - instead would give only 4 distinct rows and read the
                // same 4 rows of every chunk: the first version of this did, and
                // it is the sort of error that looks like a bad transpose.)
                const int k = k0 + n / (MMB / 4);
                const int b = (n % (MMB / 4)) * 4;
                // The tile column, NOT `b0 + b`: the fill enumerates all 64
                // columns of the tile, while `b0` is only this thread's own
                // output column.
                const int r = blockIdx.y * MMB + b;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (k < K && r + j < B) {
                        // A channel boundary can fall inside these four columns
                        // (inner need not be a multiple of 4), which is why the
                        // B index is decomposed per element rather than once.
                        const int rb = r + j;
                        v[j] = in[(size_t)(rb / inner) * plane + (size_t)k * inner + (rb % inner)];
                    }
                }
                // The tile row is the K index WITHIN this chunk, `n / 16`, not
                // `n / MMB`: the latter is the float index divided by the row
                // length in floats and lands every thread in rows 0..3.
                #pragma unroll
                for (int j = 0; j < 4; ++j) sx[mm_sx<MODE0>(b + j, n / (MMB / 4))] = v[j];
            } else {
                // 64 B rows of 16 K values; here the row index is B.
                const int b = n / (MMK / 4);
                const int k = (n % (MMK / 4)) * 4;
                const int r = blockIdx.y * MMB + b;
                if (r < B) {
                    // `inner` is the contiguous axis here, so these four K are
                    // contiguous - but the base is `(r % outer) * inner + k0 + k`
                    // and `inner` is not necessarily a multiple of 4.
                    const size_t base =
                        (size_t)(r / outer) * plane + (size_t)(r % outer) * inner + k0 + k;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (k0 + k + j < K) v[j] = in[base + j];
                    }
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) sx[mm_sx<MODE0>(b, k + j)] = v[j];
            }
        }
        // The weight tile, [A][K] with K contiguous.
        {
            const int n = tid;
            const int a = n / (MMK / 4);
            const int k = (n % (MMK / 4)) * 4;
            const int ar = blockIdx.x * MMA + a;
            float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (ar < A) {
                // `w + ar*K` is 16-byte aligned only when K is a multiple of 4,
                // and the K tail here is not necessarily full, so this one is
                // scalar too. It is a 64x16 patch per chunk against 64x64x16
                // multiply-accumulates, so the cost of not vectorising it is
                // noise next to the risk of an unaligned load.
                const size_t base = (size_t)ar * K + k0 + k;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (k0 + k + j < K) v[j] = w[base + j];
                }
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) sw[a * (MMK + 1) + k + j] = v[j];
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < MMK; ++k) {
            float xv[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) xv[j] = sx[mm_sx<MODE0>(tx * 4 + j, k)];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float wv = sw[(ty * 4 + i) * (MMK + 1) + k];
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += wv * xv[j];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int a = a0 + i, b = b0 + j;
            if (a < A && b < B) {
                const int ch = MODE0 ? b / inner : b / outer;
                const int bl = MODE0 ? b % inner : b % outer;
                out[(size_t)ch * plane + (MODE0 ? (size_t)a * inner + bl : (size_t)bl * inner + a)] =
                    acc[i][j];
            }
        }
    }
}

// mode 0: the Dense reduces over the activation's `outer` axis.
extern "C" __global__ void mx_gate_mm_t0(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c, int outer, int inner)
{
    mm_body<1>(in, w, bias, out, c, outer, inner);
}

// mode 1: it reduces over `inner`, the contiguous axis.
extern "C" __global__ void mx_gate_mm_t1(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c, int outer, int inner)
{
    mm_body<0>(in, w, bias, out, c, outer, inner);
}

// Bilinear resize, jax's rule.
//
// jax's `image.resize` builds a weight matrix with
//   sample_f[i] = (i + 0.5) * inv_scale - 0.5,      inv_scale = in/out
//   kernel_scale = max(inv_scale, 1)
//   w[i][j] = triangle(|sample_f[i] - j| / kernel_scale), masked to
//            sample_f in [-0.5, in - 0.5] and renormalised to sum 1
// i.e. align_corners=False with a triangle that WIDENS when downsampling, which
// is the antialiasing torch only applies with `antialias=True`. The mask drops
// the weights of out-of-range taps and the survivors are renormalised, so the
// edge is a clamped, slightly asymmetric interpolation. A tap is worth summing
// exactly when `|sample - j| < kernel_scale`, which is why the loop's bounds are
// `ceil(sample - kernel_scale)` and `floor(sample + kernel_scale)`: the endpoints
// have weight 0 and are skipped by the `d >= 1` test.
//
// `axis` 2 = horizontal, 1 = vertical. One thread per output ELEMENT of the pass
// it is running, so each thread computes the tap range for its own single output
// and they all share the few source values a warp needs. The horizontal pass
// leaves the plane the target width, so the caller runs it first and the vertical
// pass second.
//
// THE FAST AXIS IS PICKED PER DIRECTION, and that is the second half of this
// kernel's history. It used to make the OUTPUT ROW the fast axis in both
// directions, which is right for the horizontal pass and backwards for the
// vertical one: there the free index IS the output row, so consecutive threads
// read `src[j*win + g]` with `g` fixed - 32 reads a whole temp row apart, 2560
// bytes at the eval size - and write `dst[o*wout + g]` the same way. 57 MB of
// traffic in 4.2 ms, about 13 GB/s. A column is what is contiguous in both
// operands, so now axis 1 holds the output row in the SLOW index (on blockIdx.y)
// and lets a warp walk the column, and the slow index also takes two of the four
// integer divisions out of the per-element path.
//
// Measured at the eval size, c = 32, against a host reference in jax's own
// order, output identical to the element at every pass (0 differing of
// 2293760 / 573440 / 2293760 / 9175040):
//   down H 448x640 -> 448x160  0.388 -> 0.304 ms (1.28x)  213 -> 272 GB/s
//   down V 448x160 -> 112x160  0.168 -> 0.080 ms (2.11x)  123 -> 259 GB/s
//   up   H 112x160 -> 112x640  0.149 -> 0.111 ms (1.35x)  184 -> 248 GB/s
//   up   V 112x640 -> 448x640  0.899 -> 0.469 ms (1.92x)  122 -> 235 GB/s
// (GB/s is effective: the taps a warp reads plus the store.) The one-thread-per-
// (c, row) shape this superseded was 14x slower still in the DOWN direction
// (13.676 ms against 0.916 at 448x640 -> 224x320, bit-identical output) because
// it made each thread walk a whole output row's tap range: latency, not
// bandwidth, was that cost.
//
// The 64x4 block is chosen for the heavy pass; a sweep of 32x8, 64x4, 128x2,
// 128x4 and 256x1 is flat within the run-to-run spread at all four passes, so no
// block shape is evidence of anything except that a 2-D block beats a flat 256 on
// a pass whose fast axis is 160 wide.
// `in_plane`/`out_plane` are the number of elements in one CHANNEL of each
// operand, and they are arguments rather than derivations on purpose: the two
// passes write different planes. The horizontal pass reads `hin x win` and writes
// `hin x wout` (the caller's temp), the vertical pass reads that temp and writes
// `hout x wout`. Deriving the output plane from `hout*wout` is right for the
// second pass and wrong for the first, where it strided channels by
// `hout*wout` instead of `hin*wout` and displaced every channel but the first.
extern "C" __global__ void mx_resize_axis(
    const float *__restrict__ in, float *__restrict__ out,
    int c, int hin, int win, int hout, int wout, int in_plane, int out_plane, int axis)
{
    const int rows = (axis == 2) ? hin : win;
    const int n_in = (axis == 2) ? win : hin;
    const int n_out = (axis == 2) ? wout : hout;
    // Which index runs FAST - and it is chosen per direction, not fixed. See the
    // comment above for the 1.3-2.1x this is worth and why.
    //
    //   axis 2 (horizontal): the output COLUMN is the fast axis; the slow index
    //           enumerates (channel, row).
    //   axis 1 (vertical): the COLUMN is the fast axis and the output ROW is the
    //           slow one, because a column is what is contiguous in both operands
    //           (the temp and the destination are both row-major). This is the
    //           direction the old shape got backwards twice over.
    const int fast_n = (axis == 2) ? n_out : rows;
    const int slow_n = (axis == 2) ? c * hin : c * hout;
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    // (z, y) FLATTENED, for the same reason conv3x3 does it: gridDim.y is capped
    // at 65535 and the horizontal pass of a 2048x2048 image wants
    // c * hin / 4 = 65536 blocks on the slow axis. This kernel only ever uses `s`
    // as an index and guards it, so the split needs no exactness - any
    // y * z * blockDim.y >= slow_n will do, and with z == 0 the index is the one
    // the flat grid produced.
    const int s = (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y + threadIdx.y;
    if (f >= fast_n || s >= slow_n) return;
    int ch, g, o;
    if (axis == 2) { ch = s / rows; g = s - ch * rows; o = f; }
    else           { ch = s / n_out; o = s - ch * n_out; g = f; }
    const float inv_scale = (float)n_in / (float)n_out;
    const float kscale = inv_scale > 1.0f ? inv_scale : 1.0f;
    const float sample = ((float)o + 0.5f) * inv_scale - 0.5f;
    int j0 = (int)ceilf(sample - kscale);
    int j1 = (int)floorf(sample + kscale);
    if (j0 < 0) j0 = 0;
    if (j1 > n_in - 1) j1 = n_in - 1;
    const float *src = in + (size_t)ch * in_plane;
    float *dst = out + (size_t)ch * out_plane;
    float wsum = 0.0f;
    float acc = 0.0f;
    // The same taps and the same order as before, only one output's worth of
    // them: the arithmetic per output element is untouched, which is why the
    // down direction is bit-identical rather than merely close.
    for (int j = j0; j <= j1; ++j) {
        const float d = fabsf(sample - (float)j) / kscale;
        if (d >= 1.0f) continue;
        const float wv = 1.0f - d;
        wsum += wv;
        acc += wv * ((axis == 2) ? src[g * win + j] : src[(size_t)j * win + g]);
    }
    if (wsum > 0.0f) acc /= wsum;
    if (axis == 2) dst[g * wout + o] = acc;
    else dst[(size_t)o * wout + g] = acc;
}

// Space <-> block permutation for the gMLP layers.
//
// For an (h, w) plane tiled into (gh, gw) blocks of (fh, fw) pixels:
//   g = gy * gw + gx      (the "grid" axis)
//   p = fy * fw + fx      (the "patch" axis)
//   pixel (y, x) = (gy * fh + fy, gx * fw + fx)
//
// `forward` gathers NCHW -> blocked, otherwise it scatters blocked -> NCHW. Both
// directions derive ONE index from the thread's own output element and read the
// other operand with it, so there is a single index derivation and no risk of the
// two directions drifting apart.
// `swap` transposes the two block axes, giving the (patch, grid) order: the
// multi-scale `signal` is concatenated over UpSampleRatio outputs whose own grid
// size changes with the scale, so both orders are needed and one kernel with a
// flag keeps the relationship visible.
//
//   forward, !swap: out[c][g][p] = in[y][x]     (block_images)
//   forward,  swap: out[c][p][g] = in[y][x]
//   !forward,!swap: out[y][x] = in[c][g][p]     (unblock_images)
//   !forward, swap: out[y][x] = in[c][p][g]
//
// THE INDEX IS THE OUTPUT'S, and that is the whole reason this kernel is worth its
// own comment. It used to enumerate (c, g, p) - the *blocked* tensor in all four
// cases - and derive the pixel from it, which cost about eight integer divisions
// per element (idx%psz, /psz, %gsz, /gsz, p/fw, p%fw, g/gw, g%gw) and made the
// STORE the gathered side whenever the blocked tensor was the source. Now the
// thread index enumerates the contiguous side of the OUTPUT, so the store is a
// linear write with no index arithmetic at all, the derivation drops to two
// divisions plus two remainders of small numbers, and one thread covers all `c`
// channels of its element - which is also what turns c separate streams into a
// single loop the memory system keeps busy.
//
// This was measured before it was believed, against a host reference, on every one
// of the 14 (gh, gw, fh, fw, swap, direction) combinations this model builds, at
// c = 32, output identical to the element (0 differing of 9175040 / 2293760 /
// 573440):
//     448x640  28x40 16x16 fwd  0.727 -> 0.321 ms   101 -> 229 GB/s
//     448x640  28x40 16x16 bwd  0.663 -> 0.322 ms   111 -> 228
//     448x640  16x16 28x40 fwd  0.637 -> 0.349      115 -> 210
//     448x640  16x16 28x40 bwd  0.641 -> 0.329      114 -> 223
//     224x320  14x20 16x16 fwd  0.161 -> 0.082      114 -> 224
//     224x320  16x16 14x20 fwd  0.161 -> 0.089      114 -> 207
//     112x160  14x20  8x8  fwd  0.042 -> 0.026      110 -> 179
// 229 GB/s is the achievable rate on this card (add/gelu measure 236-241), so this
// family has nothing left in it beyond not moving the data at all.
//
// THE VARIANT THAT LOST, and why it is not here: staging the gathered read through
// a 33-float-per-warp shared tile, so the read is coalesced and the transpose
// happens in shared memory. It is correct (the same 0 differing elements) and it
// measured 0.353 ms against this kernel's 0.321 at 448x640, 0.0874 against 0.0819
// at 224x320, 0.0289 against 0.0257 at 112x160 - i.e. no better anywhere, and
// worse on some, because the gather was never the problem: the L2 already
// coalesces the 32-byte sectors a warp's scattered feet land in, and the staging
// only added a barrier per channel. The index arithmetic and the uncoalesced
// store were the cost, and both are gone in this form.
//
// A block size sweep (128 / 256 / 512) changed nothing: 0.3203 / 0.3223 / 0.3195 ms
// at 448x640, so the launch keeps the engine's standard BLOCK.
extern "C" __global__ void mx_block_perm(
    const float *__restrict__ in, float *__restrict__ out,
    int c, int h, int wd, int gh, int gw, int fh, int fw, int swap, int forward)
{
    const int gsz = gh * gw, psz = fh * fw;
    const int tot = gsz * psz;
    const int ID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ID >= tot) return;
    // The blocked index, decomposed. `ID` IS that index (the output when the
    // blocked tensor is the destination, the source when it is not), which is why
    // the two cases are the two arms below and not two different walks.
    int g, p;
    if (swap) { g = ID % gsz; p = ID / gsz; }
    else      { p = ID % psz; g = ID / psz; }
    const int fy = p / fw, fx = p - fy * fw;
    const int gy = g / gw, gx = g - gy * gw;
    const size_t plane = (size_t)h * wd;
    const size_t pix = (size_t)(gy * fh + fy) * wd + (gx * fw + fx);
    if (forward) {
        #pragma unroll 1
        for (int ch = 0; ch < c; ++ch)
            out[(size_t)ch * plane + ID] = in[(size_t)ch * plane + pix];
    } else {
        #pragma unroll 1
        for (int ch = 0; ch < c; ++ch)
            out[(size_t)ch * plane + pix] = in[(size_t)ch * plane + ID];
    }
}
// LayerNorm over the CHANNEL axis of an NCHW tensor, one thread per spatial
// position - this engine's replacement for the toolkit's
// `lg_channel_layer_norm`, which is the same op with a different thread mapping.
//
// The toolkit kernel runs one BLOCK per spatial position, so at the eval size it
// launches 286720 blocks, and at c=32 each block has 224 of its 256 threads
// accumulate nothing before an 8-step halving tree over 1024 slots collapses 32
// live values. Measured at the four shapes this model uses, one thread per pixel
// is 10.0x (32x286720: 4.697 -> 0.468 ms), 4.5x (64x71680), 3.0x (128x17920) and
// 8.3x (32x16384), at 216-235 GB/s effective against 23-77.
//
// THE DEVIATION, stated where it matters: the toolkit kernel reduces with its
// halving tree and this one sums serially in ascending channel order, so the two
// agree only to about 7e-07 (measured: 2030138 of 9175040 elements at 32x286720,
// none larger than 4.77e-07 - last-bit rounding, not a different result). That is
// NOT a break of a documented contract: `lg_channel_layer_norm`'s own comment
// says nothing about its summation order. The toolkit's order CONTRACT belongs to
// `lg_channel_mean`, whose doc had to be rewritten once because a copy of it did
// not produce the sum its comment claimed. If a `lg_channel_layer_norm` caller
// ever needs bit-comparability with the toolkit, this kernel is the wrong choice
// for that caller and the toolkit one is still linked.
//
// THE TRAFFIC, and it is the reason for the `chln_reg<N>` forms below. This kernel
// reads x TWICE - once to accumulate the statistics, once to normalise - and writes
// y once, so it moves 3c floats per pixel where the op needs 2c. It runs at
// 224-235 GB/s, which is the achievable rate on this card, which means there is
// nothing left to win in this kernel's *schedule*: the only lever is the traffic,
// and the only way to stop reading x twice at global-memory rates is to keep the
// pixel's channels somewhere. Measured, 0 differing elements against a host
// reference in this kernel's own ascending channel order at all four shapes:
//   32x286720  0.467 -> 0.319 ms   64x71680  0.246 -> 0.158
//   128x17920  0.121 -> 0.079      128x4480  0.035 -> 0.020
// i.e. 1.46-1.72x, moving 2c+1 streams instead of 3c+1 (73.4 MB instead of 110) at
// the same 229 GB/s.
//
// WHERE the pixel's channels are kept is the whole finding, and the obvious answer
// was wrong: staging them in SHARED memory is a LOSER. It costs c*blockDim.x*4
// bytes per block - 65536 at c=64, 131072 at c=128, both over the 48 KB default,
// so those two shapes cannot use it at all at 256 threads - and it adds a shared
// round trip per element, and it measured 0.463 against this kernel's 0.467 at
// c=32 (nothing), 0.331 against 0.246 at 64x71680, 0.102 against 0.035 at
// 128x4480. REGISTERS cost nothing but the register file, and a compile-time `c`
// is what makes them addressable.
extern "C" __global__ void mx_chan_ln(
    const float *__restrict__ x, const float *__restrict__ w, const float *__restrict__ b,
    float *__restrict__ y, int c, int hw, float eps)
{
    const long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= hw) return;
    // Ascending channel order, serially: the same order the CPU twin uses, which
    // is what makes this kernel's output stable run to run and reproducible
    // against `--device cpu` rather than merely close to it.
    float s1 = 0.0f, s2 = 0.0f;
    for (int i = 0; i < c; ++i) {
        const float v = x[(size_t)i * hw + p];
        s1 += v;
        s2 += v * v;
    }
    const float n = (float)c;
    const float mean = s1 / n;
    const float var = s2 / n - mean * mean;
    const float scale = rsqrtf(fmaxf(var, 0.0f) + eps);
    for (int i = 0; i < c; ++i) {
        const size_t o = (size_t)i * hw + p;
        y[o] = (x[o] - mean) * scale * w[i] + b[i];
    }
}

// The register form: the pixel's channels live in registers instead of being
// re-read, so x is read ONCE and the traffic is 2c + 1 streams rather than 3c + 1.
// `C` is the channel count at compile time, which is what makes a register array
// addressable; the host picks the instantiation by `c` and falls back to
// `mx_chan_ln` above for any count outside these three (the plan only builds 32,
// 64 and 128, so the fallback is a safety net rather than a live path - the same
// arrangement as `mx_conv1x1_t` behind `mx_conv1x1_t4`).
//
// The arithmetic is IDENTICAL, term for term, to `mx_chan_ln`: `s1 += v` and
// `s2 += v*v` in ascending channel order, then `var = s2/n - mean*mean`, then the
// same `rsqrtf(fmaxf(var,0)+eps)` and the same store expression. That is why the
// two agree element for element rather than merely closely - which matters here
// because this is the one op whose summation order is a deliberate contract
// against `--device cpu`, so a "faster" version that regrouped the sum would be a
// different op.
//
// ptxas: 48 / 80 / 168 registers for C = 32 / 64 / 128 at 256 threads. The 168 is
// one block per SM on sm_61 (12.5% occupancy), and it was worth sweeping the block
// size to check that this does not matter: 64 / 128 / 256 / 512 threads at
// 128x17920 gives 0.0819 / 0.0820 / 0.0792 / (512 does not fit - "too many
// resources"), and at 32x286720 0.3226 / 0.3236 / 0.3192 / 0.3161. Flat, i.e.
// bandwidth-bound as the traffic accounting already said. The launch keeps 256.
template <int C>
__device__ __forceinline__ void chln_body(
    const float *__restrict__ x, const float *__restrict__ w,
    const float *__restrict__ b, float *__restrict__ y, int hw, float eps)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    float v[C];
    float s1 = 0.0f, s2 = 0.0f;
    if (p < hw) {
        #pragma unroll
        for (int i = 0; i < C; ++i) {
            v[i] = x[(size_t)i * hw + p];
            s1 += v[i];
            s2 += v[i] * v[i];
        }
    }
    const float n = (float)C;
    const float mean = s1 / n;
    const float var = s2 / n - mean * mean;
    const float scale = rsqrtf(fmaxf(var, 0.0f) + eps);
    if (p >= hw) return;
    #pragma unroll
    for (int i = 0; i < C; ++i)
        y[(size_t)i * hw + p] = (v[i] - mean) * scale * w[i] + b[i];
}

extern "C" __global__ void mx_chan_ln_c32(
    const float *__restrict__ x, const float *__restrict__ w, const float *__restrict__ b,
    float *__restrict__ y, int c, int hw, float eps)
{
    chln_body<32>(x, w, b, y, hw, eps);
}

extern "C" __global__ void mx_chan_ln_c64(
    const float *__restrict__ x, const float *__restrict__ w, const float *__restrict__ b,
    float *__restrict__ y, int c, int hw, float eps)
{
    chln_body<64>(x, w, b, y, hw, eps);
}

extern "C" __global__ void mx_chan_ln_c128(
    const float *__restrict__ x, const float *__restrict__ w, const float *__restrict__ b,
    float *__restrict__ y, int c, int hw, float eps)
{
    chln_body<128>(x, w, b, y, hw, eps);
}

// Four kernels that used to be here are now the toolkit's, because they were
// never MAXIM-specific:
//   * `mx_sigmoid`   -> `lg_sigmoid`. The comment on it claimed the toolkit had
//                       no standalone sigmoid "because its transformer engines
//                       fold it into an attention kernel", which was simply
//                       false. One difference to know about: this engine used
//                       `expf` and the toolkit's `lg_sigmoid` uses `__expf`, so
//                       the CALayer's excite step is now the fast-math form.
//   * `mx_channel_mean` -> `lg_channel_mean`. Same op (the CALayer's global
//                       average pool); the toolkit version reduces with its own
//                       static shared scratch instead of one thread per channel
//                       with a serial loop.
//   * `mx_channel_scale` -> `lg_channel_scale`. Same op, and the toolkit has it
//                       as the `shift = null` case of `lg_channel_affine`.
//   * `mx_mul`       -> `lg_mul`. Same op (an elementwise product, `long` length
//                       in both). It is promoted rather than merely shared for
//                       the reason in `docs/MAINTAINING.md`: NAFNet's
//                       SimpleGate and every other gated architecture needs a
//                       plain multiply too, so the toolkit is where it belongs.

// The gated-MLP gate: y = u * (v + 1). This is the same form in all four places
// it appears (GridGatingUnit, BlockGatingUnit, and both halves of
// GetSpatialGatingWeights' weight branch is NOT one of them - that one has no
// +1), and it is not `lg_add` + `lg_mul` by accident of arithmetic: the +1 is
// part of the gMLP definition and is what keeps the gate near the identity.
extern "C" __global__ void mx_gate_apply(
    const float *__restrict__ u, const float *__restrict__ v, float *__restrict__ y, long n)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = u[i] * (v[i] + 1.0f);
}

// Nearest-neighbour subsample with an explicit stride and offset:
//   out[y][x] = in[y*stride + offset][x*stride + offset]
//
// This is jax's `image.resize(..., method='nearest')` rule, not torch's: jax maps
// output i to input floor((i + 0.5) * m / n), so halving selects 1, 3, 5, ... and
// quartering selects 2, 6, 10, ... . Both are needed for the model's multi-scale
// input pyramid, which is why the offset is a parameter and not a hardcoded 1 -
// quartering by running the halving kernel twice would select 3, 7, 11, ... .
extern "C" __global__ void mx_down2(
    const float *__restrict__ in, float *__restrict__ out,
    int c, int h, int wd, int stride, int offset, int oh, int ow)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c * oh * ow;
    if (idx >= total) return;
    const int x = (int)(idx % ow);
    const long t = idx / ow;
    const int y = (int)(t % oh);
    const int ch = (int)(t / oh);
    const int iy = y * stride + offset;
    const int ix = x * stride + offset;
    out[idx] = in[(size_t)ch * h * wd + (size_t)iy * wd + ix];
}

