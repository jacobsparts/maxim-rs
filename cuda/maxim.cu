// MAXIM's own kernels.
//
// These are the ops the lightgpu toolkit does not already have: the two strided
// convs flax's `Conv_down`/`ConvT_up` need, the gated-MLP matmul, the bilinear
// resize with jax's antialiasing rule, the block/grid permutation, and an
// elementwise multiply (the toolkit keeps `lg_add` but no multiply, since none of
// the transformer engines needed one).
//
// Everything here is NCHW, except `mx_gate_mm` and `mx_block_perm`, which work on
// the *blocked* layout MAXIM's gMLP layers are written in.
//
// Several of these exist as TILED versions of a toolkit kernel (`mx_conv1x1_t`
// for `lg_conv1x1`, `mx_conv3x3_t2`/`_t4` for `lg_conv3x3s1p1`, `mx_gate_mm_t0`/
// `_t1` for `mx_gate_mm`). The originals are all still linked and still launched
// with `--legacy-ops`, which is how each speedup below is measured rather than
// asserted: the tiled form shares one load across several outputs, and the
// originals gave every output element its own thread and re-read everything.

// 3x3 pad-1 conv, tiled: this engine's replacement for the toolkit's
// `lg_conv3x3s1p1`, which the model still links for the legacy path.
//
// That kernel gives each output element its own thread, so per output it issues
// 9*c_in loads of the input and 9*c_in of the weight - about 1 MAC per 8 bytes
// moved - and at 512x512 this family is the largest remaining share of the run
// (1.90 s of 4.32 s, ~1.3 TFLOP/s against this card's 8.9). Here a 64-pixel block
// of outputs is computed from a shared-memory tile, so one input load feeds 64
// outputs and one weight load 64 more: ~47 MACs per load.
//
// The pixel block is a SEGMENT OF ONE ROW, which is what makes the halo cheap: 64
// outputs need 66 input columns per (ci, row), so a 3x3 window is three offsets
// into the same tile rather than three separate loads. `wd % 64 == 0` is required
// (the host falls back to the toolkit kernel otherwise) and holds for every
// feature map this model builds, because the plan pads the input to a multiple of
// 64 and every level halves from there.
//
// `PXT` (pixels per thread, 4 or 2) picks the block shape and hence the number of
// output channels per block: 4 -> 16x16 threads and 64 channels, 2 -> 32x8 and 32.
// The host picks by `c_out` so that a 32-channel layer does not run with half the
// block idle.
//
// Accumulation order is dy, dx, ci - the same as the toolkit kernel and the CPU
// twin, which makes the tiled and legacy forms agree bit-for-bit rather than
// merely within tolerance.
constexpr int C3_CI = 8;   // input channels per K chunk

template <int PXT>
__device__ __forceinline__ void c3_body(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd)
{
    constexpr int TX = 64 / PXT;            // threads along x
    constexpr int TY = 256 / TX;            // threads along y
    constexpr int LOC = TY * 4;             // output channels per block
    constexpr int PX = 64;                  // output pixels per block
    constexpr int XW = PX + 2;              // input columns: 64 outputs + halo
    constexpr int XP = XW + 1;              // padded row stride (odd)

    const int oc0 = blockIdx.x * LOC;
    const int per_row = wd / PX;
    const int y0 = blockIdx.y / per_row;
    const int x0 = (blockIdx.y % per_row) * PX;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const size_t plane = (size_t)h * wd;

    // [oc][(dy*3 + dx)*C3_CI + ci], row-padded so the strided reads below do not
    // put two threads in the same bank.
    __shared__ float sw[LOC * (C3_CI * 9 + 1)];
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
        // Weight tile, one element per thread per pass.
        for (int i = tid; i < LOC * C3_CI * 9; i += 256) {
            const int oc = i / (C3_CI * 9);
            const int t = i % (C3_CI * 9);
            const int dy = t / (3 * C3_CI), dx = (t / C3_CI) % 3, ci = t % C3_CI;
            const int goc = oc0 + oc, gci = ci0 + ci;
            float v = 0.0f;
            if (goc < c_out && gci < c_in) v = w[((size_t)goc * c_in + gci) * 9 + dy * 3 + dx];
            sw[oc * (C3_CI * 9 + 1) + t] = v;
        }
        // Input tile, halo included; out-of-range reads are zeros, which is the
        // same arithmetic as the toolkit kernel's `continue` on a bounds test.
        for (int i = tid; i < C3_CI * 3 * XW; i += 256) {
            const int ci = i / (3 * XW);
            const int r = i % (3 * XW);
            const int dy = r / XW, col = r % XW;
            const int iy = y0 + dy - 1, ix = x0 + col - 1;
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
                    float wv[4];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        wv[i] = sw[(ty * 4 + i) * (C3_CI * 9 + 1) + (dy * 3 + dx) * C3_CI + ci];
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
        const size_t row = (size_t)(oc0 + ty * 4 + i) * plane + (size_t)y0 * wd + x0 + tx * PXT;
        #pragma unroll
        for (int j = 0; j < PXT; ++j) out[row + j] = acc[i][j];
    }
}

// 4 pixels per thread: 64 output channels per block, 16x16 threads.
extern "C" __global__ void mx_conv3x3_t4(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd)
{
    c3_body<4>(in, w, bias, out, c_in, c_out, h, wd);
}

// 2 pixels per thread: 32 channels per block, 32x8 threads, for c_out < 64.
extern "C" __global__ void mx_conv3x3_t2(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd)
{
    c3_body<2>(in, w, bias, out, c_in, c_out, h, wd);
}

// 1x1 conv over a TRANSPOSED weight, `[c_in][c_out]` - this engine's own
// replacement for the toolkit's `lg_conv1x1`, which the model still links.
//
// The profile made the reason concrete: at 512x512 the gMLP's 1x1 convs hold 30%
// of the run, and `lg_conv1x1` was moving about 26x the traffic the op needs. It
// puts one thread on each output element, so a thread walks its input column one
// `plane`-strided float at a time and the block touches a 32-byte sector per tap
// to use 4 of its 8 floats - and because the weight is `[c_out][c_in]`, the
// alternative of giving each thread several output channels would stride c_in
// through the weight on every step. Both halves of that are fixed here: a
// transposed weight makes consecutive outputs consecutive, and the input column
// is then shared by TILE outputs instead of re-read by each.
//
// `wd` is the contiguous axis, so consecutive threads read consecutive inputs
// (coalesced) and the load is issued once outside the output loop, which the
// compiler hoists and keeps in registers under TILE=4.
extern "C" __global__ void mx_conv1x1_t(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd)
{
    // Not a hard requirement: the c_out remainder is handled per thread below.
    // It only has to divide the fixed-width array, which it does on any driver
    // this builds for.
    constexpr int TILE = 4;
    const size_t plane = (size_t)h * wd;
    const long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= (long)plane) return;
    int oc0 = blockIdx.y * TILE;

    float acc[TILE];
    #pragma unroll
    for (int j = 0; j < TILE; ++j) {
        const int oc = oc0 + j;
        acc[j] = (oc < c_out && bias) ? bias[oc] : 0.0f;
    }
    for (int ic = 0; ic < c_in; ++ic) {
        const float v = in[(size_t)ic * plane + (size_t)p];
        const float *wp = w + (size_t)ic * c_out + oc0;
        #pragma unroll
        for (int j = 0; j < TILE; ++j) {
            // Deviates from the toolkit's (oc, ic) accumulation order, where
            // that kernel's inner `if (wv == 0.0f) continue` makes the order
            // data-dependent anyway. Recorded rather than chased: both are f32
            // sums of the same products and the op-by-op check bounds the
            // difference against the buffer, not against exact equality.
            if (oc0 + j < c_out) acc[j] += wp[j] * v;
        }
    }
    #pragma unroll
    for (int j = 0; j < TILE; ++j) {
        const int oc = oc0 + j;
        if (oc < c_out) out[(long)oc * (long)plane + p] = acc[j];
    }
}

// 4x4 stride 2 with flax's padding='SAME' - `Conv_down`, the UNet encoder's
// downsample.
//
// NOT the toolkit's `lg_conv4x4s4`, which is a stride-4 patch embed with no
// padding: this halves the resolution and keeps ceil(h/2) rows, and it pads by
// (1,1) at the sizes this model uses (pad_top = pad_left = 1 here, since the
// caller has already made the shape a multiple of 64 and hence even).
extern "C" __global__ void mx_conv4x4s2(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int oh, int ow, int pad_top, int pad_left)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c_out * oh * ow;
    if (idx >= total) return;
    const int ox = (int)(idx % ow);
    const long t = idx / ow;
    const int oy = (int)(t % oh);
    const int oc = (int)(t / oh);
    const size_t in_plane = (size_t)h * wd;
    float acc = bias ? bias[oc] : 0.0f;
    for (int ky = 0; ky < 4; ++ky) {
        const int iy = oy * 2 + ky - pad_top;
        if (iy < 0 || iy >= h) continue;
        for (int kx = 0; kx < 4; ++kx) {
            const int ix = ox * 2 + kx - pad_left;
            if (ix < 0 || ix >= wd) continue;
            const size_t off = (size_t)iy * wd + ix;
            const float *wp = w + ((size_t)oc * c_in) * 16 + ky * 4 + kx;
            for (int ci = 0; ci < c_in; ++ci)
                acc += wp[(size_t)ci * 16] * in[(size_t)ci * in_plane + off];
        }
    }
    out[idx] = acc;
}

// 2x2 stride 2 transposed conv - `ConvT_up` (UNetDecoderBlock, CrossGatingBlock).
//
// A scatter, so the weight is `[c_in][c_out][2][2]` and spatially REVERSED at
// load time (see `weights::convt2x2`): flax's ConvTranspose is the adjoint of its
// Conv, while a scatter reproduces torch's conv_transpose2d. Getting the flip
// wrong costs 65 dB (28 dB instead of 93 dB on a decoder block).
//
// out[oc][2y+ky][2x+kx] = bias[oc] + sum_ic in[ic][y][x] * w[ic][oc][ky][kx]
extern "C" __global__ void mx_convt2x2s2(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)h * wd;
    if (idx >= total) return;
    const int x = (int)(idx % wd);
    const int y = (int)(idx / wd);
    const size_t in_plane = (size_t)h * wd;
    const int oh = h * 2, ow = wd * 2;
    const size_t out_plane = (size_t)oh * ow;
    for (int ky = 0; ky < 2; ++ky) {
        const int oy = y * 2 + ky;
        for (int kx = 0; kx < 2; ++kx) {
            const int ox = x * 2 + kx;
            for (int oc = 0; oc < c_out; ++oc) {
                float acc = bias ? bias[oc] : 0.0f;
                const float *wp = w + ((size_t)oc * 2 + ky) * 2 + kx;
                for (int ic = 0; ic < c_in; ++ic)
                    acc += in[(size_t)ic * in_plane + idx] * wp[(size_t)ic * c_out * 4];
                out[(size_t)oc * out_plane + (size_t)oy * ow + ox] = acc;
            }
        }
    }
}

// The gated-MLP matmul, in the blocked layout.
//
// The blocked layout is `[c][outer][inner]` with `inner` contiguous. A gMLP
// Dense sits on ONE spatial axis, replaces that axis, and leaves the other one
// alone, so the axis it reduces over decides which of the two shapes this is:
//
//   mode 0: reduce over `outer` (the first spatial axis of the activation).
//           w is [out_dim][outer]; `inner` is the batch.
//           out[c][a][b] = bias[a] + sum_s w[a][s] * in[c][s][b]
//   mode 1: reduce over `inner` (the contiguous axis).
//           w is [out_dim][inner]; `outer` is the batch.
//           out[c][b][a] = bias[a] + sum_s w[a][s] * in[c][b][s]
//
// Both write the result back in the input's own axis order - `out_dim` takes the
// place of the axis that was reduced. Which of the four cases a call is depends
// on the ORDER the caller put the activation in: the grid gate reduces over the
// grid axis and the block gate over the patch axis, and `mx_block_perm` is what
// chooses which of those ends up outer (see `GridGatingUnit`'s swapaxes).
// One thread per output element, so no atomics and a deterministic order.
extern "C" __global__ void mx_gate_mm(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c, int outer, int inner, int mode)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (mode == 0) {
        const long total = (long)c * outer * inner;
        if (idx >= total) return;
        const int b = (int)(idx % inner);
        const long t = idx / inner;
        const int a = (int)(t % outer);
        const int ch = (int)(t / outer);
        const float *x = in + (size_t)ch * outer * inner;
        const float *wa = w + (size_t)a * outer;
        float acc = bias ? bias[a] : 0.0f;
        for (int s = 0; s < outer; ++s) acc += wa[s] * x[(size_t)s * inner + b];
        out[idx] = acc;
    } else {
        const long total = (long)c * outer * inner;
        if (idx >= total) return;
        const int a = (int)(idx % inner);
        const long t = idx / inner;
        const int b = (int)(t % outer);
        const int ch = (int)(t / outer);
        const float *x = in + (size_t)ch * outer * inner + (size_t)b * inner;
        const float *wa = w + (size_t)a * inner;
        float acc = bias ? bias[a] : 0.0f;
        for (int s = 0; s < inner; ++s) acc += wa[s] * x[s];
        out[idx] = acc;
    }
}

// The gated-MLP matmul again, this time as an implicit GEMM over a shared-memory
// tile. `mx_gate_mm` above gives each output element its own thread, so every
// output re-reads its whole input row and its whole weight row: at 512x512 this
// family is the largest single share of the run (2.13 s of 5.97 s) and it runs at
// ~101 GFLOP/s, about 1% of this card's fp32 peak.
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

// elementwise multiply: y = a * b. The CALayer's `x * sigmoid(y)` and the
// cross-gating `x * gate` both need one, and `lg_add` is not it.
extern "C" __global__ void mx_mul(
    const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ y, long n)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
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
// `axis` 2 = horizontal (one thread per (c, y) row), 1 = vertical (one thread per
// (c, x) column). The horizontal pass leaves the plane the target width, so the
// caller runs it first and the vertical pass second.
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
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c * rows;
    if (idx >= total) return;
    const int g = (int)(idx % rows);
    const int ch = (int)(idx / rows);
    const int n_in = (axis == 2) ? win : hin;
    const int n_out = (axis == 2) ? wout : hout;
    const float inv_scale = (float)n_in / (float)n_out;
    const float kscale = inv_scale > 1.0f ? inv_scale : 1.0f;
    const float *src = in + (size_t)ch * in_plane;
    float *dst = out + (size_t)ch * out_plane;
    for (int o = 0; o < n_out; ++o) {
        const float sample = ((float)o + 0.5f) * inv_scale - 0.5f;
        int j0 = (int)ceilf(sample - kscale);
        int j1 = (int)floorf(sample + kscale);
        if (j0 < 0) j0 = 0;
        if (j1 > n_in - 1) j1 = n_in - 1;
        float wsum = 0.0f;
        float acc = 0.0f;
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
}

// Space <-> block permutation for the gMLP layers.
//
// For an (h, w) plane tiled into (gh, gw) blocks of (fh, fw) pixels:
//   g = gy * gw + gx      (the "grid" axis)
//   p = fy * fw + fx      (the "patch" axis)
//   pixel (y, x) = (gy * fh + fy, gx * fw + fx)
//
// `forward` gathers NCHW -> blocked, otherwise it scatters blocked -> NCHW. Both
// directions walk (c, g, p) and derive the pixel from it, so there is one index
// derivation and no risk of the two directions drifting apart.
// `swap` transposes the two block axes, giving the (patch, grid) order: the
// multi-scale `signal` is concatenated over UpSampleRatio outputs whose own grid
// size changes with the scale, so both orders are needed and one kernel with a
// flag keeps the relationship visible.
//
//   forward, !swap: out[c][g][p] = in[y][x]     (block_images)
//   forward,  swap: out[c][p][g] = in[y][x]
//   !forward,!swap: out[y][x] = in[c][g][p]     (unblock_images)
//   !forward, swap: out[y][x] = in[c][p][g]
extern "C" __global__ void mx_block_perm(
    const float *__restrict__ in, float *__restrict__ out,
    int c, int h, int wd, int gh, int gw, int fh, int fw, int swap, int forward)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c * gh * gw * fh * fw;
    if (idx >= total) return;
    const int gsz = gh * gw, psz = fh * fw;
    const int p = (int)(idx % psz);
    const long t = idx / psz;
    const int g = (int)(t % gsz);
    const int ch = (int)(t / gsz);
    const int fy = p / fw, fx = p % fw;
    const int gy = g / gw, gx = g % gw;
    const size_t plane = (size_t)h * wd;
    const size_t pix = (size_t)(gy * fh + fy) * wd + (gx * fw + fx);
    const size_t blk = swap ? ((size_t)p * gsz + g) : ((size_t)g * psz + p);
    if (forward) out[(size_t)ch * plane + blk] = in[(size_t)ch * plane + pix];
    else out[(size_t)ch * plane + pix] = in[(size_t)ch * plane + blk];
}

// Per-channel mean over the spatial plane, for the CALayer's global average
// pooling. A separate kernel rather than a reduction kernel with a broadcast
// flag: the output here is `c` floats, so one thread per channel with a serial
// loop is both the simplest and (at these sizes) the fastest form.
extern "C" __global__ void mx_channel_mean(
    const float *__restrict__ in, float *__restrict__ out, int c, int hw)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= c) return;
    const float *p = in + (size_t)ch * hw;
    float s = 0.0f;
    for (int i = 0; i < hw; ++i) s += p[i];
    out[ch] = s / (float)hw;
}

// The gated-MLP gate: y = u * (v + 1). This is the same form in all four places
// it appears (GridGatingUnit, BlockGatingUnit, and both halves of
// GetSpatialGatingWeights' weight branch is NOT one of them - that one has no
// +1), and it is not `lg_add` + `mx_mul` by accident of arithmetic: the +1 is
// part of the gMLP definition and is what keeps the gate near the identity.
extern "C" __global__ void mx_gate_apply(
    const float *__restrict__ u, const float *__restrict__ v, float *__restrict__ y, long n)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = u[i] * (v[i] + 1.0f);
}

// Per-channel scale over an NCHW plane: out[c][p] = in[c][p] * s[c].
//
// The toolkit's `lg_row_affine` indexes its vector by the CONTIGUOUS dimension,
// which for NCHW is the width, not the channel - so it cannot express this. The
// CALayer's `x * sigmoid(y)` is exactly this op with `s` = the excited weights,
// which is why it is a kernel rather than three elementwise calls.
extern "C" __global__ void mx_channel_scale(
    const float *__restrict__ in, const float *__restrict__ s, float *__restrict__ out,
    int c, int hw)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c * hw;
    if (idx >= total) return;
    const int ch = (int)(idx / hw);
    out[idx] = in[idx] * s[ch];
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

// y = 1 / (1 + exp(-x)). The toolkit has no standalone sigmoid (its transformer
// engines fold it into an attention kernel), and the CALayer needs one.
extern "C" __global__ void mx_sigmoid(
    const float *__restrict__ x, float *__restrict__ y, long n)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
}
