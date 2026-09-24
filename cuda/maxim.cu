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
