"""PyTorch re-implementation of MAXIM, used as the validation reference.

This is a development tool, not part of the engine: `maxim-rs` reproduces the
graph below in Rust, and `tools/compare.py` diffs the two on the tensors this
script dumps with `--dump`.

It is a faithful transcription of the Flax model in
google-research/maxim/maxim/models/maxim.py, with two deliberate departures:

  * layouts are NCHW instead of NHWC, because that is what the engine and the
    lightgpu kernels use. A per-pixel channel op (LayerNorm over the channel
    axis, Dense/1x1 conv, gelu) is layout-agnostic, so only the two ops that are
    genuinely spatial - `block_images`/`unblock_images` and the gating matmuls -
    are written out in the blocked form;
  * jax.image.resize is replaced by F.interpolate. Both use half-pixel centres
    (`(i + 0.5) * in/out - 0.5`) and a triangle kernel with renormalised
    boundary weights; for the integer *upsampling* ratios MAXIM uses, jax's
    antialias=True is a no-op (`kernel_scale = max(inv_scale, 1) = 1`), so
    F.interpolate(scale_factor=r, mode="bilinear", align_corners=False) is the
    exact twin. The one nearest-neighbour call, the multi-scale input pyramid,
    is *not* F.interpolate either: jax maps output i to floor((i + 0.5) * m / n),
    i.e. a 2x downsample keeps the ODD rows and columns, while torch's nearest
    keeps the even ones. It is done with a slice here.

Run against the official checkpoint and the official results:

    python3.11 tools/reference.py --ckpt ../models/checkpoint.npz \\
        --image /tmp/lol/eval15/low/1.png --out /tmp/ref_1.png
"""

import argparse
import collections
import io
import os

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6  # flax nn.LayerNorm default


# ---------------------------------------------------------------- parameters


def load_params(path):
    """Read the flax checkpoint (`opt/target/*` leaves) into a flat dict."""
    with np.load(path, allow_pickle=False) as z:
        keys = [k for k in z.keys() if k.startswith("opt/target/")]
        params = {k[len("opt/target/"):]: np.asarray(z[k]) for k in keys}
    return params


class Params:
    """Name-addressed parameters, in torch's layout.

    Flax conv kernels are [kh, kw, c_in, c_out] and dense kernels [c_in, c_out];
    torch wants [c_out, c_in, kh, kw] and [c_out, c_in]. The transposed conv is
    the one that differs: flax's ConvTranspose kernel is [kh, kw, c_out, c_in],
    which is already torch's [c_in, c_out, kh, kw] order.
    """

    def __init__(self, flat):
        self.flat = flat
        self.cache = {}

    def raw(self, name):
        return self.flat[name]

    def t(self, name):
        if name not in self.cache:
            a = self.flat[name]
            t = torch.from_numpy(np.ascontiguousarray(a))
            if t.dtype not in (torch.float32, torch.float64):
                t = t.float()
            self.cache[name] = t
        return self.cache[name]

    def conv(self, prefix, transpose=False):
        """A conv's [out, in, kh, kw] weight (torch's layout).

        Both flax convs store [kh, kw, in, out] - the transposed one included,
        which is why its permute differs: torch's ConvTranspose2d wants
        [in, out, kh, kw], and the checkpoint confirms it (`ConvTranspose_0` of a
        decoder block with features=32 taking 64 input channels is
        (2, 2, 64, 32)).
        """
        k = self.t(prefix + "/kernel")
        b = self.t(prefix + "/bias")
        if transpose:
            # flax's ConvTranspose is the ADJOINT of its Conv, so it scatters
            # with a spatially REVERSED kernel relative to torch's
            # conv_transpose2d. Verified against the real model: flipping gives
            # 93 dB on a decoder block, not flipping 28 dB.
            return k.permute(2, 3, 0, 1).flip(2, 3).contiguous(), b
        return k.permute(3, 2, 0, 1).contiguous(), b

    def dense(self, prefix):
        """A Dense layer as a 1x1 conv weight [out, in, 1, 1]."""
        k = self.t(prefix + "/kernel")
        b = self.t(prefix + "/bias")
        return k.t().contiguous().view(k.shape[1], k.shape[0], 1, 1), b

    def ln(self, prefix):
        return self.t(prefix + "/scale"), self.t(prefix + "/bias")


# ------------------------------------------------------------------ kernels


def conv1x1(x, wb):
    w, b = wb
    assert tuple(w.shape[2:]) == (1, 1), "conv1x1 given a %s kernel" % (tuple(w.shape[2:]),)
    return F.conv2d(x, w, b)


def conv3x3(x, wb, pad=1):
    """3x3 with flax's default padding='SAME', i.e. pad 1."""
    w, b = wb
    assert tuple(w.shape[2:]) == (3, 3), "conv3x3 given a %s kernel" % (tuple(w.shape[2:]),)
    return F.conv2d(x, w, b, padding=pad)


def same_pad(size, stride, k):
    """Flax's padding='SAME': the pads that make the output ceil(size/stride).

    Asymmetric for odd sizes (pad_before = total // 2, the rest after), which is
    what lax.padtype_to_pads does. The model only ever sees even sizes here - the
    padded input is a multiple of 64 and every level halves it - but the engine
    reproduces the general rule.
    """
    out = -(-size // stride)
    total = max((out - 1) * stride + k - size, 0)
    return total // 2, total - total // 2


def conv_down(x, wb):
    """4x4 stride 2 with SAME padding - the encoder's downsample.

    NOT the toolkit's `lg_conv4x4s4` (stride 4, no padding): this halves the
    resolution and keeps ceil(h/2) rows, where a stride-4 patch embed would
    quarter it.
    """
    w, b = wb
    ph, pw = same_pad(x.shape[-2], 2, 4), same_pad(x.shape[-1], 2, 4)
    if ph != pw:
        raise NotImplementedError("asymmetric SAME padding: use F.pad + conv")
    x = F.pad(x, (pw[0], pw[1], ph[0], ph[1]))
    return F.conv2d(x, w, b, stride=2)


def convt_up(x, wb):
    """2x2 stride 2 - the decoder's upsample."""
    w, b = wb
    return F.conv_transpose2d(x, w, b, stride=2)


def layer_norm(x, wb):
    """LayerNorm over the channel axis of an NCHW tensor."""
    scale, bias = wb
    m = x.mean(1, keepdim=True)
    v = x.var(1, unbiased=False, keepdim=True)
    return (x - m) * torch.rsqrt(v + EPS) * scale.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)


def gelu(x):
    return F.gelu(x)


def lrelu(x, slope=0.2):
    return F.leaky_relu(x, slope)


def block_images(x, fh, fw):
    """(1, C, gh*fh, gw*fw) -> (1, C, gh*gw, fh*fw), the einops rearrange."""
    _, c, h, w = x.shape
    gh, gw = h // fh, w // fw
    x = x.view(1, c, gh, fh, gw, fw)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    return x.view(1, c, gh * gw, fh * fw)


def unblock_images(x, gh, gw, fh, fw):
    """The inverse of block_images."""
    _, c, g, p = x.shape
    x = x.view(1, c, gh, gw, fh, fw)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    return x.view(1, c, gh * fh, gw * fw)


def upsample_ratio(x, ratio):
    """jax.image.resize(..., method='bilinear').

    Two regimes, and they are not the same op:

    * ratio > 1 (upsampling): the triangle kernel is used at its natural width
      (`kernel_scale = max(inv_scale, 1) = 1`), which is exactly
      F.interpolate(..., antialias=False) - and antialias=True agrees with it to
      167 dB, so either is fine.
    * ratio < 1 (downsampling): jax widens the triangle by `1/ratio`, i.e. it
      antialiases, and torch only matches with antialias=True. Measured on a
      24x40 random input: 150 dB with antialias=True against 22.9 dB (ratio 0.5)
      and 18.4 dB (ratio 0.25) without. The model does downsample here: the
      cross-gating signal asks for ratio 2**(j-i), which is 1/4 and 1/2 for the
      coarse levels.
    """
    if ratio == 1:
        return x
    return F.interpolate(x, scale_factor=ratio, mode="bilinear", align_corners=False,
                         antialias=ratio < 1)


def nearest_down(x, factor):
    """jax.image.resize(..., method='nearest'): output i <- input floor((i+.5)*f)."""
    if factor == 1:
        return x
    return x[:, :, factor // 2::factor, factor // 2::factor].contiguous()


# -------------------------------------------------------------- model blocks


class Maxim:
    def __init__(self, params, cfg):
        self.p = Params(params)
        self.cfg = cfg
        self.trace = None
        self.up_index = 0

    # -- helpers ---------------------------------------------------------
    def up_name(self):
        n = self.up_index
        self.up_index += 1
        return "UpSampleRatio_%d" % n

    def rec(self, name, x):
        if self.trace is not None:
            self.trace[name] = x.detach().clone()

    def deep_rec(self, name, tag, x):
        """Per-block-internal dumps, for localising a divergence to one line.

        Off by default: they are a development aid, and the engine counterparts
        are behind MAXIM_DEEP_DUMPS for the same reason.
        """
        if os.environ.get("MAXIM_DEEP_DUMPS"):
            self.rec(name.replace("/", "_") + "_" + tag, x)

    # -- leaf blocks -----------------------------------------------------
    def calayer(self, name, x, reduction):
        """Squeeze-and-excitation channel attention."""
        y = x.mean((2, 3), keepdim=True)
        self.deep_rec(name, "mean", y)
        y = conv1x1(y, self.p.conv(name + "/Conv_0"))
        self.deep_rec(name, "sq", y)
        y = F.relu(y)
        self.deep_rec(name, "relu", y)
        y = conv1x1(y, self.p.conv(name + "/Conv_1"))
        self.deep_rec(name, "excite", y)
        self.deep_rec(name, "sigmoid", torch.sigmoid(y))
        return x * torch.sigmoid(y)

    def rcab(self, name, x, reduction):
        shortcut = x
        x = layer_norm(x, self.p.ln(name + "/LayerNorm"))
        self.deep_rec(name, "ln", x)
        x = conv3x3(x, self.p.conv(name + "/conv1"))
        self.deep_rec(name, "conv1", x)
        x = lrelu(x)
        self.deep_rec(name, "lrelu", x)
        x = conv3x3(x, self.p.conv(name + "/conv2"))
        self.deep_rec(name, "conv2", x)
        x = self.calayer(name + "/channel_attention", x, reduction)
        self.deep_rec(name, "ca", x)
        return x + shortcut



    def rdcab(self, name, x, reduction):
        y = layer_norm(x, self.p.ln(name + "/LayerNorm"))
        y = conv1x1(y, self.p.dense(name + "/channel_mixing/Dense_0"))
        y = gelu(y)
        y = conv1x1(y, self.p.dense(name + "/channel_mixing/Dense_1"))
        y = self.calayer(name + "/channel_attention", y, reduction)
        return x + y

    def grid_gating_unit(self, name, x, grid_size):
        """SpatialGatingUnit over the grid axis (the gMLP 'global' mix)."""
        u, v = torch.chunk(x, 2, dim=1)
        v = layer_norm(v, self.p.ln(name + "/intermediate_layernorm"))
        g = grid_size[0] * grid_size[1]
        w, b = self.p.dense(name + "/Dense_0")
        # out[g, p, c] = sum_g' w[g, g'] v[g', p, c]
        v = torch.einsum("gi,cip->gpc", w.view(g, g), v[0])
        v = v + b.view(g, 1, 1)
        v = v.permute(2, 0, 1).unsqueeze(0)
        return u * (v + 1.0)

    def block_gating_unit(self, name, x, block_size):
        """SpatialGatingUnit over the within-block axis (the gMLP 'local' mix)."""
        u, v = torch.chunk(x, 2, dim=1)
        v = layer_norm(v, self.p.ln(name + "/intermediate_layernorm"))
        p = block_size[0] * block_size[1]
        w, b = self.p.dense(name + "/Dense_0")
        # out[g, p, c] = sum_p' w[p, p'] v[g, p', c]
        v = torch.einsum("pj,cgj->gpc", w.view(p, p), v[0])
        v = v + b.view(1, p, 1)
        v = v.permute(2, 0, 1).unsqueeze(0)
        return u * (v + 1.0)

    def grid_gmlp(self, name, x, grid_size, factor):
        _, c, h, w = x.shape
        gh, gw = grid_size
        fh, fw = h // gh, w // gw
        x = block_images(x, fh, fw)
        self.deep_rec(name, "blocked", x)
        y = layer_norm(x, self.p.ln(name + "/LayerNorm"))
        self.deep_rec(name, "ln", y)
        y = conv1x1(y, self.p.dense(name + "/in_project"))
        self.deep_rec(name, "inproject", y)
        y = gelu(y)
        self.deep_rec(name, "gelu", y)
        y = self.grid_gating_unit(name + "/GridGatingUnit", y, (gh, gw))
        self.deep_rec(name, "gated", y)
        y = conv1x1(y, self.p.dense(name + "/out_project"))
        self.deep_rec(name, "proj", y)
        x = x + y
        return unblock_images(x, gh, gw, fh, fw)

    def block_gmlp(self, name, x, block_size, factor):
        _, c, h, w = x.shape
        fh, fw = block_size
        gh, gw = h // fh, w // fw
        x = block_images(x, fh, fw)
        self.deep_rec(name, "blocked", x)
        y = layer_norm(x, self.p.ln(name + "/LayerNorm"))
        self.deep_rec(name, "ln", y)
        y = conv1x1(y, self.p.dense(name + "/in_project"))
        self.deep_rec(name, "inproject", y)
        y = gelu(y)
        self.deep_rec(name, "gelu", y)
        y = self.block_gating_unit(name + "/BlockGatingUnit", y, (fh, fw))
        self.deep_rec(name, "gated", y)
        y = conv1x1(y, self.p.dense(name + "/out_project"))
        self.deep_rec(name, "proj", y)
        x = x + y
        return unblock_images(x, gh, gw, fh, fw)

    def split_head_gmlp(self, name, x, block_size, grid_size):
        """ResidualSplitHeadMultiAxisGmlpLayer: grid + block gMLP in parallel."""
        shortcut = x
        x = layer_norm(x, self.p.ln(name + "/LayerNorm_in"))
        x = conv1x1(x, self.p.dense(name + "/in_project"))
        x = gelu(x)
        u, v = torch.chunk(x, 2, dim=1)
        u = self.grid_gmlp(name + "/GridGmlpLayer", u, grid_size, 2)
        v = self.block_gmlp(name + "/BlockGmlpLayer", v, block_size, 2)
        x = torch.cat([u, v], dim=1)
        x = conv1x1(x, self.p.dense(name + "/out_project"))
        return x + shortcut

    def spatial_gating_weights(self, name, x, block_size, grid_size):
        _, c, h, w = x.shape
        x = layer_norm(x, self.p.ln(name + "/LayerNorm_in"))
        x = conv1x1(x, self.p.dense(name + "/in_project"))
        x = gelu(x)
        u, v = torch.chunk(x, 2, dim=1)

        gh, gw = grid_size
        fh, fw = h // gh, w // gw
        u = block_images(u, fh, fw)
        dim_u = u.shape[2]
        wu, bu = self.p.dense(name + "/Dense_0")
        # out[c, i, p] = sum_j D[i, j] * u[c, j, p]: the gating matmul runs over
        # the grid axis, and stays in the blocked (c, grid, patch) layout.
        u = torch.einsum("ij,cjp->cip", wu.view(dim_u, dim_u), u[0])
        u = (u + bu.view(dim_u, 1)).unsqueeze(0)
        u = unblock_images(u, gh, gw, fh, fw)

        fh, fw = block_size
        gh, gw = h // fh, w // fw
        v = block_images(v, fh, fw)
        dim_v = v.shape[3]
        wv, bv = self.p.dense(name + "/Dense_1")
        # out[c, g, i] = sum_j D[i, j] * v[c, g, j]: here the matmul runs over the
        # within-block axis.
        v = torch.einsum("ij,cgj->cgi", wv.view(dim_v, dim_v), v[0])
        v = (v + bv.view(1, dim_v)).unsqueeze(0)
        v = unblock_images(v, gh, gw, fh, fw)

        x = torch.cat([u, v], dim=1)
        return conv1x1(x, self.p.dense(name + "/out_project"))

    def cross_gating_block(self, name, x, y, block_size, grid_size, upsample_y):
        if upsample_y:
            y = convt_up(y, self.p.conv(name + "/ConvTranspose_0", transpose=True))
        x = conv1x1(x, self.p.conv(name + "/Conv_0"))
        c = x.shape[1]
        y = conv1x1(y, self.p.conv(name + "/Conv_1"))
        shortcut_x, shortcut_y = x, y

        x = layer_norm(x, self.p.ln(name + "/LayerNorm_x"))
        x = conv1x1(x, self.p.dense(name + "/in_project_x"))
        x = gelu(x)
        gx = self.spatial_gating_weights(
            name + "/SplitHeadMultiAxisGating_x", x, block_size, grid_size)

        y = layer_norm(y, self.p.ln(name + "/LayerNorm_y"))
        y = conv1x1(y, self.p.dense(name + "/in_project_y"))
        y = gelu(y)
        gy = self.spatial_gating_weights(
            name + "/SplitHeadMultiAxisGating_y", y, block_size, grid_size)

        y = y * gx
        y = conv1x1(y, self.p.dense(name + "/out_project_y"))
        y = y + shortcut_y

        x = x * gy
        x = conv1x1(x, self.p.dense(name + "/out_project_x"))
        x = x + y + shortcut_x
        return x, y

    def bottleneck(self, name, x, block_size, grid_size, num_groups, reduction):
        x = conv1x1(x, self.p.conv(name + "/input_proj"))
        shortcut = x
        for g in range(num_groups):
            x = self.split_head_gmlp(
                name + "/SplitHeadMultiAxisGmlpLayer_%d" % g, x, block_size, grid_size)
            x = self.rdcab(
                name + "/channel_attention_block_1_%d" % g, x, reduction)
        return x + shortcut

    def unet_encoder_block(self, name, x, features, block_size, grid_size,
                           num_groups, reduction, downsample, skip=None,
                           enc=None, dec=None):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        self.deep_rec(name, "mid_cat", x)
        x = conv1x1(x, self.p.conv(name + "/Conv_0"))
        self.deep_rec(name, "mid_conv", x)
        shortcut = x
        for g in range(num_groups):
            x = self.split_head_gmlp(
                name + "/SplitHeadMultiAxisGmlpLayer_%d" % g, x, block_size, grid_size)
            self.deep_rec(name, "mid_g%d_gmlp" % g, x)
            x = self.rcab(name + "/channel_attention_block_1%d" % g, x, reduction)
            self.deep_rec(name, "mid_g%d_rcab" % g, x)
        x = x + shortcut
        self.deep_rec(name, "mid_sum", x)
        if enc is not None and dec is not None:
            x, _ = self.cross_gating_block(
                name + "/cross_gating_block", x, enc + dec, block_size, grid_size,
                upsample_y=False)
        if downsample:
            return conv_down(x, self.p.conv(name + "/Conv_1")), x
        return x

    def unet_decoder_block(self, name, x, bridge, features, block_size, grid_size,
                           num_groups, reduction):
        x = convt_up(x, self.p.conv(name + "/ConvTranspose_0", transpose=True))
        # flax auto-names the unnamed child module, so its parameters live under
        # `UNetEncoderBlock_0/` in the checkpoint.
        return self.unet_encoder_block(
            name + "/UNetEncoderBlock_0", x, features, block_size, grid_size,
            num_groups, reduction, downsample=False, skip=bridge)

    def sam(self, name, x, x_image, features):
        # Conv_2 produces `features` channels, not 3: the Flax SAM calls
        # Conv3x3(self.features) for the attention map, and the checkpoint's
        # Conv_2 kernel is (3, 3, 3, features). An earlier version of this file
        # wrote 3->3 and let torch BROADCAST a 3-channel gate over `features`
        # channels - which silently produced a different result, not an error.
        x1 = conv3x3(x, self.p.conv(name + "/Conv_0"))
        image = conv3x3(x, self.p.conv(name + "/Conv_1")) + x_image
        x2 = torch.sigmoid(conv3x3(image, self.p.conv(name + "/Conv_2")))
        x1 = x1 * x2
        return x1 + x, image

    # -- the model -------------------------------------------------------
    def __call__(self, x):
        cfg = self.cfg
        depth = cfg["depth"]
        stages = cfg["num_stages"]
        features = cfg["features"]
        nss = cfg["num_supervision_scales"]
        ngroups = cfg["num_groups"]
        nbb = cfg["num_bottleneck_blocks"]
        hres = cfg["high_res_stages"]
        bhr, blr = cfg["block_size_hr"], cfg["block_size_lr"]
        ghr = cfg["grid_size_hr"]
        red = cfg["channels_reduction"]

        self.trace = {} if self.trace is not None else None
        self.rec("input", x)

        shortcuts = [x]
        for i in range(1, nss):
            shortcuts.append(nearest_down(x, 2 ** i))

        outputs_all = []
        sam_features, encs_prev, decs_prev = [], [], []
        for s in range(stages):
            x_scales = []
            for i in range(nss):
                xs = conv3x3(shortcuts[i], self.p.conv("stage_%d_input_conv_%d" % (s, i)))
                if s > 0:
                    bs = bhr if i < hres else blr
                    gs = ghr if i < hres else blr
                    xs, _ = self.cross_gating_block(
                        "stage_%d_input_fuse_sam_%d" % (s, i), xs, sam_features.pop(),
                        bs, gs, upsample_y=False)
                self.rec("stage%d_input_conv%d" % (s, i), xs)
                x_scales.append(xs)

            encs = []
            x = x_scales[0]
            for i in range(depth):
                bs = bhr if i < hres else blr
                gs = ghr if i < hres else blr
                x_scale = x_scales[i] if i < nss else None
                enc_prev = encs_prev.pop() if s > 0 else None
                dec_prev = decs_prev.pop() if s > 0 else None
                x, bridge = self.unet_encoder_block(
                    "stage_%d_encoder_block_%d" % (s, i), x, (2 ** i) * features,
                    bs, gs, ngroups, red, downsample=True, skip=x_scale,
                    enc=enc_prev, dec=dec_prev)
                self.rec("stage%d_enc%d_bridge" % (s, i), bridge)
                self.rec("stage%d_enc%d_out" % (s, i), x)
                encs.append(bridge)

            for i in range(nbb):
                x = self.bottleneck(
                    "stage_%d_global_block_%d" % (s, i), x, blr, blr, ngroups, red)
            self.rec("stage%d_global" % s, x)
            global_feature = x

            skip_features = []
            for i in reversed(range(depth)):
                bs = bhr if i < hres else blr
                gs = ghr if i < hres else blr
                signal = torch.cat([
                    conv1x1(upsample_ratio(enc, 2 ** (j - i)),
                            self.p.conv(self.up_name() + "/Conv_0"))
                    for j, enc in enumerate(encs)], dim=1)
                skips, global_feature = self.cross_gating_block(
                    "stage_%d_cross_gating_block_%d" % (s, i), signal, global_feature,
                    bs, gs, upsample_y=True)
                self.rec("stage%d_skip%d" % (s, i), skips)
                skip_features.append(skips)

            outputs, decs, sam_new = [], [], []
            for i in reversed(range(depth)):
                bs = bhr if i < hres else blr
                gs = ghr if i < hres else blr
                signal = torch.cat([
                    conv1x1(upsample_ratio(skip, 2 ** (depth - j - 1 - i)),
                            self.p.conv(self.up_name() + "/Conv_0"))
                    for j, skip in enumerate(skip_features)], dim=1)
                x = self.unet_decoder_block(
                    "stage_%d_decoder_block_%d" % (s, i), x, signal, (2 ** i) * features,
                    bs, gs, ngroups, red)
                self.rec("stage%d_dec%d" % (s, i), x)
                decs.append(x)
                if i < nss:
                    if s < stages - 1:
                        sam, output = self.sam(
                            "stage_%d_supervised_attention_module_%d" % (s, i), x,
                            shortcuts[i], (2 ** i) * features)
                        outputs.append(output)
                        sam_new.append(sam)
                        self.rec("stage%d_sam%d" % (s, i), sam)
                    else:
                        output = conv3x3(x, self.p.conv("stage_%d_output_conv_%d" % (s, i))) + shortcuts[i]
                        outputs.append(output)
                    self.rec("stage%d_out%d" % (s, i), output)
            encs_prev = encs[::-1]
            decs_prev = decs
            sam_features = sam_new
            outputs_all.append(outputs)
        return outputs_all


# ------------------------------------------------------------------ pipeline


def mod_padding_symmetric(image, factor=64):
    """The reference eval pipeline's padding: reflect, centred, to the next
    multiple of `factor` (or nothing when already a multiple).

    Note the odd-looking bound: `((h + factor) // factor) * factor` rounds up to
    the next multiple strictly *above* h when h is not already a multiple, which
    is what the reference does. The pad is split evenly, so with an even h (the
    caller makes the shape even first) padh is even and nothing is lost to the
    floor division.
    """
    h, w = image.shape[-2], image.shape[-1]
    hp = ((h + factor) // factor) * factor
    wp = ((w + factor) // factor) * factor
    padh = hp - h if h % factor != 0 else 0
    padw = wp - w if w % factor != 0 else 0
    return F.pad(image, (padw // 2, padw // 2, padh // 2, padh // 2), mode="reflect")


def make_shape_even(image):
    """Pad the bottom and right by one row/column of reflection when odd."""
    h, w = image.shape[-2], image.shape[-1]
    ph, pw = 1 if h % 2 else 0, 1 if w % 2 else 0
    if ph or pw:
        image = F.pad(image, (0, pw, 0, ph), mode="reflect")
    return image


def preprocess(img_chw, factor=64):
    x = make_shape_even(img_chw)
    h_even, w_even = x.shape[-2], x.shape[-1]
    x = mod_padding_symmetric(x, factor)
    return x, (h_even, w_even)


def postprocess(pred, orig_h, orig_w, even_h, even_w):
    """The inverse of preprocess, exactly as run_eval.py crops it: centred on the
    even shape, then cropped to the original (possibly odd) height and width."""
    nh, nw = pred.shape[-2], pred.shape[-1]
    hs = nh // 2 - even_h // 2
    ws = nw // 2 - even_w // 2
    return pred[:, :, hs:hs + orig_h, ws:ws + orig_w]


def read_image(path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    a = np.asarray(img, np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).contiguous()


def write_image(path, x):
    from PIL import Image
    a = x[0].permute(1, 2, 0).clamp(0, 1).numpy()
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(path)


CONFIGS = {
    "S-1": dict(features=32, depth=3, num_stages=1, num_groups=2,
                num_bottleneck_blocks=2, block_gmlp_factor=2, grid_gmlp_factor=2,
                input_proj_factor=2, channels_reduction=4),
    "S-2": dict(features=32, depth=3, num_stages=2, num_groups=2,
                num_bottleneck_blocks=2, block_gmlp_factor=2, grid_gmlp_factor=2,
                input_proj_factor=2, channels_reduction=4),
    "S-3": dict(features=32, depth=3, num_stages=3, num_groups=2,
                num_bottleneck_blocks=2, block_gmlp_factor=2, grid_gmlp_factor=2,
                input_proj_factor=2, channels_reduction=4),
    "M-1": dict(features=64, depth=3, num_stages=1, num_groups=2,
                num_bottleneck_blocks=2, block_gmlp_factor=2, grid_gmlp_factor=2,
                input_proj_factor=2, channels_reduction=4),
    "M-2": dict(features=64, depth=3, num_stages=2, num_groups=2,
                num_bottleneck_blocks=2, block_gmlp_factor=2, grid_gmlp_factor=2,
                input_proj_factor=2, channels_reduction=4),
    "M-3": dict(features=64, depth=3, num_stages=3, num_groups=2,
                num_bottleneck_blocks=2, block_gmlp_factor=2, grid_gmlp_factor=2,
                input_proj_factor=2, channels_reduction=4),
}


def model_config(variant):
    cfg = dict(CONFIGS[variant])
    cfg.update(num_supervision_scales=3, num_outputs=3, use_bias=True,
               dropout_rate=0.0, high_res_stages=2,
               block_size_hr=(16, 16), block_size_lr=(8, 8),
               grid_size_hr=(16, 16), grid_size_lr=(8, 8))
    return cfg


def run(params, img, variant="S-2", factor=64, dump=None, trace=False):
    cfg = model_config(variant)
    m = Maxim(params, cfg)
    if trace:
        m.trace = {}
    x, even = preprocess(img, factor)
    outs = m(x)
    pred = outs[-1][-1]
    pred = postprocess(pred, img.shape[-2], img.shape[-1], even[0], even[1])
    if dump:
        tr = dict(m.trace or {})
        for k, v in tr.items():
            np.save(os.path.join(dump, k.replace("/", "_") + ".npy"),
                    v.numpy().astype(np.float32))
        np.save(os.path.join(dump, "final.npy"), pred.numpy().astype(np.float32))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="S-2")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out")
    ap.add_argument("--dump")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    params = load_params(args.ckpt)
    img = read_image(args.image)
    with torch.no_grad():
        pred = run(params, img, args.variant, dump=args.dump, trace=args.trace or bool(args.dump))
    if args.out:
        write_image(args.out, pred)
        print("wrote", args.out)
    print("output", tuple(pred.shape), "range %.4f..%.4f" % (float(pred.min()), float(pred.max())))


if __name__ == "__main__":
    main()
