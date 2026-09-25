//! Compiles this engine's kernels into two modules: the shared `lightgpu`
//! toolkit's `cuda/kernels.cu` (TOOLKIT_KERNELS) and this project's own
//! `cuda/maxim.cu` (PROJECT_KERNELS). Each compiles to its own fatbin with its
//! own `--entries` list and `src/cuda.rs` loads them as separate modules, so
//! neither can shadow a name in the other.
//!
//! A kernel missing from its list is PRUNED from the fatbin and fails at launch
//! rather than at build time, so both lists are checked against the source they
//! are compiled from before nvcc runs: a typo, or a kernel moved between files,
//! fails the build.

/// Generic ops from the shared toolkit. Exactly the ones `exec_gpu` launches:
/// an entry here is a kernel embedded in the binary, so a name that no op calls
/// is wasted bytes, and `lg_noop`/`lg_scale`/`lg_copy` are not called (a plain
/// copy is a device-to-device memcpy, which is faster than launching for it).
const TOOLKIT_KERNELS: &[&str] = &[
    "lg_add",
    // An elementwise product. This engine shipped its own (`mx_mul`) until the
    // toolkit took one: NAFNet's SimpleGate and the other gated architectures
    // need a plain multiply too, so it is generic rather than MAXIM-specific.
    "lg_mul",
    "lg_gelu_erf",
    "lg_lrelu",
    // The CALayer is three toolkit ops: excite (a sigmoid), the global average
    // pool, and the per-channel scale. All three were this engine's own kernels
    // until they were recognised as generic; the toolkit's `lg_channel_scale` is
    // its `lg_channel_affine` with no shift, and `lg_sigmoid` differs from the
    // deleted `mx_sigmoid` only in using `__expf` where this file used `expf`.
    "lg_sigmoid",
    "lg_channel_mean",
    "lg_channel_scale",
    // The channel-axis LayerNorm. MAXIM needs it in two places that look
    // different and are the same op: an NCHW (c, h*w) tensor, and a gMLP tensor
    // in the blocked layout (c, grid*patch) whose channel axis is contiguous.
    "lg_channel_layer_norm",
    // `lg_conv3x3s1p1` USED TO BE HERE and is deliberately gone. It was embedded
    // as the fallback for "a width the tiled kernel cannot tile", but no such
    // width exists: `c3_body` derives `ow` from `x0` in-kernel, so a row whose
    // width is not a multiple of 64 costs one short segment rather than the whole
    // row, and `--factor` only changes the segment count. Nothing in this engine
    // ever named it, so it was embedded bytes that no op could launch.
    //
    // Its numbers, which is the reason the entry is worth this note: one thread
    // per output element, 9*c_in input loads and 9*c_in weight loads per output
    // (~1 MAC per 8 bytes moved), 1.07-1.27 TFLOP/s at c_in=128 against this
    // card's 8.2, and 1.90 s of a 4.32 s 512x512 run in the profile that
    // motivated the tiled form. The Winograd variant was never used either: at
    // c_in=3 the transform does not amortise.
];

/// This project's own kernels, in `cuda/maxim.cu`.
const PROJECT_KERNELS: &[&str] = &[
    "mx_conv1x1_t",
    "mx_conv1x1_t4",
    "mx_conv1x1_t8",
    "mx_conv3x3_t2",
    "mx_conv3x3_t4",
    "mx_conv3x3_w4",
    "mx_conv4x4s2",
    "mx_convt2x2s2",
    "mx_gate_mm_t0",
    "mx_gate_mm_t1",
    "mx_gate_apply",
    "mx_chan_ln",
    "mx_chan_ln_c32",
    "mx_chan_ln_c64",
    "mx_chan_ln_c128",
    "mx_resize_axis",
    "mx_block_perm",
    "mx_down2",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/maxim.cu");

    // `cargo build --no-default-features` is the pure-Rust CPU build: it must
    // not need nvcc, and `src/cuda.rs` (which includes the fatbins) is not
    // compiled at all, so the env vars it would embed are not needed.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let toolkit = lightgpu_build::toolkit_kernels_cu()
        .expect("locate lightgpu's cuda/kernels.cu (set LA_GPU_DIR to override)");
    let toolkit = toolkit.to_string_lossy().into_owned();

    for k in TOOLKIT_KERNELS {
        assert!(
            lightgpu_build::known_kernel(k),
            "unknown kernel `{k}`: not defined by the toolkit - did it move into cuda/maxim.cu?"
        );
    }
    let src = std::fs::read_to_string("cuda/maxim.cu").expect("read cuda/maxim.cu");
    let defined = lightgpu_build::kernel_names_in(&src);
    for k in PROJECT_KERNELS {
        assert!(
            defined.iter().any(|d| d == k),
            "`{k}` is not defined in cuda/maxim.cu (it has {})",
            defined.join(", ")
        );
    }
    // The other direction matters just as much: a kernel defined but NOT listed
    // is pruned from the fatbin by `--entries`, and then it fails at LAUNCH
    // rather than at build time. Checking both directions makes the list and the
    // source agree rather than merely overlap.
    for d in &defined {
        assert!(
            PROJECT_KERNELS.contains(&d.as_str()),
            "cuda/maxim.cu defines `{d}`, which PROJECT_KERNELS does not list -              it would be pruned from the fatbin and fail at launch"
        );
    }

    lightgpu_build::fatbin_modules(&[
        lightgpu_build::Source {
            path: &toolkit,
            out_name: "maxim_toolkit.fatbin",
            entries: Some(TOOLKIT_KERNELS),
        },
        lightgpu_build::Source {
            path: "cuda/maxim.cu",
            out_name: "maxim_project.fatbin",
            entries: Some(PROJECT_KERNELS),
        },
    ]);
}
