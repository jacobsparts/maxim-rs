//! Compiles this engine's kernels into two modules: the shared `lightgpu`
//! toolkit's `cuda/kernels.cu` (TOOLKIT_KERNELS) and this project's own
//! `cuda/maxim.cu` (PROJECT_KERNELS). Each compiles to its own fatbin with its
//! own `--entries` list and `src/gpu.rs` loads them as separate modules, so
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
    "lg_gelu_erf",
    "lg_lrelu",
    // The channel-axis LayerNorm. MAXIM needs it in two places that look
    // different and are the same op: an NCHW (c, h*w) tensor, and a gMLP tensor
    // in the blocked layout (c, grid*patch) whose channel axis is contiguous.
    "lg_channel_layer_norm",
    // 1x1 conv. NO LONGER LAUNCHED: it is the model's biggest op family, and its
    // one-thread-per-output form re-reads the input `c_in` times (see
    // `mx_conv1x1_t`). It stays in the list because leaving a kernel out of
    // `--entries` prunes it from the fatbin and a stale caller would then fail at
    // LAUNCH - the compile-time completeness check at the bottom of this file is
    // what keeps the two in step, and it cannot tell a call site from a
    // declaration.
    "lg_conv1x1",
    // 3x3 pad-1 conv, the SAM block and the stage input convs. The Winograd
    // variant is *not* used: at c_in=3 the transform does not amortise, and the
    // rest of the 3x3s are few enough that the direct form is simpler to trust.
    "lg_conv3x3s1p1",
];

/// This project's own kernels, in `cuda/maxim.cu`.
const PROJECT_KERNELS: &[&str] = &[
    "mx_conv1x1_t",
    "mx_conv3x3_t2",
    "mx_conv3x3_t4",
    "mx_conv4x4s2",
    "mx_convt2x2s2",
    "mx_gate_mm",
    "mx_gate_mm_t0",
    "mx_gate_mm_t1",
    "mx_gate_apply",
    "mx_mul",
    "mx_resize_axis",
    "mx_block_perm",
    "mx_channel_mean",
    "mx_channel_scale",
    "mx_down2",
    "mx_sigmoid",
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
