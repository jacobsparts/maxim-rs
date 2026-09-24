//! End-to-end checks on the built binary.
//!
//! The parity tools (`tools/compare.py`, `--verify-gpu`) compare the engine with
//! the torch reference and the two backends with each other. All of them can pass
//! while the model is wrong in a way both backends agree on - a transposed
//! weight, a block applied twice - because they never look at the ground truth.
//! What closes that gap is measuring the OUTPUT against the task: the model is
//! trained to brighten a low-light image, and a mean output level near the input
//! level means the network did nothing, however self-consistent it was.
//!
//! These tests need the converted checkpoint and the LOL dataset, so they are
//! skipped (not failed) when either is absent: a fresh clone has no weights, and
//! a test that cannot run is not a failing test.

use std::path::{Path, PathBuf};
use std::process::Command;

const MODEL: &str = "/home/jacob/lightgpu-family/models/maxim-lol.safetensors";
const ARCHIVE: &str = "/home/jacob/lightgpu-family/models/LOLdataset_train_test.zip";

fn binary() -> PathBuf {
    // `cargo test` puts the test binary in target/<profile>/deps; the engine is
    // two directories up.
    let mut p = std::env::current_exe().expect("the test binary's path");
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("maxim")
}

fn have_inputs() -> Option<PathBuf> {
    if !Path::new(MODEL).exists() || !Path::new(ARCHIVE).exists() {
        return None;
    }
    // One low-light image, unpacked on demand next to the test's temp output.
    let out = std::env::temp_dir().join("maxim-e2e");
    std::fs::create_dir_all(&out).ok()?;
    let low = out.join("1.png");
    if !low.exists() {
        let status = Command::new("unzip")
            .args(["-o", "-j", ARCHIVE, "eval15/low/1.png", "-d"])
            .arg(&out)
            .status()
            .ok()?;
        if !status.success() {
            return None;
        }
    }
    low.exists().then_some(low)
}

/// Run the engine on `low` and return the output's mean level in 0..1, using the
/// same PNG reader the engine uses so the test does not depend on an image crate.
fn run(low: &Path, device: &str) -> Option<(f64, f64)> {
    let out = std::env::temp_dir().join(format!("maxim-e2e-out-{device}.png"));
    let status = Command::new(binary())
        .arg("--model").arg(MODEL)
        .arg("-i").arg(low)
        .arg("-o").arg(&out)
        .arg("--device").arg(device)
        .status()
        .expect("run the engine");
    assert!(status.success(), "the engine exited with {status}");

    let read = |p: &Path| -> f64 {
        let img = maxim::image::read_png(p.to_str().unwrap()).expect("read the PNG");
        let sum: f64 = img.data.iter().map(|v| *v as f64).sum();
        sum / img.data.len() as f64
    };
    Some((read(low), read(&out)))
}

#[test]
fn the_enhancement_model_brightens_a_dark_image() {
    let Some(low) = have_inputs() else {
        eprintln!("skipped: needs {MODEL} and {ARCHIVE}");
        return;
    };
    let (before, after) = run(&low, "cpu").expect("run the CPU backend");
    // The LOL low-light inputs sit at ~0.08 mean level and the model is trained
    // to bring them to the ground truth's ~0.53. The bar is deliberately far
    // from both: this test is about the model RUNNING and doing its job, not
    // about a particular level, which the PSNR eval in tools/eval.py measures.
    assert!(
        before < 0.2,
        "the test image is not a low-light input (mean level {before:.3})"
    );
    assert!(
        after > before + 0.2,
        "the model did not brighten the image: {before:.3} -> {after:.3}"
    );
    assert!(
        after < 0.95,
        "the model saturated the image ({after:.3}) - that is a bug, not enhancement"
    );
}

/// The GPU and CPU runs must be the same picture. `--verify-gpu` compares them
/// op by op, which is what localises a bug; this checks the thing a user sees,
/// and it is the one test that would catch a backend that is right on every op
/// and wrong overall (a stale arena, a missing readback).
#[cfg(feature = "cuda")]
#[test]
fn the_gpu_agrees_with_the_cpu() {
    let Some(low) = have_inputs() else {
        eprintln!("skipped: needs {MODEL} and {ARCHIVE}");
        return;
    };
    let (_, cpu) = run(&low, "cpu").expect("run the CPU backend");
    let (_, gpu) = run(&low, "gpu").expect("run the GPU backend");
    assert!(
        (cpu - gpu).abs() < 0.002,
        "the backends disagree on the output's mean level: cpu {cpu:.5}, gpu {gpu:.5}"
    );
}
