//! A reflection cannot pad past the dimension it mirrors, and MAXIM's pipeline
//! pads to the next multiple of 64 - so a SMALL image is an input the reference
//! itself refuses, and this engine must say so rather than index past the end of
//! the plane.
//!
//! Found the hard way: a 16x16 PNG aborted with `index out of bounds: the len is
//! 768 but the index is 920` from inside `reflect_pad2`, with or without a CUDA
//! driver, because the pipeline pads a 16-pixel axis by 24 a side. torch, which
//! the reference eval script pads with, raises the same objection in its own
//! words (`Padding size should be less than the corresponding input dimension`).
//! The boundary is not a fixed size and it is not `factor/2`: it is the first
//! even `d` whose centred pad `(ceil(d/64)*64 - d)/2` is smaller than `d`, which
//! is 22 (pad 21) - so 16x16 is refused (pad 24) and 32x32 works (pad 16), and a
//! tall thin image can satisfy one axis and not the other, which is why the
//! message names the axis and the numbers.

use maxim::image::{preprocess, Image};

fn rgb(h: usize, w: usize) -> Image {
    Image { c: 3, h, w, data: (0..3 * h * w).map(|i| (i % 251) as f32 / 251.0).collect() }
}

/// `Result<Padded, Error>` cannot use `expect`: `Padded` holds the plane and has
/// no `Debug` (printing a megabyte of floats on a failure helps nobody), so the
/// error is the only thing worth formatting.
fn ok(r: Result<maxim::image::Padded, maxim::Error>, what: &str) -> maxim::image::Padded {
    match r {
        Ok(p) => p,
        Err(e) => panic!("{what}: {e}"),
    }
}

/// The other direction, for the same reason: `expect_err` wants `Padded: Debug`.
fn err(r: Result<maxim::image::Padded, maxim::Error>, what: &str) -> maxim::Error {
    match r {
        Ok(_) => panic!("{what}: expected a refusal and got a padded plane"),
        Err(e) => e,
    }
}

#[test]
fn a_sixteen_pixel_image_is_refused_with_the_numbers() {
    let m = err(preprocess(&rgb(16, 16), 64), "16x16 must be refused").to_string();
    assert!(m.contains("too small"), "{m}");
    assert!(m.contains("height is 16 pixels"), "{m}");
    assert!(m.contains("24 pixels of padding a side"), "{m}");
    assert!(m.contains("at least 22 pixels"), "{m}");
}

#[test]
fn one_axis_over_the_line_is_refused_by_name() {
    // 16 wide, 64 tall: the height needs no padding at all and the width is over
    // the line, so the refusal has to be about the width.
    let m = err(preprocess(&rgb(64, 16), 64), "a 16-pixel width must be refused").to_string();
    assert!(m.contains("width is 16 pixels"), "{m}");
    assert!(!m.contains("the height is 16"), "{m}");
}

#[test]
fn a_square_over_the_boundary_pads_to_one_factor_block() {
    let p = ok(preprocess(&rgb(32, 32), 64), "32x32 is over the boundary");
    assert_eq!((p.img.h, p.img.w), (64, 64));
    assert_eq!((p.even_h, p.even_w), (32, 32));
    assert_eq!((p.orig_h, p.orig_w), (32, 32));
}

#[test]
fn the_boundary_itself_is_accepted() {
    // 22 is the first even dimension whose centred pad (21) is smaller than it,
    // so it is the smallest a 64-multiple pipeline will reflect.
    let p = ok(preprocess(&rgb(22, 22), 64), "22x22 is exactly the boundary");
    assert_eq!((p.img.h, p.img.w), (64, 64));
}

#[test]
fn an_odd_size_is_made_even_first_and_still_pads() {
    let p = ok(preprocess(&rgb(33, 33), 64), "33x33 is over the boundary");
    assert_eq!((p.even_h, p.even_w), (34, 34));
    assert_eq!((p.img.h, p.img.w), (64, 64));
    assert_eq!(p.img.data.len(), 3 * 64 * 64);
}
