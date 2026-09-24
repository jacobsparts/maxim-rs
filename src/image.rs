//! PNG I/O and the reference eval pipeline's padding/cropping.
//!
//! `run_eval.py` pads to an even shape, then to a multiple of 64 with a centred
//! reflect, and crops the output back centred on the EVEN shape. Reproducing
//! that exactly is what makes the published result images comparable.

use crate::Error;

/// An 8-bit RGB image, `[c][h][w]` with values in 0..1 (what the model takes).
pub struct Image {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn new(c: usize, h: usize, w: usize) -> Image {
        Image { c, h, w, data: vec![0.0; c * h * w] }
    }

    pub fn len(&self) -> usize {
        self.c * self.h * self.w
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Read a PNG as `[3][h][w]`, scaled to 0..1. Palette and grayscale inputs are
/// expanded to RGB, matching `Image.convert('RGB')`.
pub fn read_png(path: &str) -> Result<Image, Error> {
    let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().map_err(|e| format!("{path}: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("{path}: {e}"))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let mut img = Image::new(3, h, w);
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Indexed => {
            return Err(format!("{path}: indexed PNG; re-save it as RGB").into())
        }
    };
    for y in 0..h {
        for x in 0..w {
            let s = (y * w + x) * channels;
            let (r, g, b) = match channels {
                1 => (buf[s], buf[s], buf[s]),
                2 => (buf[s], buf[s], buf[s]),
                _ => (buf[s], buf[s + 1], buf[s + 2]),
            };
            for (i, v) in [r, g, b].iter().enumerate() {
                img.data[i * h * w + y * w + x] = *v as f32 / 255.0;
            }
        }
    }
    Ok(img)
}

/// Write `[3][h][w]` in 0..1 as an 8-bit PNG, rounding half up - what
/// `(x * 255 + 0.5).astype(uint8)` does in the reference.
pub fn write_png(path: &str, img: &Image) -> Result<(), Error> {
    let file = std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), img.w as u32, img.h as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| format!("{path}: {e}"))?;
    let mut out = vec![0u8; 3 * img.h * img.w];
    for y in 0..img.h {
        for x in 0..img.w {
            for c in 0..3 {
                let v = img.data[c * img.h * img.w + y * img.w + x].clamp(0.0, 1.0);
                out[(y * img.w + x) * 3 + c] = (v * 255.0 + 0.5) as u8;
            }
        }
    }
    writer.write_image_data(&out).map_err(|e| format!("{path}: {e}"))?;
    Ok(())
}

/// Flax's `padding='SAME'` for one axis: the pads that make the output
/// `ceil(size / stride)`, asymmetric for odd sizes (the larger pad last).
pub fn same_pad(size: usize, stride: usize, k: usize) -> (usize, usize) {
    let out = (size + stride - 1) / stride;
    let total = ((out - 1) * stride + k).saturating_sub(size);
    (total / 2, total - total / 2)
}

/// `jnp.pad(..., mode='reflect')` on the two spatial axes, `before`/`after` per
/// axis. Reflection excludes the edge value itself, so it needs `pad < dim`.
pub fn reflect_pad2(img: &Image, top: usize, bottom: usize, left: usize, right: usize) -> Image {
    let (h, w) = (img.h, img.w);
    let (nh, nw) = (h + top + bottom, w + left + right);
    let mut out = Image::new(img.c, nh, nw);
    for y in 0..nh {
        let sy = reflect_index(y as isize - top as isize, h);
        for x in 0..nw {
            let sx = reflect_index(x as isize - left as isize, w);
            for c in 0..img.c {
                out.data[c * nh * nw + y * nw + x] = img.data[c * h * w + sy * w + sx];
            }
        }
    }
    out
}

/// torch's / jax's `reflect`: mirror about the edge WITHOUT repeating it.
fn reflect_index(i: isize, n: usize) -> usize {
    if i < 0 {
        (-i) as usize
    } else if i as usize >= n {
        (2 * (n as isize) - 2 - i) as usize
    } else {
        i as usize
    }
}

/// The reference pipeline's preprocessing: make the shape even (pad bottom and
/// right by one with reflect), then pad to the next multiple of `factor` with a
/// CENTRED reflect. Note the reference's odd bound - `((h + factor) // factor) *
/// factor` rounds up to the next multiple strictly above `h` when `h` is not
/// already a multiple - and that the pad is split with a floor, which is why an
/// odd `h` would lose a row. An even `h` cannot, so this is faithful either way.
pub struct Padded {
    pub img: Image,
    pub orig_h: usize,
    pub orig_w: usize,
    pub even_h: usize,
    pub even_w: usize,
}

pub fn preprocess(img: &Image, factor: usize) -> Padded {
    let (h, w) = (img.h, img.w);
    let ph = h % 2;
    let pw = w % 2;
    let even = if ph != 0 || pw != 0 {
        reflect_pad2(img, 0, ph, 0, pw)
    } else {
        copy(img)
    };
    let (even_h, even_w) = (even.h, even.w);
    let hp = if even_h % factor != 0 { ((even_h + factor) / factor) * factor } else { even_h };
    let wp = if even_w % factor != 0 { ((even_w + factor) / factor) * factor } else { even_w };
    let padh = hp - even_h;
    let padw = wp - even_w;
    let padded = if padh != 0 || padw != 0 {
        reflect_pad2(&even, padh / 2, padh / 2, padw / 2, padw / 2)
    } else {
        even
    };
    Padded { img: padded, orig_h: h, orig_w: w, even_h, even_w }
}

pub fn copy(img: &Image) -> Image {
    Image { c: img.c, h: img.h, w: img.w, data: img.data.clone() }
}

/// The reference's crop: centred on the even shape, then to the original size.
pub fn crop_out(img: &Image, p: &Padded) -> Image {
    let hs = img.h / 2 - p.even_h / 2;
    let ws = img.w / 2 - p.even_w / 2;
    let mut out = Image::new(img.c, p.orig_h, p.orig_w);
    for c in 0..img.c {
        for y in 0..p.orig_h {
            for x in 0..p.orig_w {
                out.data[c * p.orig_h * p.orig_w + y * p.orig_w + x] =
                    img.data[c * img.h * img.w + (hs + y) * img.w + (ws + x)];
            }
        }
    }
    out
}

/// Minimal `.npy` writer, so an intermediate can be diffed against the Python
/// reference with `numpy.load` and no extra dependency on either side.
pub fn save_npy(path: &str, c: usize, h: usize, w: usize, data: &[f32]) -> Result<(), Error> {
    use std::io::Write;
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({c}, {h}, {w}), }}"
    );
    let mut hdr = header.into_bytes();
    let base = 10 + hdr.len() + 1;
    let pad = (64 - (base % 64)) % 64;
    hdr.extend(std::iter::repeat(b' ').take(pad));
    hdr.push(b'\n');
    let f = std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
    let mut f = std::io::BufWriter::new(f);
    f.write_all(b"\x93NUMPY\x01\x00").map_err(|e| e.to_string())?;
    f.write_all(&(hdr.len() as u16).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&hdr).map_err(|e| e.to_string())?;
    for v in data {
        f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}
