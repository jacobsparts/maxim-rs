//! Parameter access, and the layout conversions the lightgpu kernels need.
//!
//! Flax stores every conv kernel as `[kh, kw, c_in, c_out]` and every Dense as
//! `[c_in, c_out]`. The toolkit's convs want `[c_out][c_in][kh][kw]` and step
//! `c_in` with stride `kh*kw` inside a tap, so the conversions are not just
//! transposes and are worth doing once, here, in an obvious place.
//!
//! The transposed conv is the one that is not a plain rearrangement: flax's
//! `ConvTranspose` is the ADJOINT of its `Conv`, so a scatter reproduces it only
//! with a spatially REVERSED kernel. Getting that wrong costs 65 dB of accuracy
//! (28 dB instead of 93 dB on a decoder block), which is why it is spelled out
//! rather than left to the caller.

use crate::Error;
use lightgpu::safetensors;

pub struct Weights {
    pub file: safetensors::File,
    /// The architecture the checkpoint was trained with, read from the file
    /// itself.
    pub config: crate::config::Config,
}

impl Weights {
    /// Open a converted checkpoint.
    ///
    /// THE ARCHITECTURE IS A PROPERTY OF THE FILE, NOT A FLAG. MAXIM ships six
    /// variants that differ only in feature width and stage count, and the
    /// weights themselves say which one they are - a `stage_2_*` name exists
    /// only in the three-stage models. So the variant is DERIVED here, and there
    /// is no way to ask for one that does not match the file: passing the wrong
    /// variant by hand would build a graph that reads parameters the checkpoint
    /// does not have, which fails deep inside the builder instead of at the
    /// door. `tools/convert.py` records the variant in `__metadata__` so the
    /// answer is stated rather than inferred where it can be, and the naming
    /// below is the fallback for a file converted before that existed.
    pub fn open(path: &str) -> Result<Weights, Error> {
        let file = safetensors::File::open(path).map_err(Error)?;
        let config = crate::config::Config::of_checkpoint(&file)?;
        Ok(Weights { file, config })
    }
    pub fn shape(&self, name: &str) -> Result<&[usize], Error> {
        self.file.shape(name).map_err(Error)
    }

    pub fn has(&self, name: &str) -> bool {
        self.file.contains(name)
    }

    /// A 1-D parameter, e.g. a LayerNorm `scale` or a conv `bias`.
    pub fn vec(&self, name: &str) -> Result<&[f32], Error> {
        self.file.f32(name).map_err(Error)
    }

    /// A `Conv1x1`/`Dense` weight as `[c_out][c_in]`, which is what
    /// `lg_conv1x1` reads (it steps `c_in` contiguously inside an output row).
    pub fn conv1x1(&self, name: &str) -> Result<Vec<f32>, Error> {
        let k = self.vec(name)?;
        let s = self.shape(name)?;
        if s.len() == 2 {
            // Dense: [c_in][c_out] -> [c_out][c_in]
            let (cin, cout) = (s[0], s[1]);
            let mut out = vec![0.0f32; k.len()];
            for i in 0..cout {
                for j in 0..cin {
                    out[i * cin + j] = k[j * cout + i];
                }
            }
            Ok(out)
        } else if s.len() == 4 && s[0] == 1 && s[1] == 1 {
            // Conv1x1: [1, 1, c_in, c_out] -> [c_out][c_in]
            let (cin, cout) = (s[2], s[3]);
            let mut out = vec![0.0f32; k.len()];
            for i in 0..cout {
                for j in 0..cin {
                    out[i * cin + j] = k[j * cout + i];
                }
            }
            Ok(out)
        } else {
            Err(format!("{name}: expected a 1x1 or Dense kernel, got {s:?}").into())
        }
    }

    /// The same weight TRANSPOSED to `[c_in][c_out]`, for this engine's
    /// `mx_conv1x1_t`.
    ///
    /// The toolkit's `lg_conv1x1` reads the `[c_out][c_in]` form because it gives
    /// one thread a single output channel, so its inner loop walks `c_in`
    /// contiguously. That kernel moves about `c_in` times the traffic the op
    /// needs (see `mx_conv1x1_t`), and the fix - several output channels per
    /// thread - makes the weight the *strided* operand unless it is transposed
    /// first. Two layouts, both defensible, chosen by which kernel reads them;
    /// the CPU executor keeps using the `[c_out][c_in]` one, so both blobs are
    /// built for a GPU run and only the first for a CPU one.
    pub fn conv1x1_t(&self, name: &str) -> Result<Vec<f32>, Error> {
        let k = self.vec(name)?;
        let s = self.shape(name)?;
        match s.len() {
            // Dense: already [c_in][c_out].
            2 => {}
            4 if s[0] == 1 && s[1] == 1 => {}
            _ => return Err(format!("{name}: expected a 1x1 or Dense kernel, got {s:?}").into()),
        }
        // No permutation: flax stores both the Dense and the 1x1 conv as
        // `[..., c_in, c_out]`, which is exactly the transposed layout the tiled
        // kernel indexes. The `[c_out][c_in]` form the toolkit reads is the one
        // that has to be built, not this one.
        Ok(k.to_vec())
    }

    /// A k x k conv weight as `[c_out][c_in][k][k]`, the layout
    /// `lg_conv3x3s1p1` reads: it steps `c_in` with stride `k*k` inside a tap.
    pub fn convkxk(&self, name: &str) -> Result<Vec<f32>, Error> {
        let k = self.vec(name)?;
        let s = self.shape(name)?;
        if s.len() != 4 {
            return Err(format!("{name}: expected a 4-D conv kernel, got {s:?}").into());
        }
        let (kh, kw, cin, cout) = (s[0], s[1], s[2], s[3]);
        let mut out = vec![0.0f32; k.len()];
        for oc in 0..cout {
            for ic in 0..cin {
                for y in 0..kh {
                    for x in 0..kw {
                        out[((oc * cin + ic) * kh + y) * kw + x] =
                            k[((y * kw + x) * cin + ic) * cout + oc];
                    }
                }
            }
        }
        Ok(out)
    }

    /// The same k x k conv weight as `[k*k][c_in][c_out]` - the `[tap][ci][oc]`
    /// layout this engine's `mx_conv3x3_t*` read.
    ///
    /// Why a second layout exists at all: with the toolkit's `[c_out][c_in][k][k]`
    /// the four weights a thread needs for its four output channels are 512 floats
    /// apart, so the inner loop of the tiled conv spends four SHARED-MEMORY loads
    /// on them per three multiply-accumulates. Here they are contiguous, so those
    /// four become one 128-bit load. Measured against a host reference, with the
    /// output identical to the element to the old kernel's: 2.245 -> 1.708 ms at
    /// 64->64 @224x320 and 2.616 -> 2.026 at 128->128 @112x160 (the 16x16 form),
    /// 3.406 -> 2.174 at 32->32 @448x640 (the 32x8 form, whose weight loads the
    /// old layout strided further apart still).
    ///
    /// Flax stores `[kh][kw][c_in][c_out]`, so this is not a permutation of the
    /// blob `convkxk` builds but a different walk of the checkpoint's own order.
    pub fn conv3x3_t(&self, name: &str) -> Result<Vec<f32>, Error> {
        let k = self.vec(name)?;
        let s = self.shape(name)?;
        if s.len() != 4 {
            return Err(format!("{name}: expected a 4-D conv kernel, got {s:?}").into());
        }
        let (kh, kw, cin, cout) = (s[0], s[1], s[2], s[3]);
        let mut out = vec![0.0f32; k.len()];
        for y in 0..kh {
            for x in 0..kw {
                for ic in 0..cin {
                    for oc in 0..cout {
                        out[((y * kw + x) * cin + ic) * cout + oc] = k[((y * kw + x) * cin + ic) * cout + oc];
                    }
                }
            }
        }
        Ok(out)
    }

    /// A transposed conv weight as `[c_in][c_out][2][2]`, spatially FLIPPED, for
    /// the engine's scatter kernel `mx_convt2x2s2`.
    pub fn convt2x2(&self, name: &str) -> Result<Vec<f32>, Error> {
        let k = self.vec(name)?;
        let s = self.shape(name)?;
        if s.len() != 4 || s[0] != 2 || s[1] != 2 {
            return Err(format!("{name}: expected a 2x2 transposed kernel, got {s:?}").into());
        }
        let (cin, cout) = (s[2], s[3]);
        let mut out = vec![0.0f32; k.len()];
        for ic in 0..cin {
            for oc in 0..cout {
                for y in 0..2 {
                    for x in 0..2 {
                        out[((ic * cout + oc) * 2 + y) * 2 + x] =
                            k[((1 - y) * 2 + (1 - x)) * cin * cout + ic * cout + oc];
                    }
                }
            }
        }
        Ok(out)
    }

    /// A Dense weight as `[c_out][c_in]`, which is the layout the gating matmul
    /// kernel indexes. The checkpoint stores `[c_in][c_out]`, and the matrices in
    /// this model are square, so the transpose is a real permutation and not a
    /// no-op.
    pub fn dense(&self, name: &str) -> Result<Vec<f32>, Error> {
        let k = self.vec(name)?;
        let s = self.shape(name)?;
        if s.len() != 2 {
            return Err(format!("{name}: expected a Dense kernel, got {s:?}").into());
        }
        let (cin, cout) = (s[0], s[1]);
        let mut out = vec![0.0f32; k.len()];
        for o in 0..cout {
            for i in 0..cin {
                out[o * cin + i] = k[i * cout + o];
            }
        }
        Ok(out)
    }

    /// Copy a parameter out as a plain `Vec`, for storing in the plan.
    pub fn owned(&self, name: &str) -> Result<Vec<f32>, Error> {
        Ok(self.vec(name)?.to_vec())
    }
}
