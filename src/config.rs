//! Model variants and the task table, from `maxim/models/maxim.py` and
//! `maxim/run_eval.py`.

/// One MAXIM configuration. The defaults in the Flax module are the `S-3`
/// geometry with `M` features made explicit here, so a variant is readable
/// without cross-referencing two files.
#[derive(Clone, Debug)]
pub struct Config {
    pub features: usize,
    pub depth: usize,
    pub num_stages: usize,
    pub num_groups: usize,
    pub num_supervision_scales: usize,
    pub num_outputs: usize,
    pub use_cross_gating: bool,
    pub high_res_stages: usize,
    pub block_size_hr: usize,
    pub block_size_lr: usize,
    /// The gMLP's grid size in CELLS per axis. The reference passes its
    /// `grid_size` through unchanged, so unlike `block_size` there is no
    /// pixel/cell ambiguity - but note the reference's low-res GRID size is
    /// `block_size_lr`, which is why this defaults to the same 8.
    pub grid_size_hr: usize,
    pub grid_size_lr: usize,
    pub channels_reduction: usize,
    pub num_bottleneck_blocks: usize,
}

impl Config {
    /// `Model(variant=...)` from the reference, which only overrides the fields
    /// named in `_MODEL_VARIANT_DICT`'s configs; everything else keeps the
    /// module defaults.
    pub fn variant(v: &str) -> Result<Config, crate::Error> {
        let (features, num_stages) = match v {
            "S-1" => (32, 1),
            "S-2" => (32, 2),
            "S-3" => (32, 3),
            "M-1" => (64, 1),
            "M-2" => (64, 2),
            "M-3" => (64, 3),
            _ => return Err(format!("unknown variant {v} (expected S-1..S-3, M-1..M-3)").into()),
        };
        Ok(Config {
            features,
            depth: 3,
            num_stages,
            num_groups: 2,
            num_supervision_scales: 3,
            num_outputs: 3,
            use_cross_gating: true,
            high_res_stages: 2,
            block_size_hr: 16,
            block_size_lr: 8,
            grid_size_hr: 16,
            grid_size_lr: 8,
            channels_reduction: 4,
            num_bottleneck_blocks: 2,
        })
    }

    /// `_MODEL_VARIANT_DICT`: the variant the released checkpoint for a task was
    /// trained with.
    pub fn for_task(task: &str) -> Result<Config, crate::Error> {
        let v = match task.to_ascii_lowercase().as_str() {
            "denoising" | "deblurring" => "S-3",
            "deraining" | "dehazing" | "enhancement" => "S-2",
            other => {
                return Err(format!(
                    "unknown task {other} (expected denoising, deblurring, deraining, dehazing, enhancement)"
                )
                .into())
            }
        };
        Config::variant(v)
    }

    /// The block/grid size at a given encoder/decoder level: the high-res value
    /// below `high_res_stages`, the low-res one above. Note the reference uses
    /// `block_size_lr` for the grid size in the low-res case, which is a quirk of
    /// the published code rather than a typo here.
    pub fn block_size(&self, level: usize) -> usize {
        if level < self.high_res_stages {
            self.block_size_hr
        } else {
            self.block_size_lr
        }
    }

    /// features at a level, and in a "6c" form for the gMLP splits.
    pub fn channels(&self, level: usize) -> usize {
        (1 << level) * self.features
    }

    pub fn n_stage_fuse(&self, stage: usize) -> usize {
        // stage > 0 fuses the previous stage's SAM features; the parameter names
        // in the checkpoint are stage_{s}_input_fuse_sam_{i}.
        stage
    }
}
