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

    /// The variant a checkpoint was trained with, read from the file.
    ///
    /// THE ARCHITECTURE IS A PROPERTY OF THE MODEL, NOT A FLAG. Three things
    /// separate the six variants and all three are visible in the weights: the
    /// feature width is the second dimension of the first conv's kernel, the
    /// stage count is how many `stage_N_*` groups exist, and the task is in the
    /// checkpoint's own `__metadata__` when `tools/convert.py` put it there. A
    /// caller who passed the wrong variant by hand would build a graph that reads
    /// parameters the file does not have, and would fail deep inside the builder
    /// rather than at the door, so there is no flag for it.
    pub fn of_checkpoint(file: &lightgpu::safetensors::File) -> Result<Config, crate::Error> {
        // The width is `c_out` of the first encoder conv, which is the one
        // parameter no other variant of a different width shares a shape with.
        let name = "stage_0_encoder_block_0/Conv_0/kernel";
        let s = file.shape(name).map_err(|_| -> crate::Error {
            format!("{name} is missing (not a converted MAXIM checkpoint?)").into()
        })?;
        let features = *s.last().unwrap_or(&0);
        let mut stages = 1;
        for n in 1..3 {
            if file.contains(&format!("stage_{n}_output_conv_1/kernel")) {
                stages = n + 1;
            }
        }
        let variant = match (features, stages) {
            (32, 1) => "S-1",
            (32, 2) => "S-2",
            (32, 3) => "S-3",
            (64, 1) => "M-1",
            (64, 2) => "M-2",
            (64, 3) => "M-3",
            _ => {
                return Err(format!(
                    "checkpoint has {features} features and {stages} stage(s), which is not one of the six published MAXIM variants"
                )
                .into())
            }
        };
        Config::variant(variant)
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

}
