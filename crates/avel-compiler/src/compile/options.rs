use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct CompileOptions {
    /// Concrete shape of each graph input.
    pub input_shapes: BTreeMap<String, Vec<u64>>,
    #[serde(default)]
    pub limits: Limits,
    #[serde(default)]
    pub features: Features,
    /// Store activations and constants in f16. Needs `shader-f16`, and a graph
    /// the exporter validated in f16.
    #[serde(default)]
    pub f16: bool,
    /// Use f16 operands and products in GEMMs, with f32 accumulation.
    #[serde(default)]
    pub f16_compute: bool,
    /// Snap every symbolic input dimension up to a multiple of this, capped
    /// by its declared maximum, so variable-length sequences reuse plans
    /// within a bucket. The plan records the requested shapes alongside.
    #[serde(default)]
    pub bucket: Option<u64>,
    #[serde(default)]
    pub media: BTreeMap<String, crate::MediaInput>,
    /// Specialized custom kernels, keyed by their output value name.
    #[serde(default)]
    pub custom_kernels: BTreeMap<String, crate::CustomKernel>,
    #[serde(default)]
    pub gpu_class: crate::GpuClass,
    /// Per-output tuning overrides. Validated before code generation.
    #[serde(default)]
    pub tiles: BTreeMap<String, crate::TileConfig>,
}

/// Adapter limits, defaulting to the WebGPU specification minimums.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct Limits {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_storage_buffers_per_shader_stage: u32,
    pub min_storage_buffer_offset_alignment: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_buffer_size: 256 << 20,
            max_storage_buffer_binding_size: 128 << 20,
            max_storage_buffers_per_shader_stage: 8,
            min_storage_buffer_offset_alignment: 256,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct Features {
    pub shader_f16: bool,
    /// The `subgroups` feature, with the smallest subgroup size of the
    /// adapter. Cooperative reductions then run as one subgroup per row when
    /// subgroups have at least [`MIN_SUBGROUP`] invocations.
    #[serde(default)]
    pub subgroups: bool,
    #[serde(default)]
    pub subgroup_min_size: u32,
}

/// The smallest subgroup worth a row of a reduction on its own.
pub const MIN_SUBGROUP: u32 = 16;

impl Features {
    /// The workgroup size of a subgroup reduction, if the device has one.
    pub fn subgroup_lanes(&self) -> Option<u32> {
        let size = self.subgroup_min_size;
        (self.subgroups && size >= MIN_SUBGROUP && size.is_power_of_two())
            .then_some(size.min(crate::kernel::REDUCTION_LANES * 2))
    }
}
