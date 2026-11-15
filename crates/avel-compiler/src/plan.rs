//! The frozen execution plan handed to the TypeScript runtime.

use serde::{Deserialize, Serialize};
#[cfg(feature = "ts")]
use tsify::Tsify;

use crate::graph::Dtype;

pub const PLAN_VERSION: u32 = 6;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct Plan {
    pub version: u32,
    pub media: std::collections::BTreeMap<String, crate::MediaInput>,
    pub inputs: Vec<TensorDesc>,
    pub outputs: Vec<TensorDesc>,
    pub shaders: Vec<Shader>,
    pub buffers: Vec<BufferDesc>,
    /// Run once when the plan is created, to fill the `Constant` buffers.
    pub prepare: Vec<Dispatch>,
    /// Run on every inference.
    pub dispatches: Vec<Dispatch>,
    /// Set when the plan was compiled with a bucket size: the shapes the
    /// caller asked for, parallel to `inputs` and `outputs`. The plan itself
    /// is compiled for the bucketed shapes, and the runtime pads inputs up to
    /// them and trims outputs back down.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested: Option<RequestedShapes>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct RequestedShapes {
    pub inputs: Vec<Vec<u64>>,
    pub outputs: Vec<Vec<u64>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct TensorDesc {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Index into [`Plan::buffers`].
    pub buffer: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct Shader {
    pub wgsl: String,
    pub entry_point: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct BufferDesc {
    pub size: u64,
    pub role: BufferRole,
    /// Element type of the buffer: `i32`, `i64` and `bool` buffers hold `u32`
    /// words, `i64` values being narrowed to 32 bits.
    pub dtype: Dtype,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum BufferRole {
    /// Written by the runtime before each run. Takes precedence over `Output`
    /// when a graph returns an input unchanged.
    Input,
    /// Read back after each run.
    Output,
    /// Filled once from the named safetensors tensor.
    Parameter { tensor: String },
    /// Persistent state, initialized from the named safetensors tensor and
    /// rewritten by the last dispatches of each run. Shared by every plan of
    /// a model, so a prefill plan and a decode plan see the same state.
    State { tensor: String },
    /// Computed from parameters alone by the `prepare` dispatches.
    Constant,
    /// Holds every intermediate tensor, at offsets planned from their lifetimes.
    Arena,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct Dispatch {
    /// Index into [`Plan::shaders`].
    pub shader: u32,
    pub bindings: Vec<Binding>,
    pub workgroups: [u32; 3],
    /// Graph values the dispatch computes, several when ops were fused.
    pub ops: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct Binding {
    /// Index into [`Plan::buffers`].
    pub buffer: u32,
    pub offset: u64,
    pub size: u64,
}
