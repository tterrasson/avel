//! Compiles an exported avel graph into a frozen execution plan.
//!
//! The crate has no browser dependency: it runs natively for tests and the
//! CLI, and in the browser through `avel-wasm`.

mod codegen;
mod compile;
#[cfg(feature = "cpu")]
pub mod cpu;
mod error;
mod extension;
mod fuse;
mod graph;
mod kernel;
mod layout;
mod media;
mod memory;
mod ops;
mod plan;
mod view;

pub use compile::{CompileOptions, Features, Limits, compile, validate};
pub use error::CompileError;
pub use extension::{
    CustomKernel, GpuClass, ResolvedAttributes, ShapeInput, ShapeOutputs, ShapeRequest,
    SymbolRequest, TileConfig, infer_outputs, resolve_attributes,
};
pub use graph::{Attr, Dim, Dtype, FORMAT_VERSION, Graph, Metadata, Node, Parameter, ValueInfo};
pub use media::{ImageLayout, ImageResize, MediaInput, MediaSource};
pub use ops::{PRESERVED_OPS, SUPPORTED_OPS};
pub use plan::{
    Binding, BufferDesc, BufferRole, Dispatch, PLAN_VERSION, Plan, RequestedShapes, Shader,
    TensorDesc,
};
