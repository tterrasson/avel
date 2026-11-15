use thiserror::Error;

use crate::graph::Dtype;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CompileError {
    #[error("unsupported graph format version {found}, expected {expected}")]
    UnsupportedFormat { found: u32, expected: u32 },
    #[error("no shape given for input `{0}`")]
    MissingInputShape(String),
    #[error("shape given for unknown input `{0}`")]
    UnknownInput(String),
    #[error("value `{name}` has rank {found}, expected {expected}")]
    RankMismatch {
        name: String,
        found: usize,
        expected: usize,
    },
    #[error("value `{name}` has size {found} at dimension {axis}, expected {expected}")]
    DimMismatch {
        name: String,
        axis: usize,
        found: u64,
        expected: String,
    },
    #[error("output `{0}` is not produced by the graph")]
    UnknownOutput(String),
    #[error("output `{name}` is {found:?}, expected {expected:?}")]
    OutputTypeMismatch {
        name: String,
        found: Dtype,
        expected: Dtype,
    },
    #[error("unsupported op `{0}`")]
    UnsupportedOp(String),
    #[error("invalid `{op}`: {message}")]
    InvalidOp { op: String, message: String },
    #[error("unknown value `{0}`")]
    UnknownValue(String),
    #[error("value `{name}` is {dtype:?}, which plans cannot hold")]
    UnsupportedDtype { name: String, dtype: Dtype },
    #[error("value `{name}` is {found:?}, the exporter says {expected:?}")]
    DtypeMismatch {
        name: String,
        found: Dtype,
        expected: Dtype,
    },
    #[error("symbol `{0}` has inconsistent sizes across the inputs")]
    InconsistentSymbol(String),
    #[error("tensor `{0}` is empty")]
    EmptyTensor(String),
    #[error("f16 activations are unavailable: {0}")]
    F16Unavailable(&'static str),
    #[error("activations need a {size} byte arena, over the {limit} byte buffer limit")]
    ArenaTooLarge { size: u64, limit: u64 },
    #[error("tensor `{name}` needs {size} bytes, over the {limit} byte binding limit")]
    BufferTooLarge { name: String, size: u64, limit: u64 },
    #[cfg(feature = "cpu")]
    #[error("CPU backend does not support {0}")]
    CpuUnsupported(String),
    #[cfg(feature = "cpu")]
    #[error("CPU parameter `{name}`: {message}")]
    CpuParameter { name: String, message: &'static str },
    #[cfg(feature = "cpu")]
    #[error("could not allocate {bytes} bytes for the CPU plan")]
    CpuAllocation { bytes: usize },
}
