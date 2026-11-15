//! Exposes the compiler to the TypeScript runtime.

#[cfg(feature = "webgpu")]
use avel_compiler::{CompileOptions, Graph, Plan};
use tsify::Ts;
use wasm_bindgen::prelude::*;

#[cfg(feature = "cpu")]
mod cpu;

/// A graph parsed and validated once, then compiled for any number of input
/// shapes without crossing the wasm boundary again. Owned by one model, which
/// calls `free()` when it is destroyed.
#[wasm_bindgen]
#[cfg(feature = "webgpu")]
pub struct LoadedGraph {
    graph: Graph,
}

#[wasm_bindgen]
#[cfg(feature = "webgpu")]
impl LoadedGraph {
    /// Parses and validates `graph`; later edits to the JS object do not reach it.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: Ts<Graph>) -> Result<LoadedGraph, JsError> {
        let graph = graph.to_rust()?;
        avel_compiler::validate(&graph)?;
        Ok(Self { graph })
    }

    pub fn compile(&self, options: Ts<CompileOptions>) -> Result<Ts<Plan>, JsError> {
        let plan = avel_compiler::compile(&self.graph, &options.to_rust()?)?;
        Ok(Ts::from_rust(&plan)?)
    }
}

/// The dtype and shape of every computed output of an ATen node.
#[wasm_bindgen(js_name = inferOutputs)]
pub fn infer_outputs(
    request: Ts<avel_compiler::ShapeRequest>,
) -> Result<Ts<avel_compiler::ShapeOutputs>, JsError> {
    let request = request.to_rust()?;
    let outputs = avel_compiler::infer_outputs(&request.node, &request.inputs, &request.symbols)?;
    if outputs
        .iter()
        .flat_map(|o| &o.shape)
        .any(|&n| n > u64::from(u32::MAX))
    {
        return Err(JsError::new("dimension exceeds u32"));
    }
    Ok(Ts::from_rust(&avel_compiler::ShapeOutputs(outputs))?)
}

#[wasm_bindgen(js_name = resolveAttributes)]
#[cfg(feature = "webgpu")]
pub fn resolve_attributes(
    request: Ts<avel_compiler::SymbolRequest>,
) -> Result<Ts<avel_compiler::ResolvedAttributes>, JsError> {
    let request = request.to_rust()?;
    let attributes = avel_compiler::resolve_attributes(&request.attributes, &request.symbols)?;
    Ok(Ts::from_rust(&avel_compiler::ResolvedAttributes {
        attributes,
    })?)
}
