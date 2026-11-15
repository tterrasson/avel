use std::collections::BTreeMap;

use avel_compiler::{Dtype, Graph, TensorDesc, cpu};
use serde::{Deserialize, Serialize};
use tsify::{Ts, Tsify};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct CpuGraph(cpu::CpuModel);

#[wasm_bindgen]
impl CpuGraph {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: Ts<Graph>) -> Result<CpuGraph, JsError> {
        Ok(Self(cpu::CpuModel::new(graph.to_rust()?)?))
    }

    #[wasm_bindgen(js_name = setParameter)]
    pub fn set_parameter(&mut self, name: &str, bytes: &[u8]) -> Result<(), JsError> {
        Ok(self.0.set_parameter(name, bytes)?)
    }

    pub fn compile(&self, options: Ts<CpuCompileOptions>) -> Result<CpuExecutable, JsError> {
        let options = options.to_rust()?;
        Ok(CpuExecutable(
            self.0.compile(options.input_shapes, options.bucket)?,
        ))
    }
}

#[derive(Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct CpuCompileOptions {
    input_shapes: BTreeMap<String, Vec<u64>>,
    #[serde(default)]
    bucket: Option<u64>,
}

#[wasm_bindgen]
pub struct CpuExecutable(cpu::CpuPlan);

#[derive(Serialize, Tsify)]
pub struct CpuTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<u64>,
    offset: u32,
    length: u32,
}

#[derive(Serialize, Tsify)]
pub struct CpuPlanInfo {
    inputs: Vec<CpuTensor>,
    outputs: Vec<CpuTensor>,
}

impl CpuTensor {
    fn new((desc, words): (&TensorDesc, &[u32])) -> Self {
        Self {
            name: desc.name.clone(),
            dtype: desc.dtype,
            shape: desc.shape.clone(),
            offset: words.as_ptr() as usize as u32,
            length: words.len() as u32,
        }
    }
}

#[wasm_bindgen]
impl CpuExecutable {
    pub fn info(&self) -> Result<Ts<CpuPlanInfo>, JsError> {
        Ok(Ts::from_rust(&CpuPlanInfo {
            inputs: self.0.inputs().map(CpuTensor::new).collect(),
            outputs: self.0.outputs().map(CpuTensor::new).collect(),
        })?)
    }

    pub fn run(&mut self) {
        self.0.run();
    }
}
