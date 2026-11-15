//! Browser-independent extension ABI and tile choices.
use crate::{Attr, CompileError, Dtype, Limits, Node};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
#[cfg(feature = "ts")]
use tsify::Tsify;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct CustomKernel {
    pub shape: Vec<u64>,
    #[serde(default)]
    pub requested_shape: Option<Vec<u64>>,
    /// Group 0: dense f32 inputs in argument order, then output. Bindings use
    /// read_write for arena usage compatibility; kernels must only write output.
    pub wgsl: String,
    pub workgroups: [u32; 3],
}
impl CustomKernel {
    pub(crate) fn validate(&self, inputs: usize, limits: &Limits) -> Result<(), CompileError> {
        if inputs + 1 > limits.max_storage_buffers_per_shader_stage as usize
            || self.workgroups.iter().any(|&n| n == 0 || n > 65535)
            || self.wgsl.trim().is_empty()
        {
            return Err(CompileError::InvalidOp {
                op: "custom kernel".into(),
                message: "invalid bindings, workgroups or empty WGSL".into(),
            });
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub enum GpuClass {
    #[default]
    Integrated,
    Discrete,
    Mobile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct TileConfig {
    pub rows: u64,
    pub columns: u64,
    pub split: u64,
    /// Square shared-memory matmul tile, 8 or 16; 0 selects the scalar kernel.
    pub matmul: u32,
}
impl TileConfig {
    pub(crate) fn validate(&self) -> Result<(), CompileError> {
        if ![1, 2, 4].contains(&self.rows)
            || ![1, 2, 4].contains(&self.columns)
            || ![1, 2, 4, 8, 16].contains(&self.split)
            || ![0, 8, 16].contains(&self.matmul)
        {
            return Err(CompileError::InvalidOp {
                op: "tile configuration".into(),
                message: "unsupported tile size or reduction split".into(),
            });
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
pub struct ShapeInput {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
}

/// Inferred output descriptors, serialized as an array at the wasm boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(transparent)]
pub struct ShapeOutputs(pub Vec<ShapeInput>);

/// Shape and dtype inference for the explicit builder and custom-op
/// registration pass: every computed output of `node`, in schema order.
pub fn infer_outputs(
    node: &Node,
    inputs: &[ShapeInput],
    symbols: &BTreeMap<String, u64>,
) -> Result<Vec<ShapeInput>, CompileError> {
    let shapes: Vec<&[u64]> = inputs.iter().map(|v| v.shape.as_slice()).collect();
    let dtypes: Vec<Dtype> = inputs.iter().map(|v| v.dtype).collect();
    let symbols: HashMap<String, u64> = symbols.iter().map(|(k, v)| (k.clone(), *v)).collect();
    let (_, outputs) = crate::ops::resolve(node, &shapes, &dtypes, &symbols)?;
    Ok(outputs
        .into_iter()
        .map(|output| ShapeInput {
            dtype: output.dtype,
            shape: output.shape,
        })
        .collect())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
pub struct ShapeRequest {
    pub node: Node,
    pub inputs: Vec<ShapeInput>,
    pub symbols: BTreeMap<String, u64>,
}

/// Attributes to resolve against concrete symbol sizes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
pub struct SymbolRequest {
    #[cfg_attr(feature = "ts", tsify(type = "Record<string, Attr>"))]
    pub attributes: BTreeMap<String, Attr>,
    pub symbols: BTreeMap<String, u64>,
}

/// Attributes without symbolic integers, as custom ops receive them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
pub struct ResolvedAttributes {
    #[cfg_attr(feature = "ts", tsify(type = "Record<string, Attr>"))]
    pub attributes: BTreeMap<String, Attr>,
}

/// Replaces every symbolic integer in `attributes` with its value.
pub fn resolve_attributes(
    attributes: &BTreeMap<String, Attr>,
    symbols: &BTreeMap<String, u64>,
) -> Result<BTreeMap<String, Attr>, CompileError> {
    let symbols: HashMap<String, u64> = symbols.iter().map(|(k, v)| (k.clone(), *v)).collect();
    let eval = |expr: &str| crate::ops::eval_int(expr, &symbols);
    attributes
        .iter()
        .map(|(name, value)| {
            let value = value
                .resolve(&eval)
                .map_err(|message| CompileError::InvalidOp {
                    op: format!("attribute `{name}`"),
                    message,
                })?;
            Ok((name.clone(), value))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Attr;

    fn node(op: &str, inputs: &[&str]) -> Node {
        Node {
            op: op.to_owned(),
            inputs: inputs.iter().map(|name| Some(name.to_string())).collect(),
            outputs: Vec::new(),
            attributes: BTreeMap::new(),
        }
    }

    fn symbols(pairs: &[(&str, u64)]) -> BTreeMap<String, u64> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), *value))
            .collect()
    }

    #[test]
    fn infer_outputs_returns_every_output_in_schema_order() {
        let outputs = infer_outputs(
            &node("aten.relu.default", &["x"]),
            &[ShapeInput {
                dtype: Dtype::F32,
                shape: vec![2, 3],
            }],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(
            outputs,
            vec![ShapeInput {
                dtype: Dtype::F32,
                shape: vec![2, 3]
            }]
        );

        // topk computes values and indices.
        let mut topk = node("aten.topk.default", &["x"]);
        topk.attributes = BTreeMap::from([
            ("k".to_owned(), Attr::Int(2)),
            ("dim".to_owned(), Attr::Int(-1)),
        ]);
        let outputs = infer_outputs(
            &topk,
            &[ShapeInput {
                dtype: Dtype::F32,
                shape: vec![2, 5],
            }],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(
            outputs,
            vec![
                ShapeInput {
                    dtype: Dtype::F32,
                    shape: vec![2, 2]
                },
                ShapeInput {
                    dtype: Dtype::I64,
                    shape: vec![2, 2]
                },
            ]
        );
    }

    #[test]
    fn infer_outputs_rejects_invalid_shapes_and_unknown_ops() {
        assert!(
            infer_outputs(
                &node("aten.view.default", &["x"]),
                &[ShapeInput {
                    dtype: Dtype::F32,
                    shape: vec![2, 3]
                }],
                &BTreeMap::new(),
            )
            .is_err()
        );
        assert!(matches!(
            infer_outputs(
                &node("aten.not_an_op.default", &["x"]),
                &[ShapeInput {
                    dtype: Dtype::F32,
                    shape: vec![3]
                }],
                &BTreeMap::new(),
            ),
            Err(CompileError::UnsupportedOp(_))
        ));
    }

    #[test]
    fn resolve_attributes_substitutes_symbols_and_rejects_unknown_ones() {
        let attributes = BTreeMap::from([
            ("rows".to_owned(), Attr::Sym("s27".to_owned())),
            (
                "config".to_owned(),
                Attr::Map(BTreeMap::from([(
                    "nested".to_owned(),
                    Attr::List(vec![Attr::Int(1), Attr::Sym("2*s27".to_owned())]),
                )])),
            ),
            ("plain".to_owned(), Attr::Int(7)),
        ]);
        let resolved = resolve_attributes(&attributes, &symbols(&[("s27", 17)])).unwrap();
        assert_eq!(
            resolved,
            BTreeMap::from([
                ("rows".to_owned(), Attr::Int(17)),
                (
                    "config".to_owned(),
                    Attr::Map(BTreeMap::from([(
                        "nested".to_owned(),
                        Attr::List(vec![Attr::Int(1), Attr::Int(34)])
                    )]))
                ),
                ("plain".to_owned(), Attr::Int(7)),
            ])
        );
        assert!(resolve_attributes(&attributes, &BTreeMap::new()).is_err());
    }
}
