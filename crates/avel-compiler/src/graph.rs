//! The `model.graph.json` format written by the Python exporter.

use std::collections::BTreeMap;

use serde::de::{self, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(feature = "ts")]
use tsify::Tsify;

pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct Graph {
    pub format_version: u32,
    pub opset: String,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub nodes: Vec<Node>,
    pub parameters: Vec<Parameter>,
    /// Media inputs read straight from images or video frames, by input name.
    #[serde(default)]
    pub media: BTreeMap<String, crate::MediaInput>,
    /// Free-form data for the application. The compiler skips it unread.
    #[serde(default)]
    #[cfg_attr(feature = "ts", tsify(type = "Record<string, unknown>"))]
    pub metadata: Metadata,
    /// The exporter validated the model with f16 activations, so plans may
    /// store them in f16.
    #[serde(default)]
    pub f16: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    F32,
    F16,
    I32,
    /// A PyTorch `int64`. WGSL has no 64-bit integers, so values are narrowed
    /// to `i32` in buffers: the exporter checks that they fit, and the runtime
    /// narrows inputs on upload and widens outputs on readback.
    I64,
    I8,
    Bool,
}

impl Dtype {
    /// Size of one stored element in bytes. Booleans are stored as `u32`, and
    /// `i64` values as `i32`.
    pub fn size(self) -> u64 {
        match self {
            Dtype::F32 | Dtype::I32 | Dtype::I64 | Dtype::Bool => 4,
            Dtype::F16 => 2,
            Dtype::I8 => 1,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Dtype::F32 | Dtype::F16)
    }

    pub fn is_int(self) -> bool {
        matches!(self, Dtype::I32 | Dtype::I64 | Dtype::I8)
    }
}

/// A dimension: a fixed size, or a named size with optional bounds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(untagged)]
pub enum Dim {
    Fixed(u64),
    Symbolic {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<u64>,
    },
}

impl Dim {
    pub fn accepts(&self, size: u64) -> bool {
        match self {
            Dim::Fixed(fixed) => *fixed == size,
            Dim::Symbolic { min, max, .. } => {
                min.is_none_or(|min| size >= min) && max.is_none_or(|max| size <= max)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct ValueInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<Dim>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct Node {
    pub op: String,
    /// One entry per tensor argument in schema order, `None` for an optional
    /// tensor left out. A `Tensor[]` argument contributes one entry per tensor.
    pub inputs: Vec<Option<String>>,
    /// One entry per output in schema order, `None` for the outputs the
    /// graph never reads (such as the indices of a max pooling).
    pub outputs: Vec<Option<ValueInfo>>,
    /// Non-tensor arguments, keyed by schema name.
    #[serde(default)]
    #[cfg_attr(feature = "ts", tsify(type = "Record<string, Attr>"))]
    pub attributes: BTreeMap<String, Attr>,
}

impl Node {
    /// Names of the tensor arguments present, in schema order.
    pub fn tensors(&self) -> impl Iterator<Item = &str> {
        self.inputs.iter().flatten().map(String::as_str)
    }
}

/// A weight or buffer stored in `model.safetensors`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct Parameter {
    pub name: String,
    pub tensor: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Set for persistent state, such as a buffer the model mutates: the
    /// graph value holding its new content. The tensor is then the initial
    /// state, and every run starts from the state the previous one left.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update: Option<String>,
}

/// A non-tensor argument. Integers that depend on a dynamic input dimension
/// are symbolic: `{"$sym": "2*s27"}`, resolved against the input shapes.
/// The single-key `$sym` object is reserved; any other object is a map, kept
/// for the attributes of custom ops.
#[derive(Debug, Clone, PartialEq)]
pub enum Attr {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Sym(String),
    List(Vec<Attr>),
    Map(BTreeMap<String, Attr>),
}

#[cfg(feature = "ts")]
#[wasm_bindgen::prelude::wasm_bindgen(typescript_custom_section)]
const ATTR_TS: &str = r#"
/** A non-tensor argument; `{ $sym: expr }` is a symbolic integer. */
export type Attr = null | boolean | number | string | { $sym: string } | Attr[] | { [key: string]: Attr };
"#;

const SYM: &str = "$sym";

impl Attr {
    /// A plain integer, without resolving symbols.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Attr::Int(value) => Some(*value),
            _ => None,
        }
    }

    /// Replaces every symbolic integer with its value.
    pub fn resolve(&self, eval: &impl Fn(&str) -> Result<i64, String>) -> Result<Attr, String> {
        Ok(match self {
            Attr::Sym(expr) => Attr::Int(eval(expr)?),
            Attr::List(items) => Attr::List(
                items
                    .iter()
                    .map(|item| item.resolve(eval))
                    .collect::<Result<_, _>>()?,
            ),
            Attr::Map(map) => Attr::Map(
                map.iter()
                    .map(|(key, item)| Ok((key.clone(), item.resolve(eval)?)))
                    .collect::<Result<_, String>>()?,
            ),
            other => other.clone(),
        })
    }
}

impl Serialize for Attr {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Attr::Null => serializer.serialize_unit(),
            Attr::Bool(value) => serializer.serialize_bool(*value),
            Attr::Int(value) => serializer.serialize_i64(*value),
            Attr::Float(value) => serializer.serialize_f64(*value),
            Attr::Str(value) => serializer.serialize_str(value),
            Attr::Sym(expr) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(SYM, expr)?;
                map.end()
            }
            Attr::List(items) => items.serialize(serializer),
            Attr::Map(map) => map.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Attr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct AttrVisitor;

        impl<'de> Visitor<'de> for AttrVisitor {
            type Value = Attr;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an attribute")
            }
            fn visit_unit<E>(self) -> Result<Attr, E> {
                Ok(Attr::Null)
            }
            fn visit_none<E>(self) -> Result<Attr, E> {
                Ok(Attr::Null)
            }
            fn visit_bool<E>(self, value: bool) -> Result<Attr, E> {
                Ok(Attr::Bool(value))
            }
            fn visit_i64<E>(self, value: i64) -> Result<Attr, E> {
                Ok(Attr::Int(value))
            }
            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Attr, E> {
                i64::try_from(value)
                    .map(Attr::Int)
                    .map_err(|_| E::custom(format!("integer {value} is too large")))
            }
            fn visit_f64<E>(self, value: f64) -> Result<Attr, E> {
                Ok(Attr::Float(value))
            }
            fn visit_str<E>(self, value: &str) -> Result<Attr, E> {
                Ok(Attr::Str(value.to_owned()))
            }
            fn visit_string<E>(self, value: String) -> Result<Attr, E> {
                Ok(Attr::Str(value))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Attr, A::Error> {
                let mut items = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Attr::List(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Attr, A::Error> {
                let mut map = BTreeMap::new();
                let mut sym = None;
                while let Some(key) = access.next_key::<String>()? {
                    if key == SYM {
                        sym = Some(access.next_value::<String>()?);
                    } else {
                        map.insert(key, access.next_value()?);
                    }
                }
                match sym {
                    None => Ok(Attr::Map(map)),
                    Some(expr) if map.is_empty() => Ok(Attr::Sym(expr)),
                    Some(_) => Err(de::Error::custom("`$sym` must be the only key")),
                }
            }
        }

        deserializer.deserialize_any(AttrVisitor)
    }
}

/// Application metadata. The compiler never reads it, so it is skipped while
/// parsing and serializes as an empty object.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Metadata;

impl Serialize for Metadata {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_map(Some(0))?.end()
    }
}

impl<'de> Deserialize<'de> for Metadata {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        IgnoredAny::deserialize(deserializer).map(|_| Metadata)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn attr(value: serde_json::Value) -> Result<Attr, serde_json::Error> {
        serde_json::from_value(value)
    }

    #[test]
    fn reads_every_attribute_kind() {
        assert_eq!(attr(json!(null)).unwrap(), Attr::Null);
        assert_eq!(attr(json!(true)).unwrap(), Attr::Bool(true));
        assert_eq!(attr(json!(-3)).unwrap(), Attr::Int(-3));
        assert_eq!(attr(json!(0.5)).unwrap(), Attr::Float(0.5));
        assert_eq!(attr(json!("tanh")).unwrap(), Attr::Str("tanh".into()));
        assert_eq!(
            attr(json!([1, {"$sym": "2*s0"}])).unwrap(),
            Attr::List(vec![Attr::Int(1), Attr::Sym("2*s0".into())])
        );
        // Objects other than `$sym` are maps, nested as deep as needed.
        assert_eq!(
            attr(json!({"tile": {"size": 4}})).unwrap(),
            Attr::Map(BTreeMap::from([(
                "tile".into(),
                Attr::Map(BTreeMap::from([("size".into(), Attr::Int(4))]))
            )]))
        );
        assert!(attr(json!({"$sym": "s0", "other": 1})).is_err());
        assert!(attr(json!({"$sym": 3})).is_err());
        assert!(attr(json!(u64::MAX)).is_err());
    }

    #[test]
    fn attributes_round_trip_through_json() {
        let value = json!({"a": [1, 2.5, "x", null, {"$sym": "s0"}], "b": {"c": false}});
        let parsed = attr(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), value);
    }

    #[test]
    fn resolves_symbols_in_nested_attributes() {
        let parsed =
            attr(json!({"sizes": [{"$sym": "s0"}, 2], "nested": {"n": {"$sym": "s0"}}})).unwrap();
        let eval = |expr: &str| match expr {
            "s0" => Ok(7),
            _ => Err(format!("unknown `{expr}`")),
        };
        assert_eq!(
            serde_json::to_value(parsed.resolve(&eval).unwrap()).unwrap(),
            json!({"sizes": [7, 2], "nested": {"n": 7}})
        );
        assert!(Attr::Sym("s1".into()).resolve(&eval).is_err());
    }

    #[test]
    fn skips_metadata_unread() {
        let metadata: Metadata =
            serde_json::from_value(json!({"labels": ["a", "b"], "deep": {"x": [1, {}]}})).unwrap();
        assert_eq!(serde_json::to_value(metadata).unwrap(), json!({}));
    }
}
