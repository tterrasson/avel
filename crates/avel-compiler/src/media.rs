//! Browser-independent media descriptors and fused texture reads.
use crate::codegen::wgsl;
use crate::{CompileError, Dtype, TensorDesc};
use serde::{Deserialize, Serialize};
#[cfg(feature = "ts")]
use tsify::Tsify;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum MediaSource {
    External,
    Image { width: u32, height: u32 },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "lowercase")]
pub enum ImageLayout {
    #[default]
    Nchw,
    Nhwc,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub enum ImageResize {
    #[default]
    Stretch,
    CenterCrop,
}

fn ones() -> [f32; 3] {
    [1.0; 3]
}
fn one() -> f32 {
    1.0
}

/// RGB values in sRGB [0, 1], transformed as `(pixel * scale - mean) / std`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(Tsify))]
#[serde(rename_all = "camelCase")]
pub struct MediaInput {
    pub source: MediaSource,
    #[serde(default)]
    pub layout: ImageLayout,
    #[serde(default)]
    pub resize: ImageResize,
    #[serde(default)]
    pub mean: [f32; 3],
    #[serde(default = "ones")]
    pub std: [f32; 3],
    #[serde(default = "one")]
    pub scale: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MediaRead {
    pub config: MediaInput,
    pub width: u64,
    pub height: u64,
}

impl MediaInput {
    pub(crate) fn resolve(self, input: &TensorDesc) -> Result<MediaRead, CompileError> {
        let invalid = |message: &str| CompileError::InvalidOp {
            op: format!("media input `{}`", input.name),
            message: message.to_owned(),
        };
        let shape = &input.shape;
        if input.dtype != Dtype::F32 || shape.len() != 4 || shape[0] != 1 {
            return Err(invalid(
                "requires an f32 image with rank 4 and batch size 1",
            ));
        }
        let (channels, height, width) = match self.layout {
            ImageLayout::Nchw => (shape[1], shape[2], shape[3]),
            ImageLayout::Nhwc => (shape[3], shape[1], shape[2]),
        };
        if channels != 3 {
            return Err(invalid("requires exactly three RGB channels"));
        }
        if !self.scale.is_finite()
            || self.mean.iter().any(|x| !x.is_finite())
            || self.std.iter().any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(invalid(
                "normalization must be finite and std must be positive",
            ));
        }
        if matches!(
            self.source,
            MediaSource::Image { width: 0, .. } | MediaSource::Image { height: 0, .. }
        ) {
            return Err(invalid("image source dimensions must be positive"));
        }
        Ok(MediaRead {
            config: self,
            width,
            height,
        })
    }
}

impl MediaRead {
    pub fn declaration(self, name: &str, binding: usize) -> String {
        let external = self.config.source == MediaSource::External;
        let texture = if external {
            "texture_external"
        } else {
            "texture_2d<f32>"
        };
        let (c, x, y) = match self.config.layout {
            ImageLayout::Nchw => (
                format!("index / {}u", self.width * self.height),
                format!("index % {}u", self.width),
                format!("index / {}u % {}u", self.width, self.height),
            ),
            ImageLayout::Nhwc => (
                "index % 3u".to_owned(),
                format!("index / 3u % {}u", self.width),
                format!("index / {}u", self.width * 3),
            ),
        };
        let crop = if self.config.resize == ImageResize::CenterCrop {
            wgsl!("media_crop.wgsl", width = self.width, height = self.height,)
        } else {
            "let extent = dims;".to_owned()
        };
        let level = if external { "" } else { ", 0" };
        let vector = |v: [f32; 3]| format!("vec3<f32>({:?}, {:?}, {:?})", v[0], v[1], v[2]);
        wgsl!(
            "media_read.wgsl",
            binding = binding,
            name = name,
            texture = texture,
            c = c,
            crop = crop,
            x = x,
            y = y,
            width = self.width,
            height = self.height,
            level = level,
            scale = format_args!("{:?}", self.config.scale),
            mean = vector(self.config.mean),
            std = vector(self.config.std),
        )
    }
}
