@group(0) @binding({{binding}}) var {{name}}: {{texture}};

fn load_{{name}}(index: u32) -> f32 {
    let c = {{c}};
    let dims = vec2<f32>(textureDimensions({{name}}));
    {{crop}}
    let uv = (vec2<f32>(f32({{x}}), f32({{y}})) + 0.5) / vec2<f32>({{width}}.0, {{height}}.0);
    let pos = min(vec2<i32>((dims - extent) * 0.5 + uv * extent), vec2<i32>(dims) - 1);
    let rgb = textureLoad({{name}}, pos{{level}}).rgb;
    return (rgb[c] * {{scale}} - {{mean}}[c]) / {{std}}[c];
}
