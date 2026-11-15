fn avel_is_nan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn avel_max(x: f32, y: f32) -> f32 {
    return select(x, y, y > x || avel_is_nan(y));
}

fn avel_min(x: f32, y: f32) -> f32 {
    return select(x, y, y < x || avel_is_nan(y));
}
