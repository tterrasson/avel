// WGSL rejects NaN and infinity in constant expressions. Decode their
// storage representation in a runtime expression instead.
fn avel_bits(bits: u32) -> f32 {
    return bitcast<f32>(bits);
}
