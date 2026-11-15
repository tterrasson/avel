// Avoid passing overflowing values to exp. Preserve PyTorch's infinity.
fn avel_exp(x: f32) -> f32 {
    let limit = 88.72283;
    if (x > limit) { return avel_bits(0x7f800000u); }
    return exp(x);
}

fn avel_expm1(x: f32) -> f32 {
    // The polynomial keeps the small result when exp(x) would round to 1.
    if (abs(x) < 0.0625) {
        return x * (1.0 + x * (0.5 + x * (0.16666667 + x * (0.041666667 + x * 0.008333334))));
    }
    return avel_exp(x) - 1.0;
}

fn avel_log(x: f32) -> f32 {
    if (x < 0.0) { return avel_bits(0x7fc00000u); }
    if (x == 0.0) { return avel_bits(0xff800000u); }
    return log(x);
}

fn avel_log1p(x: f32) -> f32 {
    if (abs(x) < 0.0625) {
        return x * (1.0 + x * (-0.5 + x * (0.33333334 + x * (-0.25 + x * (0.2 + x * (-0.16666667 + x * 0.14285715))))));
    }
    return avel_log(1.0 + x);
}

fn avel_tanh(x: f32) -> f32 {
    // min() returns the non-NaN operand, so propagate NaN explicitly.
    if ((bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u) { return x; }
    let m = avel_expm1(-2.0 * min(abs(x), 12.0));
    let magnitude = -m / (2.0 + m);
    return bitcast<f32>((bitcast<u32>(magnitude) & 0x7fffffffu) | (bitcast<u32>(x) & 0x80000000u));
}

fn avel_round_even(x: f32) -> f32 {
    if ((bitcast<u32>(x) & 0x7fffffffu) >= 0x4b000000u) { return x; }
    let magnitude = abs(x);
    let whole = floor(magnitude);
    let fraction = magnitude - whole;
    let rounded = whole + select(0.0, 1.0, fraction > 0.5 || (fraction == 0.5 && whole % 2.0 != 0.0));
    return bitcast<f32>(bitcast<u32>(rounded) | (bitcast<u32>(x) & 0x80000000u));
}

fn avel_remainder_f32(x: f32, y: f32) -> f32 {
    let r = x % y;
    return select(r, r + y, r != 0.0 && ((r < 0.0) != (y < 0.0)));
}

fn avel_remainder_i32(x: i32, y: i32) -> i32 {
    if (y == -1) { return 0; }
    let r = x % y;
    return select(r, r + y, r != 0 && ((r < 0) != (y < 0)));
}
