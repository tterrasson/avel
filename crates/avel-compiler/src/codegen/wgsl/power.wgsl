fn avel_pow(x: f32, y: f32) -> f32 {
    if (y == 0.0 || x == 1.0) { return 1.0; }
    let a = abs(x);
    let infinity = avel_bits(0x7f800000u);
    // C pow: (-1)^(+-inf) is 1, where exp2(inf * log2(1)) would be NaN.
    if (a == 1.0 && abs(y) == infinity) { return 1.0; }
    var magnitude = pow(select(a, 1.0, a == 0.0), y);
    if (a == 0.0 && y != 0.0) {
        magnitude = select(0.0, infinity, y < 0.0);
    }
    // A finite negative base has no real non-integer power; -inf does.
    if (x < 0.0 && a != infinity && trunc(y) != y) {
        return avel_bits(0x7fc00000u);
    }
    let odd = abs(y) % 2.0 == 1.0;
    return select(magnitude, -magnitude, (bitcast<u32>(x) >> 31u) != 0u && odd);
}

fn avel_pow_odd(x: f32, y: f32) -> f32 {
    let a = abs(x);
    let magnitude = select(pow(select(a, 1.0, a == 0.0), y),
        select(0.0, avel_bits(0x7f800000u), y < 0.0), a == 0.0);
    return select(magnitude, -magnitude, (bitcast<u32>(x) >> 31u) != 0u);
}

fn avel_pow_int(x: i32, y: i32) -> i32 {
    if (y < 0) {
        return select(select(0, 1, x == 1), select(1, -1, (y & 1) != 0), x == -1);
    }
    var base = x;
    var exponent = u32(y);
    var value = 1;
    while (exponent != 0u) {
        if ((exponent & 1u) != 0u) {
            value *= base;
        }
        base *= base;
        exponent >>= 1u;
    }
    return value;
}
