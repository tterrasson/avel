// Reduce to [-pi/4, pi/4] with a split pi/2. The high product is exact
// throughout this range, including on backends that do not fuse fma.
fn avel_trig(x: f32, phase: i32) -> f32 {
    if (x == 0.0) { return select(x, 1.0, phase != 0i); }
    if ((bitcast<u32>(x) & 0x7fffffffu) >= 0x7f800000u) {
        return avel_bits(0x7fc00000u);
    }
    if (abs(x) > 8192.0) {
        // Leave very large arguments to the backend's range reduction.
        if (phase == 0) { return sin(x); }
        return cos(x);
    }
    let turns = round(x * 0.6366197723675814);
    // Two explicit operations also prevent reassociation into a rounded
    // single pi/2 literal on fast-math backends.
    let high = fma(-turns, 1.5703125, x);
    let r = fma(-turns, 0.00048382679489661923, high);
    let quadrant = (i32(turns) + phase) & 3i;
    let z = r * r;
    var value = 0.0;
    if ((quadrant & 1i) == 0i) {
        let p = fma(z, 0.00000275573192, -0.000198412698);
        value = r * fma(z, fma(z, fma(z, p, 0.00833333333), -0.16666666667), 1.0);
    } else {
        let p = fma(z, -0.000000275573192, 0.0000248015873);
        value = fma(z, fma(z, fma(z, fma(z, p, -0.00138888889), 0.04166666667), -0.5), 1.0);
    }
    return select(value, -value, quadrant >= 2i);
}

fn avel_sin(x: f32) -> f32 {
    return avel_trig(x, 0i);
}

fn avel_cos(x: f32) -> f32 {
    return avel_trig(x, 1i);
}
