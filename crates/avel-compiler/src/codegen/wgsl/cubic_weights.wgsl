fn cubic_near(x: f32) -> f32 {
    let a = -0.75;
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
}

fn cubic_far(x: f32) -> f32 {
    let a = -0.75;
    return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
}

fn cubic_weights(t: f32) -> vec4f {
    return vec4f(cubic_far(t + 1.0), cubic_near(t), cubic_near(1.0 - t), cubic_far(2.0 - t));
}
