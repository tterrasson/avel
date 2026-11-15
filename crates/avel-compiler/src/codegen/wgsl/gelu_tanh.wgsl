fn gelu_tanh(x: f32) -> f32 {
    let t = clamp(x, -6.0, 6.0);
    return 0.5 * x * (1.0 + tanh(0.79788456 * (t + 0.044715 * t * t * t)));
}
