struct RowState {
    count: f32,
    mean: f32,
    m2: f32,
}

fn welford_add(a: RowState, x: f32) -> RowState {
    let count = a.count + 1.0;
    let delta = x - a.mean;
    let mean = a.mean + delta / count;
    return RowState(count, mean, a.m2 + delta * (x - mean));
}

fn combine(a: RowState, b: RowState) -> RowState {
    if (b.count == 0.0) { return a; }
    if (a.count == 0.0) { return b; }
    let count = a.count + b.count;
    let delta = b.mean - a.mean;
    let fraction = b.count / count;
    return RowState(count, a.mean + delta * fraction, a.m2 + b.m2 + delta * delta * a.count * fraction);
}
