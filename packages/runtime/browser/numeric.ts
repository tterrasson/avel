/** Numerical comparisons shared by browser validation and benchmark checks. */
export function assertClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: { rtol: number; atol: number },
  label: string,
  scaled = false,
): void {
  if (actual.length !== expected.length) throw new Error(`${label}: output shape differs`);
  let scale = 0;
  if (scaled) {
    for (let i = 0; i < expected.length; i++) {
      if (Number.isFinite(expected[i]!)) scale = Math.max(scale, Math.abs(expected[i]!));
    }
  }
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i]!,
      b = expected[i]!;
    if (a === b || (Number.isNaN(a) && Number.isNaN(b))) continue;
    const slack = tolerance.atol + tolerance.rtol * (scaled ? scale : Math.abs(b));
    if (!Number.isFinite(a) || !Number.isFinite(b) || !(Math.abs(a - b) <= slack)) {
      throw new Error(
        `${label}[${i}]: got ${a}, expected ${b} (rtol ${tolerance.rtol}, atol ${tolerance.atol})`,
      );
    }
  }
}

/** Benchmarks require finite outputs; NaN must never serialize to a passing null drift. */
export function relativeError(actual: Float32Array, expected: Float32Array): number {
  if (actual.length !== expected.length) throw new Error("benchmark output shape differs");
  let scale = 0,
    worst = 0;
  for (let i = 0; i < expected.length; i++) {
    if (!Number.isFinite(actual[i]!) || !Number.isFinite(expected[i]!)) {
      throw new Error(`non-finite benchmark output at ${i}`);
    }
    scale = Math.max(scale, Math.abs(expected[i]!));
    worst = Math.max(worst, Math.abs(actual[i]! - expected[i]!));
  }
  return worst / Math.max(scale, 1e-30);
}
