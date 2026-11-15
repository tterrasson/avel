import { expect, test } from "bun:test";
import { assertClose, relativeError } from "../browser/numeric.js";

test("comparisons reject NaNs and mismatched infinities without weakening scaled tolerance", () => {
  const tolerance = { rtol: 1e-4, atol: 1e-5 };
  expect(() => assertClose([NaN], [1], tolerance, "test")).toThrow();
  expect(() => assertClose([1], [Infinity], tolerance, "test")).toThrow();
  expect(() => assertClose([-Infinity], [Infinity], tolerance, "test")).toThrow();
  expect(() => assertClose([Infinity, 5], [Infinity, 1], tolerance, "test", true)).toThrow();
  expect(() =>
    assertClose([Infinity, NaN, 1.00001], [Infinity, NaN, 1], tolerance, "test", true),
  ).not.toThrow();
  expect(() => assertClose([], [1], tolerance, "test")).toThrow();
});

test("benchmarks reject non-finite outputs before JSON can turn their drift into null", () => {
  for (const value of [NaN, Infinity, -Infinity]) {
    expect(() => relativeError(new Float32Array([value]), new Float32Array([1]))).toThrow(
      "non-finite",
    );
  }
  expect(relativeError(new Float32Array([2, 0]), new Float32Array([1, 0]))).toBe(1);
  expect(() => relativeError(new Float32Array(), new Float32Array(1))).toThrow("shape");
});
