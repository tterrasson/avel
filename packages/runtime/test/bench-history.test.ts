import { expect, test } from "bun:test";
import { historyTable, percentile } from "../browser/bench-history.js";

test("charts chronological commit medians and marks missing modes", () => {
  const rows = (medianMs: number) => [
    { model: "resnet18", runtime: "avel", medianMs, p10Ms: medianMs, p90Ms: medianMs },
  ];
  const chart = historyTable([
    { revision: "aaaabbbb0000", time: "2026-11-05T12:00:00Z", rows: rows(2) },
    { revision: "ccccdddd0000", time: "2026-11-05T13:00:00Z", rows: [] },
    { revision: "eeeeffff0000", time: "2026-11-05T14:00:00Z", rows: rows(1) },
  ]);
  expect(chart).toContain("`aaaabbbb` → `ccccdddd` → `eeeeffff`");
  expect(chart).toContain("| resnet18 / avel | █·▁ | 1.000 | 1.000 | 2.000 |");
});

test("interpolates percentiles between the middle samples", () => {
  expect(percentile([1, 2, 3, 4], 0.5)).toBe(2.5);
  expect(percentile([1, 2, 3], 0.5)).toBe(2);
  expect(percentile([1, 2, 3, 4], 0)).toBe(1);
  expect(percentile([1, 2, 3, 4], 1)).toBe(4);
  expect(percentile([5], 0.9)).toBe(5);
});
