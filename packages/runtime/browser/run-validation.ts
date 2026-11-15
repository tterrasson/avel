/**
 * Runs the browser validation suite (`validation.ts`) in headless Chrome and
 * prints its results. See `harness.ts` for the environment variables.
 */
import { runPage, timeout } from "./harness.js";

interface Report {
  fatal?: string;
  results?: { name: string; ok: boolean; ms: number; error?: string }[];
  logs?: string[];
}

const query = process.env.AVEL_SKIP_EXTERNAL_VIDEO === "1" ? "skipExternalVideo=1" : "";
const parsed = JSON.parse(await runPage("validation.ts", timeout(180_000), query)) as Report;
if (!parsed.fatal && !parsed.results?.length)
  throw new Error("browser reported no validation results");
for (const line of parsed.logs ?? []) {
  console.log(`  | ${line}`);
}
let failed = parsed.fatal ? 1 : 0;
for (const result of parsed.results ?? []) {
  if (result.ok) {
    console.log(`ok    ${result.ms.toFixed(0).padStart(5)}ms  ${result.name}`);
  } else {
    failed++;
    console.log(
      `FAIL  ${result.ms.toFixed(0).padStart(5)}ms  ${result.name}\n      ${result.error}`,
    );
  }
}
if (parsed.fatal) {
  console.error(`fatal: ${parsed.fatal}`);
}
const total = parsed.results?.length ?? 0;
console.log(failed === 0 ? `${total} passed` : `${failed} of ${total} failed`);
process.exit(failed === 0 ? 0 : 1);
