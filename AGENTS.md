# AGENTS.md

avel runs small PyTorch networks in the browser on WebGPU, with a Wasm CPU fallback.

## Rules

- Alpha: change formats and APIs freely, update every caller, and leave no compatibility shims.
- The Rust types in `crates/avel-compiler` define the graph and plan formats. `packages/runtime/wasm/` is generated: never edit it.
- `python/avel-export/src/avel_export/format.py` mirrors `graph.rs` by hand: change both, plus `tests/fixtures/identity.graph.json`.
- `avel-compiler` has no browser dependencies.
- `ExecutionPlan.run` is the hot path: no allocations, pipeline creation or wasm calls. Do the work at compile time.
- The `wasm-bindgen` pin in `Cargo.toml` matches the `wasm-bindgen-cli` version in CI.

## Checks

Run what your change touches. Rerun `bun run build:wasm` after any Rust change.

```bash
cargo fmt --all --check && cargo clippy --workspace --all-targets --all-features -- -D warnings && cargo test --workspace --all-features
bun run build && bun run format:check && bun run typecheck && bun run test
bun run test:browser  # WebGPU, needs headless Chrome
cd python/avel-export && uv run ruff check . && uv run ruff format --check . && uv run pytest
```
