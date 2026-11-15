#!/bin/sh
# Builds the WebGPU compiler and the CPU compiler/executor as separate Wasm modules.
set -eu

cd "$(dirname "$0")/.."

expected=$(sed -n 's/^wasm-bindgen = "=\([^"]*\)"/\1/p' Cargo.toml)
actual=$(wasm-bindgen --version | cut -d ' ' -f 2)
if [ "$actual" != "$expected" ]; then
  echo "wasm-bindgen-cli $actual does not match Cargo.toml ($expected)" >&2
  echo "Run: cargo install wasm-bindgen-cli --version $expected --locked" >&2
  exit 1
fi

cargo build --locked -p avel-wasm --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/avel_wasm.wasm \
  --target web \
  --remove-name-section \
  --remove-producers-section \
  --out-dir packages/runtime/wasm

cargo build --locked -p avel-wasm --target wasm32-unknown-unknown --profile cpu \
  --no-default-features --features cpu
wasm-bindgen target/wasm32-unknown-unknown/cpu/avel_wasm.wasm \
  --target web \
  --remove-name-section \
  --remove-producers-section \
  --out-name avel_cpu_wasm \
  --out-dir packages/runtime/wasm
