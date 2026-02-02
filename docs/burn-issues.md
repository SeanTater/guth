# Burn Backend Issues & Workarounds

This document tracks backend issues we’ve encountered in Burn (and related crates) and the
workarounds we’ve adopted in this repo. The goal is to keep our changes minimal and upstreamable.

## Known Issues

### burn-ndarray `is_nan` does not detect NaNs reliably
- **Impact:** NaN checks can miss invalid values during inference.
- **Workaround:** We avoid relying on `is_nan()` and use explicit checks in `flow_lm`.
- **Code reference:** `src/model/flow_lm.rs`

### burn-ndarray zero-length dimension divide-by-zero
- **Impact:** Some zero-length tensor shapes can trigger a divide-by-zero in `burn-ndarray` 0.20.1.
- **Workaround:** Avoid zero-length dims in hot paths; we use helper functions and guard logic.
- **Code reference:** `src/modules/linear.rs`, `src/model/tts.rs`

### burn-wgpu + Vulkan (SPIR-V) autotune stack overflow
- **Impact:** Enabling SPIR-V with the default `burn-wgpu` features crashes with a stack overflow
  during matmul autotune on Intel Iris Xe (Mesa 25.2.3).
- **Workaround:** Disable `burn-wgpu` default features and enable only `std`, `fusion`, and
  `vulkan` in `Cargo.toml`, which disables autotune.
- **Code reference:** `Cargo.toml` (burn-wgpu dependency configuration)

## Performance Notes

### WGPU backend faster than ndarray in release builds (Intel Iris Xe)
- **Observation:** With SPIR-V (Vulkan) enabled and autotune disabled, WGPU is significantly faster
  than `burn-ndarray` in release builds on the Intel Iris Xe iGPU.
- **Status:** Using WGPU as the preferred performance path; autotune crash remains an issue.

### OpenBLAS did not improve performance in our tests
- **Observation:** Both `openblas-src` and system OpenBLAS were slower than the default ndarray
  backend on our workload.
- **Status:** Not enabled; we default to static-friendly builds and focus on WGPU.
