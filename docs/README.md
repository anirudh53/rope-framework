# Developer Documentation

Reference for contributors and integrators working with the ROPE source code directly.

| Document | Contents |
|----------|----------|
| [architecture.md](architecture.md) | System overview, module map, data flow, design principles |
| [building.md](building.md) | CMake options, dependency management, test categories, packaging |
| [pipeline.md](pipeline.md) | The full inference pipeline step by step: driver loading, sequence building, base model rollout, meta fusion, Unscented Transform, latent decoding |
| [driver-system.md](driver-system.md) | SpaceWeatherDB, DriverCacheManager, binary formats, IC table, preprocessing script |
| [interpolation.md](interpolation.md) | ForecastGrid structure, coordinate handling, trilinear spatial interpolation in log₁₀ space, temporal blending |
| [forecast-cache-format.md](forecast-cache-format.md) | The on-disk forecast-grid cache file: layout, atomic writes, memory-mapped reads |
| [model-artifacts.md](model-artifacts.md) | ONNX models, stats files, manifest-driven driver/IC config, version compatibility |
| [addons.md](addons.md) | Optional add-ons: data-assimilation initial conditions, F10.7 forecaster, input uncertainty, GPU decoder |
| [io.md](io.md) | CsvReader, ConfigReader, Stats/FeatureNormalizer, binary format details, adding new formats |
| [adding-a-pipeline.md](adding-a-pipeline.md) | How to add a new pipeline kind: registry, manifest spec, checklist |
| [version-control.md](version-control.md) | Branch model: main/develop/feature/bugfix conventions and release tagging |
| [github-actions.md](github-actions.md) | CI workflows: build matrix per branch type, release publish, cache cleanup |

## Quick orientation

If you are new to the codebase, read in this order:

1. **architecture.md** — understand the system before reading code
2. **pipeline.md** — the core algorithm; everything else supports it
3. **driver-system.md** or **interpolation.md** — whichever area you're working in
4. **building.md** — get a build running and tests passing

## Source layout

```
src/
  cli/           entry point, argument parsing, forecast/get commands
  forecast/      inference pipeline (pipeline.cpp + internal headers)
  interpolate/   GridInterpolator<Grid>
  io/            all file I/O, incl. the forecast-grid cache file
  capi/          C shared-library wrapper (memory-maps the cache file)
  core/
    platform/    OS-specific: exe_path, cache_dir, memory-mapped files

include/rope/    public headers mirroring src/ layout
```

## Rules to keep in mind

- No OS-specific `#ifdef` outside `src/core/platform/`. If you need platform-conditional code elsewhere, add a function to `platform.h` and implement it in both platform files.
- Fail loudly. No silent fallbacks, no clamping, no fabricated values.
- Uncertainty is not optional. Every forecast result carries both density and uncertainty.
- Model artifacts are version-locked. Do not upgrade ORT or LibTorch without regression testing all supported hardware variants.
