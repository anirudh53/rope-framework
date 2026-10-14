# Model Artifacts

All runtime artifacts live in a single flat directory (`exported_dir`, configured via `paths.exported_dir` in `rope.conf`). They are produced by an external training pipeline and loaded when `rope forecast` runs. This service never modifies them.

---

## Directory layout

```
<exported_dir>/
    model_manifest.json                           # self-describing manifest; drives everything below
    base_model_00.onnx  …  base_model_14.onnx     # 15 base temporal models
    meta_model.onnx                               # ensemble fusion
    coae_decoder.onnx                             # ONNX Runtime decoder (default)
    coae_decoder.pt                               # LibTorch TorchScript decoder (optional)
    coae_decoder.onnx.data                        # external weights (ORT large-model support)
    stats_ts.bin                                  # feature normalizer statistics
    stats_cae.bin                                 # CAE denormalizer statistics
    meta_model_out_shape.bin                      # meta model output shape metadata
    ic_table.icbin                                # IC lookup table (binary)
```

See the project's agent-facing knowledge base (`model-registry.md`) for the full `model_manifest.json` schema — driver columns/source, IC config, and grid shape are all declared there, not in separate files.

---

## Base models — `base_model_00.onnx` … `base_model_14.onnx`

**Architecture split:**
- `00`–`04`: LSTM
- `05`–`09`: GRU
- `10`–`14`: Transformer

**I/O contract:**
- Input: `(1, S=3, D)` — one S-step normalized feature window
- Output: `(1, K=10)` — one normalized latent prediction

The pipeline calls each base model H times (the rollout loop), not once with the full horizon. This is the auto-regressive structure: each prediction is fed back as the latent component of the next input window.

Transformers use 2 inter-op ORT threads; LSTM/GRU use 1. This reflects empirical throughput characteristics — Transformers benefit from internal parallelism while LSTM/GRU are memory-bandwidth bound.

---

## Meta model — `meta_model.onnx`

Learned ensemble fusion. Takes the driver context and all 15 base predictions; produces a fused latent mean and per-timestep spread.

**Input (packed flat):**
```
concat(
    x_chunk[0..H-1],          # (H, S, D) driver context
    base_latents_norm          # (M=15, H, K) base predictions
)
```

**Output:** `[mean (H×K) | std (H×K)]` — the first `H*K` floats are the mean latent trajectory; the remainder are per-timestep standard deviations (used to construct the ensemble covariance for the UT).

Two output shapes are supported, auto-detected from `meta_model_out_shape.bin` at runtime:
- `{T, K, 2}` — per-dimension fusion weights (richer)
- `{T, 2*K}` — flat concat of mean and std

---

## COAE decoder — `coae_decoder.onnx` / `coae_decoder.pt`

Decodes latent vectors to physical density grids.

**Input:** `(batch, K=10)` — latent vectors already denormalized to physical space
**Output:** `(batch, 1, n_lst, n_lat, n_alt)` — log₁₀-normalized density, where `n_lst`/`n_lat`/`n_alt` are this model's `grid` shape declared in `model_manifest.json` (72×36×45 for the reference model below, but per-model in general)

**Denormalization:**
```
density_phys = 10 ^ (output_norm × σ_cae + μ_cae)
```

where `(μ_cae, σ_cae)` come from `stats_cae.bin`. This step is applied by `CoaeDenormalizer::apply_inplace()` (`rope/io/stats.h`) in the `LatentDecoder`.

The decoder is the most compute-intensive step for long horizons (or when UT is enabled, where it processes `H×21` inputs). It is called with batches of up to `decode_batch_size` latent vectors (manifest default 120; capped lower via `rope.conf`'s `forecast.decode_batch_size` — see [pipeline.md](pipeline.md)). Manifest-side, `decode_batch_size` lives at `decoder.params.decode_batch_size`, alongside the per-stage `decoder.params.stages` array — both top-level, kind-agnostic fields (see `model-registry.md`'s "COAE decoder" section for the full manifest shape and the `IDecoder`/`make_decoder()`/`CoaeDecoder` abstraction).

**Backend selection:** ONNX Runtime is the default. If built with `ROPE_USE_LIBTORCH=ON`, the `.pt` TorchScript file is used instead and respects `decoder.device = cuda`.

---

## `stats_ts.bin` — feature normalizer

Z-score statistics for the full time-series feature vector `[latent | driver]`.

**Binary format:**
```
uint32  ndim
uint32  shape[ndim]     // typically shape = [D]
float32 mu[N]           // N = product(shape)
float32 sigma[N]
```

`D = K + driver_dim`, inferred at load time. Used by `FeatureNormalizer`:
- `norm_full_inplace(float* x)` — normalizes the full `(K + driver_dim)` vector
- `norm_driver_inplace(float* drv)` — normalizes only the driver portion (during rollout)
- `denorm_latents_inplace(float* z)` — recovers physical latents from normalized latents

The shape of `stats_ts.bin` implicitly determines `D`, which determines `driver_dim = D - K`.

---

## `stats_cae.bin` — CAE denormalizer

Z-score statistics for the decoder output.

**Shape variants:**
- `(1,)` — single scalar mean/sigma applied uniformly to all voxels
- `(1, n_lst, n_lat, n_alt)` — per-voxel statistics (spatially varying normalization), matching this model's `grid` shape

The shape is read at load time and the correct denormalization logic is selected automatically.

---

## Driver columns and IC config

Both declared as top-level, required blocks in `model_manifest.json` (`drivers`, `ic`) — no separate config files:

- `drivers.columns` (array of `{name, description}`) + `drivers.source` (string) drive `fill_driver()` in the pipeline, which now resolves each column generically via `DriverRow::get(name)` rather than a hardcoded per-name dispatch. Adding a new driver feature (e.g. `ap`, `dst`) really is a training-time decision now — no code changes required, as long as the raw data source (CSV or `.swbin`) actually supplies that column. `drivers.source` is the key passed to `DriverCacheManager::get_path()` when no explicit `driver_path` is set.
- `ic.params.grid_axes` (exactly 2 axis names) and the top-level `latent_dim` validate that the loaded IC table matches the model's expected axes and `K` — the pipeline cross-checks the table's own auto-detected/stored axis names against `grid_axes` and throws on mismatch.

See `model-registry.md` for the full schema.

---

## `ic_table.icbin`

Binary IC lookup table (see [driver-system.md](driver-system.md) for the v2 format). Maps a 2-axis grid (commonly F10, Kp, but declared per model via `ic.params.grid_axes`) to `K`-dimensional latent initial conditions via bilinear interpolation.

Converted from `IC_Table_modified.csv` via `rope convert-ic`.

---

## Add-on artifacts (optional)

`addons/*.dabin` (data-assimilation initial conditions) and `addons/*.f10bin` (F10.7 forecaster
ensembles), declared under the manifest's optional `addons` block. Off unless enabled in rope.conf;
see [addons.md](addons.md).

## Version compatibility

The `forecast-invariants.md` rule applies: model artifacts are version-locked to the runtime they were exported with. ORT and LibTorch versions are pinned in `cmake/Dependencies.cmake`. Loading a model exported against a different ORT version is a hard failure. Never silently downgrade or substitute — a mismatch that appears to work may produce wrong results.
