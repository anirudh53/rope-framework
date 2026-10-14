# Add-ons: data-assimilation initial conditions and the F10.7 forecaster

Two optional inputs for `stacked_ensemble` models. Both are **off by default**; with both off the
forecast is byte-identical to a build without add-on support. Switch them on in `rope.conf` or
per run (`rope forecast --da on --f10-forecaster on`, `Rope.forecast(..., use_da=True,
use_f10_forecaster=True)`).

| Add-on | What changes | Input uncertainty |
|---|---|---|
| `[data_assimilation]` | The rollout is seeded with the assimilated latent state at the start time, not the IC-table lookup. Warm-up is skipped. | 10 dims: the mean of the 3 seed states, with the filter covariance |
| `[f10_forecaster]` | F10.7 after the start time comes from a forecaster ensemble (its mean); observed drivers up to the start are kept. `f10_41day_avg` is recomputed from the new series. | 1 dim: one offset on the forecast F10.7, σ = ensemble spread averaged over the forecast days |

With `compute_uncertainty = true`, the active input dimensions (10, 1 or 11) go through one unscented
transform (2n+1 sigma points; α=1, β=2, κ=0). Each sigma point is a full deterministic forecast.
The ensemble (model) unscented transform runs once at the central point, and the cached
uncertainty is

    uncertainty = sqrt(input_variance + model_variance)

The cached density is the input-UT mean. The cache-file format does not change.

## Configuration

```ini
[data_assimilation]
enabled = false
# states_path = ...        # .dabin override (default: the model's addons.data_assimilation.file)
ic_cov_mode = hour_avg     # hour_avg | mean | last  (3-hour joint covariance -> 10x10)
ic_cov_scale = 1.0         # multiplies the DA standard deviations

[f10_forecaster]
enabled = false
# ensemble = LSTM_dynamic  # a key of the manifest's addons.f10_forecaster.ensembles
# ensemble_path = ...      # .f10bin override
sigma = 0                  # fixed F10.7 sigma (sfu); 0 = ensemble spread
```

`ic_cov_mode`:
- `hour_avg`: the mean of the three per-hour 10×10 blocks.
- `mean`: the covariance of the mean state, A Σ Aᵀ with A = [I I I]/3.
- `last`: the start-hour block.

A sigma point shifts all three seed rows by the same vector, so the central point is the
assimilated state itself.

`rope forecast` adds an `addons` object to its JSON status: the DA state time and its observation
count, the daily F10.7 forecast (mean and std), σ_F10, and the sigma-point counts.

## Model artifacts

Declared in the model's `model_manifest.json`. Builds without add-on support ignore the block.

```jsonc
"addons": {
  "data_assimilation": { "kind": "da_state_table", "file": "addons/da_states_20240229_20240801.dabin",
                         "coverage": ["2024-02-29T04:00:00", "2024-08-01T00:00:00"] },
  "f10_forecaster": { "default": "LSTM_dynamic",
                      "ensembles": { "LSTM_dynamic": { "kind": "f10_ensemble", "file": "addons/f10_LSTM_dynamic.f10bin" },
                                     "MLP_dynamic":  { "kind": "f10_ensemble", "file": "addons/f10_MLP_dynamic.f10bin" } } }
}
```

### `.dabin`: assimilated initial conditions

Each record holds one hour T of an assimilation run: the seq_len hourly latent states T−S+1 … T,
and their joint covariance, **in the model's own latent space**. The model's export tooling fits
latents to this model's decoder and carries the filter covariance across with the decoder
Jacobians. Format: `include/rope/io/da_states.h`. A start hour outside the table fails with the
table's coverage in the message.

### `.f10bin`: F10.7 forecaster ensemble

This is the dynamic-lookback ensemble of Daniell & Mehta (2023), *Probabilistic Solar Proxy
Forecasting with Neural Network Ensembles* ([arXiv:2306.02169](https://arxiv.org/abs/2306.02169)):
180 members, lookbacks of 7–22 days, LSTM or MLP. Members are stored as plain layer weights and
activations. The framework runs them itself (`src/forecast/addons/f10_ensemble.cpp`, double
precision), so no extra runtime is needed.

Each member works as follows:
- Its input is the last `lookback` days of [F10.7, 12-day trailing mean], normalized with one
  scalar mean and std.
- It predicts the next day.
- The prediction and an extrapolated mean are appended, and the window slides by one day.

The daily history comes from the driver data: the F10.7 value at 00 UT of each preceding day. The
hourly F10.7 after the start is linear between daily 00 UT anchors. The golden test
(`tests/cpp/fixtures/addons/make_f10_fixture.py`) checks the forward pass against the reference
Keras implementation.

## GPU decoder (CUDA builds)

CUDA builds (`-DROPE_HARDWARE=cuda12`) run the ONNX decoder on the GPU through ONNX Runtime's CUDA
execution provider when `decoder.device = cuda` or `cuda:N`, with TF32 disabled. If the provider
can't be created (no GPU, or missing CUDA 12 / cuDNN 9 runtime libraries), the decoder stays on the
CPU and a warning is printed.

To pick a GPU, prefer `CUDA_VISIBLE_DEVICES=N` with `device = cuda:0`: ONNX Runtime can still touch
GPU 0 when asked for `cuda:N`. Example timing, 48-hour forecast, RTX A6000 vs 32-core CPU: about
4.5 s vs about 75 s, with the same result to float32 precision.
