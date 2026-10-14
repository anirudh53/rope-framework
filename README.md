# ROPE

ROPE is a tool for forecasting upper-atmosphere density. Given a start time and a forecast window, it produces a 3-D grid of density and uncertainty estimates that can be queried at any point within the grid.

## Notices

Beta release — not recommended for production. Breaking API changes may land before the full release.

## Contact Info

Questions: [Violet Player](mailto:violet.player@noaa.gov), [Piyush Mehta](mailto:piyush.mehta@mail.wvu.edu) or [Anirudh Tapedia](mailto:anirudh.tapedia@mail.wvu.edu).
Technical support: [Anirudh Tapedia](mailto:anirudh.tapedia@mail.wvu.edu) or [Violet Player](mailto:violet.player@noaa.gov).

## Requirements

- Linux (x86_64, glibc 2.31+), macOS 12+, or Windows 10+
- Python 3.8+ (python wrapper only)

## Installation

Download the release archive for your platform and hardware from the [GitHub Releases page](https://github.com/ASSISTLabOrg/rope-framework/releases).

| Platform | File |
|----------|------|
| Linux, CPU | `rope_framework-<version>-linux-x86_64-cpu.tar.gz` |
| Linux, CUDA 12 | `rope_framework-<version>-linux-x86_64-cuda12.tar.gz` |
| macOS (Apple Silicon) | `rope_framework-<version>-macos-arm64-cpu.tar.gz` |
| Windows, CPU | `rope_framework-<version>-windows-x64-cpu.zip` |
| Windows, CUDA 12 | `rope_framework-<version>-windows-x64-cuda12.zip` |

**Currently Unsupported:** macOS (Intel), Linux (ARM64), ROCm, CUDA 13+

Extract the archive. The layout inside is:

```
rope_framework-<version>-<platform>/
    bin/        rope executable
    lib/        runtime libraries
    config/     rope.conf
    data/       models/ and drivers/ (download separately)
    include/    C API header (rope.h)
```

**macOS only:** After extracting, macOS Gatekeeper will block the bundled `.dylib` files with `library load disallowed by system policy`. Clear the quarantine attribute before use:

```
xattr -r -d com.apple.quarantine rope_framework-<version>-macos-arm64-cpu/
```

**Download models and data separately.** The `data/` directory (model artifacts and driver data) is not included in the release archive. Download it from [Google Drive](https://drive.google.com/drive/folders/1gVQ0gqzwfKDaZIuT8tWoCyWQOL_hT4Ol?usp=drive_link) and place it inside the extracted folder alongside `bin/` and `lib/`.

Once in place the directory structure should look like the layout above.

## Configuration

Default config file: `config/rope.conf`, pre-configured for the standard layout — usually nothing to change. Paths, thread counts, and other settings are documented with comments inside the file.

The one setting you may want to change is `decoder.device`:

```
[decoder]
device = cpu    # or: cuda, cuda:0, cuda:1
```

Only affects CPU builds (Linux, macOS, Windows) — the CUDA 12 builds (Linux, Windows) use ONNX Runtime's CUDA backend for the decoder and ignore this setting.

## Command Line Interface

All commands are run through the `rope` executable in `bin/`. On Linux and macOS you may need to prefix commands with `./bin/rope` if the directory is not on your PATH.

### 1. Run a forecast

```
rope forecast --start "2024-06-01 00:00:00" --horizon 24
```

Runs a 24-hour forecast starting at the given UTC time and writes the resulting grid to a cache file. A second `rope forecast` fully replaces the cached grid.

`--start` accepts `YYYY-MM-DD HH:MM:SS` or `YYYY-MM-DDTHH:MM:SS` in UTC.

### 2. Query a point

```
rope get --mode interp --time "2024-06-01T06:00:00" --lst 12.5 --lat 45.0 --alt 400.0
```

Returns density and uncertainty at the requested position, interpolated from the cached forecast.

| Argument | Description |
|----------|-------------|
| `--mode interp` | Interpolate in time between model hours |
| `--mode hold` | Snap to the next model hour (no time blending) |
| `--time` | UTC time within the forecast window |
| `--lst` | Local Solar Time, hours [0, 24) |
| `--lat` | Geodetic latitude, degrees [-87.5, 87.5] |
| `--alt` | Altitude, km [100, 980] |

Output is a JSON object on stdout:

```json
{"density": 4.72e-12, "uncertainty": 3.5e-13}
```

Density and uncertainty are in kg/m³.

### Batch queries

To query many points at once, prepare a CSV file with the columns `YYYY,MM,DD,HH,MIN,SS,lst,lat,alt_km` and pass it with `--file`:

```
rope get --mode interp --file queries.csv --output results.csv
```

### 3. Inspect the model manifest

```
rope manifest
```

Prints the model's driver columns, grid shape, and validation status. Add `--json` for machine-readable output.

### Space-weather driver data

By default `rope forecast` downloads and caches space-weather data (F10.7, Kp, Ap) from [CelesTrak](https://celestrak.org/SpaceData/) — nothing to configure. See `[driver_cache]` in `rope.conf` for the refresh interval and cache location.

To supply your own data, point `--driver <path>` (or `paths.driver_path`) at a `.swbin` or CSV file in ROPE's own format (`datetime,f10,kp,...`). A raw CelesTrak CSV (`DATE,...` columns) isn't usable as-is — convert it first:

```
rope convert-sw --input SW-Last5Years.csv --output sw.swbin
rope forecast --driver sw.swbin --start "2024-06-01 00:00:00" --horizon 24
```

`convert-sw` auto-detects raw CelesTrak vs. ROPE's own CSV format from the header — no flag needed either way.

### Add-ons: data assimilation and F10.7 forecasting

Models that ship add-on files (see `rope manifest`) can be seeded with an assimilated initial condition and/or driven
by an F10.7 forecast after the start time. Both are off by default and switched on in `rope.conf`
(`[data_assimilation] enabled`, `[f10_forecaster] enabled`) or per run:

```
rope forecast --start "2024-05-10 18:00:00" --horizon 72 --da on --f10-forecaster on
```

The output cache is unchanged: density, plus an uncertainty that then also includes the input (initial-condition
and/or F10.7) uncertainty. The JSON status gains an `addons` entry. See [docs/addons.md](docs/addons.md).

### Use a non-default config

```
rope forecast --config /path/to/rope.conf --start "2024-06-01 00:00:00" --horizon 24
```

`--config` is also accepted by `rope manifest`.

### Export as Zarr

```
rope forecast --start "2024-06-01 00:00:00" --horizon 24 --zarr /path/to/output_dir
```

`--zarr` names a container directory; the store lands at `<path>/forecast_<start>_H<horizon+1>/`. Read it with `xarray.open_dataset(path, engine="zarr")`.

## Python

A Python wrapper is included in the `python/` directory. Copy `rope.py` to your project or add `python/` to your Python path. Requires Python 3.8 or later and no additional packages.

### Setup

```python
from rope import Rope

# Paths are resolved automatically from the archive layout.
# Pass explicit paths if your layout differs.
r = Rope(
    lib_path="lib/librope.so",   # or .dylib / .dll
    exe_path="bin/rope",
    config_path="config/rope.conf",
)
```

- Constructor parameters are all optional; when omitted, `rope.py` resolves them relative to the directory one level above its own location (the archive root).
- To override the decoder device (e.g. CPU ↔ GPU) without editing `rope.conf`, pass `device="cuda"` — the wrapper writes a temporary config with that setting for the forecast subprocess.

### Example

```python
from rope import Rope, ROPE_INTERP, ROPE_HOLD

r = Rope()

# 1. Run a forecast (writes the cache file).
result = r.forecast("2024-06-01 00:00:00", horizon=24)
print(result["window_start"], "→", result["window_end"])

# 2. Query a single point.
#    The 'with' block opens an interpolation handle on entry and closes it on exit.
with r:
    pt = r.get(time="2024-06-01T07:00:00Z", lst=12.5, lat=45.0, alt_km=400.0)
    print(pt)  # {'density': 4.72e-12, 'uncertainty': 3.5e-13}

# 3. Query many points at once.
with r:
    pts = r.get_batch(
        times=["2024-06-01T07:00:00Z", "2024-06-01T08:00:00Z"],
        lsts=[12.5, 6.0],
        lats=[45.0, -30.0],
        alts_km=[400.0, 300.0],
        mode=ROPE_INTERP,
    )
    for p in pts:
        print(p["density"], p["uncertainty"])

# 4. Tight-loop use: get_density() returns a bare float with no dict allocation.
with r:
    for t in my_timestamps:
        rho = r.get_density(t, lst=0.0, lat=0.0, alt_km=400.0)

# 5. Release the handle when done (unmaps the cache file).
r.shutdown()
```

The `with` block calls `open()` on entry and `close()` on exit. `shutdown()` is an alias for `close()`, also called automatically on interpreter exit.

### Time formats

`forecast()`, `get()`, and `get_batch()` all accept query times in any of these forms:

- **Unix timestamp** (float): `1717225200.0`
- **ISO 8601 string**: `"2024-06-01T07:00:00"`, `"2024-06-01T07:00:00Z"`, `"2024-06-01 07:00:00"`
- **`datetime` object**: timezone-aware or naive (naive is treated as UTC)

### API summary

| Member | Description |
|--------|-------------|
| `Rope(lib_path, exe_path, cache_path, config_path, device)` | Constructor — all parameters optional |
| `forecast(start, horizon, drivers=None, use_da=None, use_f10_forecaster=None)` → `dict` | Run a forecast; returns `{"status", "window_start", "window_end"}` (+ `"addons"` when one is on). `drivers` supplies driver data inline for this call only, overriding `paths.driver_path`/`manifest.drivers.source`. `use_da` / `use_f10_forecaster` switch the add-ons for this call (`None` = `rope.conf`) |
| `open()` | Memory-map the cached grid and open an interpolation handle |
| `close()` | Release the handle (unmaps the cache file) |
| `refresh()` | Re-map the cache file after a new forecast |
| `get(time, lst, lat, alt_km, mode)` → `dict` | Single-point query; returns `{"density", "uncertainty"}` in kg/m³ |
| `get_density(time, lst, lat, alt_km, mode)` → `float` | Single-point density only; faster for tight loops |
| `get_batch(times, lsts, lats, alts_km, mode)` → `list[dict]` | N-point query; returns list of `{"density", "uncertainty"}` |
| `get_model_info()` → `dict` | Model manifest summary: `kind`, `latent_dim`, `grid`, `validated`, `ic`, `drivers` |
| `print_model()` | Pretty-print `get_model_info()` |
| `shutdown()` | Alias for `close()` |
| `device` | Property: active decoder device string |

**Mode constants:** `ROPE_INTERP` (1, default) — interpolate in time; `ROPE_HOLD` (0) — snap to next model hour.

Errors are raised as `RopeError(RuntimeError)` with a `.code` attribute matching the `ROPE_ERR_*` values and a message that includes the error name (e.g. `[time out of range] ...`).

## C API

For use from C or languages with a C FFI, `include/rope/capi/rope.h` exposes a stable C-compatible ABI. The shared library is `lib/librope.so` (Linux), `lib/librope.dylib` (macOS), or `bin/librope.dll` (Windows).

The C API handles interpolation only — it memory-maps the cache file written by `rope forecast`. Run forecasts through the `rope` CLI.

### Functions

```c
/* Open a handle onto the forecast-grid cache file (memory-mapped).
 * Returns NULL on failure; err_buf is filled with the reason. */
rope_interp_t* rope_open(const char* cache_path, char* err_buf, int err_len);

/* Query a single point. Returns ROPE_OK (0) on success. */
int rope_query(rope_interp_t* interp, int mode,
               double time_unix, double lst, double lat, double alt_km,
               double* density, double* uncertainty,
               char* err_buf, int err_len);

/* Query N points in one call. Fills density[] and uncertainty[]. */
int rope_query_batch(rope_interp_t* interp, int mode, int n,
                     const double* times_unix, const double* lst,
                     const double* lat, const double* alt_km,
                     double* density, double* uncertainty,
                     char* err_buf, int err_len);

/* Release the handle (unmaps the cache file). Safe to call with NULL. */
void rope_close(rope_interp_t* interp);

/* Writes a JSON summary of the model manifest into buf (kind, latent_dim, grid,
 * validated, ic, drivers) without needing an open handle or a cached forecast. */
int rope_get_manifest_info(const char* exported_dir, char* buf, int buf_len,
                           char* err_buf, int err_len);
```

**Mode constants:** `ROPE_HOLD` (0) — snap to nearest model hour; `ROPE_INTERP` (1) — interpolate in time.

**Return codes:**

| Code | Value | Meaning |
|------|-------|---------|
| `ROPE_OK` | 0 | Success |
| `ROPE_ERR_NO_FORECAST` | 2 | No forecast cached at this path — run `rope forecast` first |
| `ROPE_ERR_TIME_RANGE` | 3 | Query time outside forecast window |
| `ROPE_ERR_SPATIAL_RANGE` | 4 | Query point outside grid bounds |
| `ROPE_ERR_BAD_ARG` | 5 | Invalid argument (NULL pointer, etc.) |
| `ROPE_ERR_INTERNAL` | 6 | Unexpected internal failure |
| `ROPE_ERR_CACHE_CORRUPT` | 7 | Cache file exists but is corrupt or from an incompatible build |
| `ROPE_ERR_BUFFER_TOO_SMALL` | 8 | Caller-supplied buffer too small for the output; `err_buf` gives the required size |

### Example

```c
#include "rope/capi/rope.h"
#include <stdio.h>
#include <time.h>

int main(void) {
    char err[256];

    /* 1. Start a forecast from the CLI before calling rope_open. */

    /* 2. Open interpolation handle. */
    rope_interp_t* r = rope_open(NULL, err, sizeof(err));
    if (!r) { fprintf(stderr, "rope_open: %s\n", err); return 1; }

    /* 3. Query a single point. */
    double density, uncertainty;
    double time_unix = 1717225200.0; /* 2024-06-01T07:00:00 UTC */
    int rc = rope_query(r, ROPE_INTERP, time_unix,
                        12.5, 45.0, 400.0,
                        &density, &uncertainty, err, sizeof(err));
    if (rc != ROPE_OK) { fprintf(stderr, "rope_query: %s\n", err); }
    else printf("density=%.3e  uncertainty=%.3e kg/m3\n", density, uncertainty);

    /* 4. Release handle. */
    rope_close(r);
    return rc;
}
```

Compile and link against the shared library:

```
# Linux
gcc -Iinclude example.c -Llib -lrope -Wl,-rpath,'$ORIGIN/../lib' -o example

# macOS
clang -Iinclude example.c -Llib -lrope -rpath @loader_path/../lib -o example

# Windows (MSVC)
cl /I include example.c /link /LIBPATH:bin rope.lib /out:example.exe
```

## C# (.NET)

A C# binding is included in the `dotnet/` directory and published as the `RopeFramework` NuGet package. It targets .NET Framework 4.8 and .NET 8, and works on all supported platforms.

Interpolation queries call `librope` in-process; forecasts run via the `rope` CLI as a subprocess.

### Setup

`RopeFramework` isn't on nuget.org — install the downloaded `.nupkg` as a local package source.

1. Download `RopeFramework.<version>.nupkg` from the GitHub Releases page.
2. Install it as a local source:
   - **Visual Studio:** Tools → NuGet Package Manager → Package Sources → add the folder → install `RopeFramework` as usual.
   - **CLI** (works for both PackageReference and packages.config projects): `dotnet add package RopeFramework --source /path/to/folder`

Native binaries are bundled in the package and copied next to your build output automatically — no extra setup needed.

**Alternative:** reference the project directly, or copy `Rope.cs` into your project. Either way requires `AllowUnsafeBlocks`:

```xml
<ItemGroup>
  <ProjectReference Include="path/to/dotnet/Rope.csproj" />
</ItemGroup>
<PropertyGroup>
  <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
</PropertyGroup>
```

### Usage

```csharp
using RopeFramework;

// Paths are resolved automatically from the package layout.
// Pass explicit paths if your layout differs.
var r = new Rope(
    libPath: "lib/librope.so",   // or .dylib / .dll
    exePath: "bin/rope",
    configPath: "config/rope.conf"
);

// Run a forecast (writes the cache file).
var forecast = r.Forecast("2024-06-01 00:00:00", horizon: 24);
Console.WriteLine($"Window: {forecast.WindowStart} → {forecast.WindowEnd}");

// Open an interpolation handle and query.
r.Open();
var result = r.Get(
    time: new DateTime(2024, 6, 1, 7, 0, 0, DateTimeKind.Utc),
    lst: 12.5, lat: 45.0, altKm: 400.0,
    mode: Rope.Interp
);
Console.WriteLine($"density={result.Density:e3}  uncertainty={result.Uncertainty:e3} kg/m³");

// Batch query.
var times = new[] {
    new DateTime(2024, 6, 1, 7, 0, 0, DateTimeKind.Utc),
    new DateTime(2024, 6, 1, 8, 0, 0, DateTimeKind.Utc),
};
var results = r.GetBatch(times,
    lsts:   [12.5, 6.0],
    lats:   [45.0, -30.0],
    altsKm: [400.0, 300.0]);

// Dispose closes the interpolation handle (unmaps the cache file).
r.Dispose();
```

### API summary

| Member | Description |
|--------|-------------|
| `Rope(libPath, exePath, cachePath, configPath)` | Constructor — all parameters optional; defaults derived from package layout |
| `Forecast(string start, int horizon)` | Run a forecast; returns `ForecastResult` with `WindowStart` / `WindowEnd` |
| `Forecast(DateTime start, int horizon)` | DateTime overload |
| `Forecast(..., IDictionary<string,object[]> drivers)` | Overload (string or DateTime start) that supplies driver data inline for this call only, overriding `paths.driver_path`/`manifest.drivers.source` |
| `Forecast(..., drivers, bool? useDa, bool? useF10Forecaster)` | Overload that also switches the add-ons for this call (`null` = `rope.conf`; `drivers` may be `null`) |
| `Open()` | Memory-map the cached grid and open an interpolation handle |
| `Close()` | Release the handle (unmaps the cache file) |
| `Refresh()` | Re-map the cache file after a new forecast |
| `Get(double timeUnix, ...)` | Single-point query; returns `QueryResult` with `Density` and `Uncertainty` |
| `Get(DateTime time, ...)` | DateTime overload |
| `GetBatch(double[] timesUnix, ...)` | Batch query; returns `QueryResult[]` |
| `GetBatch(DateTime[] times, ...)` | DateTime overload |
| `GetModelInfo()` → `Dictionary<string, object>` | Model manifest summary: kind, latent_dim, grid, validated, ic, drivers |
| `PrintModel()` | Pretty-print `GetModelInfo()` |
| `Shutdown()` | Alias for `Close()` |
| `Dispose()` | Close the handle and unload the library |

`Rope` implements `IDisposable`. Use a `using` block or call `Dispose()` explicitly. `Rope.Interp` (1) and `Rope.Hold` (0) are the mode constants.

Errors are reported as `RopeException` with a `Code` property matching the `ROPE_ERR_*` constants above and a message that includes the error name.

## Demo

The `demo/` directory contains a Jupyter notebook that benchmarks ROPE against NRLMSIS-2.1 atmospheric drag across two scenarios: forecast and interpolation throughput (10,000 random point queries), and a 1-day ISS-like orbit integration comparing altitude decay between the two models.

Running the demo scripts is **not** required to use any of the rope library, but may be beneficial as a sanity check to ensure your system is functioning.

To set it up, install the demo dependencies from the archive root:

```
bash demo/install.sh
```

This requires Python 3.8+ and `pip`. It installs the Python wrapper from `python/` and the demo support library from `demo/`. The notebook fetches space weather data from CelesTrak on first run and caches it locally, so an internet connection is needed the first time.

Then open the notebook:

```
bash demo/run_benchmark.sh
```

The notebook uses `demo/rope.conf`, which is pre-configured to point at the `data/` directory in the standard archive layout. If you extracted the archive somewhere else, edit the paths in that file before running.

## License

Mozilla Public License 2.0. See `LICENSE`.
