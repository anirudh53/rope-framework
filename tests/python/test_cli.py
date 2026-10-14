"""
Category C — end-to-end CLI integration tests.

Exercises the full rope pipeline: forecast (writes the cache file), single-
point query, batch query, and error paths -- all via the rope CLI binary.
No server process, no sockets: `rope forecast` runs inference directly and
writes a cache file; `rope get` reads it (memory-mapped).

Required env vars (injected by CTest; fall back to sensible defaults for
manual runs from the build directory):
  ROPE_EXE          path to the rope binary
  ROPE_FIXTURE_DIR  path to the tests/fixtures directory
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Derive sensible defaults so `pytest tests/python/` works from the project
# root without needing env vars set (CTest sets them explicitly in CI).
_project_root = Path(__file__).parent.parent.parent
_default_exe  = _project_root / "build" / (
    "rope.exe" if sys.platform == "win32" else "rope"
)
_lib_ext      = {"win32": "dll", "darwin": "dylib"}.get(sys.platform, "so")
_default_lib  = _project_root / "build" / f"librope.{_lib_ext}"

ROPE_EXE    = os.environ.get("ROPE_EXE", str(_default_exe))
ROPE_LIB    = os.environ.get("ROPE_LIB", str(_default_lib))
FIXTURE_DIR = Path(os.environ.get("ROPE_FIXTURE_DIR",
                                   _project_root / "tests" / "fixtures"))

# Must match the sw_test.swbin coverage window used by the pipeline tests.
# sw_test.csv covers 2023-12-31T22:00:00 through 2024-01-01T03:00:00 (6 rows);
# horizon=3 with seq_len=3 requires (seq_len-1)+(horizon+1) = 6 rows.
FORECAST_START   = "2024-01-01 00:00:00"
FORECAST_HORIZON = 3
QUERY_TIME       = "2024-01-01T01:00:00"


def _rope(*args, timeout=60):
    return subprocess.run(
        [ROPE_EXE, *args],
        capture_output=True, text=True, timeout=timeout,
    )


def _write_conf(path: Path, exported_dir=None) -> None:
    # driver_path: explicit binary fixture — bypasses cache manager entirely.
    # ic_csv is no longer set: the IC table is auto-discovered from exported_dir
    # (test_models/ic_table.icbin, generated alongside the other fixtures).
    exported_dir = exported_dir or (FIXTURE_DIR / "test_models")
    path.write_text(
        f"[paths]\n"
        f"exported_dir = {exported_dir}\n"
        f"driver_path  = {FIXTURE_DIR / 'test_models' / 'sw_test.swbin'}\n"
    )


# ---------------------------------------------------------------------------
# Module-scoped forecast-cache fixture — one cache file shared across most
# tests. No teardown needed: there is no process to stop, just a file.
# ---------------------------------------------------------------------------

class _ForecastCache:
    def __init__(self, cache_path: str, conf: str):
        self.cache_path = cache_path
        self.conf = conf

    def run(self, *args, timeout=60):
        return _rope("--cache-path", self.cache_path, *args, timeout=timeout)


@pytest.fixture(scope="module")
def forecast_cache(tmp_path_factory):
    tmp   = tmp_path_factory.mktemp("rope_cli")
    cache = str(tmp / "forecast_grid.bin")
    conf  = tmp / "rope.conf"
    _write_conf(conf)

    result = _rope(
        "--cache-path", cache, "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", str(conf),
    )
    assert result.returncode == 0, f"forecast failed:\n{result.stderr}"

    yield _ForecastCache(cache, str(conf))


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------

def test_forecast_returns_ok(forecast_cache):
    result = forecast_cache.run(
        "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", forecast_cache.conf,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["status"] == "ok"
    assert "window_start" in data
    assert "window_end"   in data


def test_forecast_driver_override_accepts_shuffled_columns(tmp_path):
    """--driver points at an explicit CSV with columns in a different order
    than the model's manifest declares -- column lookup is by name, not
    position, so this must still succeed.
    """
    conf = tmp_path / "rope.conf"
    _write_conf(conf)
    cache = str(tmp_path / "forecast_grid.bin")

    driver_csv = tmp_path / "shuffled_drivers.csv"
    driver_csv.write_text(
        "kp,datetime,f10\n"
        "2.0,2023-12-31T22:00:00,150.0\n"
        "2.0,2023-12-31T23:00:00,150.0\n"
        "2.0,2024-01-01T00:00:00,150.0\n"
        "2.0,2024-01-01T01:00:00,150.0\n"
        "2.0,2024-01-01T02:00:00,150.0\n"
        "2.0,2024-01-01T03:00:00,150.0\n"
    )

    result = _rope(
        "--cache-path", cache, "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", str(conf), "--driver", str(driver_csv),
    )
    assert result.returncode == 0, f"forecast with --driver failed:\n{result.stderr}"
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["status"] == "ok"


def test_forecast_warmup_runs_wider_but_writes_and_queries_only_the_requested_window(tmp_path):
    """warmup_time_hours widens the run window, but only [start, start+horizon] is ever written or queryable."""
    conf = tmp_path / "rope.conf"
    conf.write_text(
        f"[forecast]\n"
        f"warmup_time_hours = 24\n"
        f"[paths]\n"
        f"exported_dir = {FIXTURE_DIR / 'test_models'}\n"
        f"driver_path  = {FIXTURE_DIR / 'sw_test_long.csv'}\n"
    )
    cache = str(tmp_path / "forecast_grid.bin")

    result = _rope(
        "--cache-path", cache, "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", str(conf),
    )
    assert result.returncode == 0, f"forecast with warmup failed:\n{result.stderr}"
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["status"] == "ok"
    assert data["window_start"] == "2024-01-01T00:00:00"
    assert data["window_end"]   == "2024-01-01T03:00:00"

    in_range = _rope(
        "--cache-path", cache, "get", "--mode", "hold",
        "--time", FORECAST_START.replace(" ", "T"),
        "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert in_range.returncode == 0, f"query at start_time should succeed:\n{in_range.stderr}"

    before_start = _rope(
        "--cache-path", cache, "get", "--mode", "hold",
        "--time", "2023-12-31T23:00:00",
        "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert before_start.returncode != 0, "warmup hours must not be queryable"


def test_rope_binding_forecast_uses_custom_cache_path(tmp_path):
    """Regression test: Rope.forecast() must pass --cache-path when a custom
    cache_path is configured -- otherwise it silently falls back to the
    default per-user cache file, which could read/write state unrelated to
    this Rope instance.
    """
    sys.path.insert(0, str(_project_root / "python"))
    from rope import Rope

    lib_path = Path(ROPE_LIB)
    assert lib_path.is_file(), f"{lib_path} not built"

    conf = tmp_path / "rope_binding.conf"
    _write_conf(conf)
    custom_cache = str(tmp_path / "custom_forecast_grid.bin")

    r = Rope(lib_path=lib_path, exe_path=Path(ROPE_EXE), cache_path=custom_cache, config_path=conf)
    result = r.forecast(FORECAST_START, FORECAST_HORIZON)
    assert result["status"] == "ok"

    # The cache file must actually have been written at the custom path (not
    # the default per-user one) -- reading it directly via the CLI proves it.
    probe = _rope(
        "--cache-path", custom_cache, "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert probe.returncode == 0


def test_rope_binding_forecast_with_explicit_drivers(tmp_path):
    """Rope.forecast(drivers=...) writes a temp CSV and passes --driver,
    overriding paths.driver_path for this call only (all-or-nothing: the
    dict must cover the full history+horizon window itself)."""
    sys.path.insert(0, str(_project_root / "python"))
    from rope import Rope

    conf = tmp_path / "rope_binding.conf"
    _write_conf(conf)
    cache = str(tmp_path / "explicit_drivers_forecast_grid.bin")

    r = Rope(lib_path=Path(ROPE_LIB), exe_path=Path(ROPE_EXE), cache_path=cache, config_path=conf)
    result = r.forecast(
        FORECAST_START, FORECAST_HORIZON,
        drivers={
            "datetime": [
                "2023-12-31T22:00:00", "2023-12-31T23:00:00",
                "2024-01-01T00:00:00", "2024-01-01T01:00:00",
                "2024-01-01T02:00:00", "2024-01-01T03:00:00",
            ],
            "kp":  [2.0] * 6,
            "f10": [150.0] * 6,
        },
    )
    assert result["status"] == "ok"


def test_rope_binding_forecast_with_explicit_drivers_f10_41day_avg(tmp_path):
    """drivers=... may supply 'f10_41day_avg' directly like any other raw
    column -- it is no longer a reserved, always-derived name."""
    sys.path.insert(0, str(_project_root / "python"))
    from rope import Rope

    conf = tmp_path / "rope_binding.conf"
    _write_conf(conf)
    cache = str(tmp_path / "f10_41day_avg_forecast_grid.bin")

    r = Rope(lib_path=Path(ROPE_LIB), exe_path=Path(ROPE_EXE), cache_path=cache, config_path=conf)
    result = r.forecast(
        FORECAST_START, FORECAST_HORIZON,
        drivers={
            "datetime": [
                "2023-12-31T22:00:00", "2023-12-31T23:00:00",
                "2024-01-01T00:00:00", "2024-01-01T01:00:00",
                "2024-01-01T02:00:00", "2024-01-01T03:00:00",
            ],
            "kp":            [2.0] * 6,
            "f10":           [150.0] * 6,
            "f10_41day_avg": [148.5] * 6,
        },
    )
    assert result["status"] == "ok"


def test_rope_binding_get_model_info(tmp_path):
    sys.path.insert(0, str(_project_root / "python"))
    from rope import Rope

    conf = tmp_path / "rope_binding.conf"
    _write_conf(conf)

    r = Rope(lib_path=Path(ROPE_LIB), exe_path=Path(ROPE_EXE), config_path=conf)
    info = r.get_model_info()
    assert info["kind"] == "stacked_ensemble"
    assert info["drivers"]["source"] == "celestrak_sw"
    names = [c["name"] for c in info["drivers"]["columns"]]
    assert names == ["f10", "kp", "t1", "t2", "t3", "t4"]
    assert info["ic"]["axes"] == ["f10", "kp"]


def test_rope_binding_print_model_smoke(tmp_path, capsys):
    sys.path.insert(0, str(_project_root / "python"))
    from rope import Rope

    conf = tmp_path / "rope_binding.conf"
    _write_conf(conf)

    r = Rope(lib_path=Path(ROPE_LIB), exe_path=Path(ROPE_EXE), config_path=conf)
    r.print_model()
    out = capsys.readouterr().out
    assert "Model kind:" in out
    assert "f10" in out


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def test_manifest_prints_driver_columns(forecast_cache):
    result = _rope("manifest", "--config", forecast_cache.conf)
    assert result.returncode == 0
    assert "Driver columns" in result.stdout
    assert "f10" in result.stdout
    assert "kp" in result.stdout


def test_manifest_json_is_well_formed(forecast_cache):
    result = _rope("manifest", "--config", forecast_cache.conf, "--json")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["kind"] == "stacked_ensemble"
    assert data["drivers"]["source"] == "celestrak_sw"
    names = [c["name"] for c in data["drivers"]["columns"]]
    assert names == ["f10", "kp", "t1", "t2", "t3", "t4"]
    assert data["ic"]["axes"] == ["f10", "kp"]


# ---------------------------------------------------------------------------
# get — single point
# ---------------------------------------------------------------------------

def test_get_interp_returns_valid_point(forecast_cache):
    result = forecast_cache.run(
        "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["density"]     > 0
    assert data["uncertainty"] >= 0


def test_get_hold_returns_valid_point(forecast_cache):
    result = forecast_cache.run(
        "get", "--mode", "hold",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["density"]     > 0
    assert data["uncertainty"] >= 0


def test_get_time_out_of_range_fails(forecast_cache):
    result = forecast_cache.run(
        "get", "--mode", "interp",
        "--time", "2030-01-01T00:00:00",
        "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
        timeout=10,
    )
    assert result.returncode != 0


def test_get_without_forecast_fails_clearly(tmp_path):
    missing_cache = str(tmp_path / "never_written.bin")
    result = _rope(
        "--cache-path", missing_cache, "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
        timeout=10,
    )
    assert result.returncode != 0
    assert "forecast" in result.stderr.lower()


# ---------------------------------------------------------------------------
# get — batch
# ---------------------------------------------------------------------------

def test_batch_get_from_csv(forecast_cache, tmp_path):
    csv = tmp_path / "queries.csv"
    csv.write_text(
        "YYYY,MM,DD,HH,MIN,SS,lst,lat,alt_km\n"
        "2024,01,01,01,00,00,12.0,45.0,400.0\n"
        "2024,01,01,01,30,00,6.0,-30.0,300.0\n"
    )
    out = tmp_path / "results.json"
    result = forecast_cache.run(
        "get", "--mode", "interp",
        "--file", str(csv), "--output", str(out),
        timeout=15,
    )
    assert result.returncode == 0
    rows = json.loads(out.read_text())
    assert len(rows) == 2
    for row in rows:
        assert row["density"]     > 0
        assert row["uncertainty"] >= 0


# ---------------------------------------------------------------------------
# get — altitude extrapolation
# ---------------------------------------------------------------------------

def test_get_altitude_extrapolation_default_on_succeeds(forecast_cache):
    result = forecast_cache.run(
        "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "1200.0",
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["density"]     > 0
    assert data["uncertainty"] >= 0


def test_get_altitude_extrapolation_disabled_fails(forecast_cache):
    result = forecast_cache.run(
        "get", "--mode", "interp", "--no-extrapolate-altitude",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "1200.0",
        timeout=10,
    )
    assert result.returncode != 0


def test_get_altitude_extrapolation_hard_ceiling_fails(forecast_cache):
    result = forecast_cache.run(
        "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "2500.0",
        timeout=10,
    )
    assert result.returncode != 0


def test_get_altitude_n_etp_pts_flag_accepted(forecast_cache):
    result = forecast_cache.run(
        "get", "--mode", "interp", "--n-etp-pts", "4",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "1200.0",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["density"] > 0


def test_get_n_etp_pts_out_of_range_fails(forecast_cache):
    result = forecast_cache.run(
        "get", "--mode", "interp", "--n-etp-pts", "1000",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "1200.0",
        timeout=10,
    )
    assert result.returncode != 0


def test_get_config_interpolation_section_disables_extrapolation(forecast_cache, tmp_path):
    conf = tmp_path / "rope_no_extrap.conf"
    _write_conf(conf)
    with open(conf, "a") as f:
        f.write("\n[interpolation]\nextrapolate_altitude = false\n")

    result = forecast_cache.run(
        "get", "--mode", "interp", "--config", str(conf),
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "1200.0",
        timeout=10,
    )
    assert result.returncode != 0


def test_get_cli_flag_overrides_config(forecast_cache, tmp_path):
    # Config leaves extrapolation on (the default); --no-extrapolate-altitude must still win.
    conf = tmp_path / "rope_extrap_on.conf"
    _write_conf(conf)
    with open(conf, "a") as f:
        f.write("\n[interpolation]\nextrapolate_altitude = true\n")

    result = forecast_cache.run(
        "get", "--mode", "interp", "--config", str(conf), "--no-extrapolate-altitude",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "1200.0",
        timeout=10,
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# get — altitude extrapolation (via Python binding)
# ---------------------------------------------------------------------------

def test_rope_binding_set_extrapolation(forecast_cache):
    sys.path.insert(0, str(_project_root / "python"))
    from rope import Rope, RopeError

    r = Rope(lib_path=Path(ROPE_LIB), cache_path=forecast_cache.cache_path)
    with r:
        result = r.get(time=QUERY_TIME, lst=12.0, lat=45.0, alt_km=1200.0)
        assert result["density"] > 0

        r.set_extrapolation(extrapolate_altitude=False)
        with pytest.raises(RopeError):
            r.get(time=QUERY_TIME, lst=12.0, lat=45.0, alt_km=1200.0)

        r.set_extrapolation(extrapolate_altitude=True, n_etp_pts=4)
        result = r.get(time=QUERY_TIME, lst=12.0, lat=45.0, alt_km=1200.0)
        assert result["density"] > 0


def test_rope_binding_constructor_extrapolation_params(forecast_cache):
    sys.path.insert(0, str(_project_root / "python"))
    from rope import Rope, RopeError

    r = Rope(lib_path=Path(ROPE_LIB), cache_path=forecast_cache.cache_path,
             extrapolate_altitude=False)
    with r:
        with pytest.raises(RopeError):
            r.get(time=QUERY_TIME, lst=12.0, lat=45.0, alt_km=1200.0)


# ---------------------------------------------------------------------------
# Discard-on-reforecast invariant
# ---------------------------------------------------------------------------

def test_second_forecast_discards_first(tmp_path):
    """A new `rope forecast` fully replaces the cache file -- there is no
    server holding old state around, so a query valid only under the first
    forecast's window must fail once a second, differently-shaped forecast
    has been written to the same cache path.
    """
    cache = str(tmp_path / "forecast_grid.bin")
    conf  = tmp_path / "rope.conf"
    _write_conf(conf)

    # horizon=3 -> window covers 2024-01-01T00:00:00 .. 03:00:00
    first = _rope(
        "--cache-path", cache, "forecast",
        "--start", FORECAST_START, "--horizon", "3", "--config", str(conf),
    )
    assert first.returncode == 0

    late_time = "2024-01-01T03:00:00"
    probe_a = _rope(
        "--cache-path", cache, "get", "--mode", "interp",
        "--time", late_time, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert probe_a.returncode == 0, "sanity check: first forecast's window must include late_time"

    # horizon=1 -> window covers 2024-01-01T00:00:00 .. 01:00:00
    second = _rope(
        "--cache-path", cache, "forecast",
        "--start", FORECAST_START, "--horizon", "1", "--config", str(conf),
    )
    assert second.returncode == 0

    probe_b = _rope(
        "--cache-path", cache, "get", "--mode", "interp",
        "--time", late_time, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
        timeout=10,
    )
    assert probe_b.returncode != 0, "old (horizon=3) window must be gone after re-forecasting with horizon=1"


# ---------------------------------------------------------------------------
# Pipeline load failure — a failed forecast never touches an existing cache
# ---------------------------------------------------------------------------

def test_bad_exported_dir_forecast_fails_but_existing_cache_survives(tmp_path):
    cache    = str(tmp_path / "forecast_grid.bin")
    good_conf = tmp_path / "rope_good.conf"
    _write_conf(good_conf)

    first = _rope(
        "--cache-path", cache, "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", str(good_conf),
    )
    assert first.returncode == 0, f"initial forecast failed:\n{first.stderr}"

    # Now point at a directory with no model artifacts at all -- the pipeline
    # fails to load. That failure must not touch the cache file written above.
    bad_conf  = tmp_path / "rope_bad.conf"
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    _write_conf(bad_conf, exported_dir=empty_dir)

    second = _rope(
        "--cache-path", cache, "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", str(bad_conf),
    )
    assert second.returncode != 0

    # The cache file from the first, successful forecast must still be intact.
    probe = _rope(
        "--cache-path", cache, "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert probe.returncode == 0, "a failed forecast must not clobber the existing cache file"


# ---------------------------------------------------------------------------
# Add-ons (docs/addons.md). The synthetic fixture model has no "addons" block, so these check
# the switches and error paths; add-on numerics are covered by tests/cpp/test_addons.cpp.
# ---------------------------------------------------------------------------

def test_forecast_addon_flags_off_match_default(forecast_cache):
    result = forecast_cache.run(
        "forecast", "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", forecast_cache.conf, "--da", "off", "--f10-forecaster", "off",
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["status"] == "ok"
    assert "addons" not in data


def test_forecast_addon_flag_rejects_bad_value(forecast_cache):
    result = forecast_cache.run(
        "forecast", "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", forecast_cache.conf, "--da", "maybe",
    )
    assert result.returncode != 0


def test_forecast_da_without_addon_fails_clearly(forecast_cache):
    result = forecast_cache.run(
        "forecast", "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", forecast_cache.conf, "--da", "on",
    )
    assert result.returncode != 0
    assert "addons.data_assimilation" in result.stderr


def test_forecast_f10_forecaster_without_addon_fails_clearly(forecast_cache):
    result = forecast_cache.run(
        "forecast", "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", forecast_cache.conf, "--f10-forecaster", "on",
    )
    assert result.returncode != 0
    assert "addons.f10_forecaster" in result.stderr


def test_forecast_cli_flag_overrides_enabled_addon_in_config(forecast_cache, tmp_path):
    conf = tmp_path / "rope_da_on.conf"
    _write_conf(conf)
    conf.write_text(conf.read_text() + "\n[data_assimilation]\nenabled = true\n")
    on = forecast_cache.run("forecast", "--start", FORECAST_START, "--horizon",
                            str(FORECAST_HORIZON), "--config", str(conf))
    assert on.returncode != 0 and "addons.data_assimilation" in on.stderr
    off = forecast_cache.run("forecast", "--start", FORECAST_START, "--horizon",
                             str(FORECAST_HORIZON), "--config", str(conf), "--da", "off")
    assert off.returncode == 0, off.stderr


def test_rope_binding_forecast_passes_addon_switches(tmp_path):
    sys.path.insert(0, str(_project_root / "python"))
    from rope import Rope, RopeError

    conf = tmp_path / "rope_binding.conf"
    _write_conf(conf)
    cache = str(tmp_path / "addon_switch_forecast_grid.bin")
    r = Rope(lib_path=Path(ROPE_LIB), exe_path=Path(ROPE_EXE), cache_path=cache, config_path=conf)

    ok = r.forecast(FORECAST_START, FORECAST_HORIZON, use_da=False, use_f10_forecaster=False)
    assert ok["status"] == "ok"
    with pytest.raises(RopeError, match="addons.data_assimilation"):
        r.forecast(FORECAST_START, FORECAST_HORIZON, use_da=True)
