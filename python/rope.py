"""
rope.py — Python binding for the ROPE framework.

Wraps librope.so via ctypes for fast in-process interpolation queries
against a memory-mapped forecast-grid cache file, and the rope CLI
subprocess to run forecasts (which write that cache file).

Typical usage
-------------
    from rope import Rope

    r = Rope() #config_path="config/rope.conf"
    r.forecast("2024-02-09 00:00:00", horizon=24)

    with r:   # opens handle, closes on exit
        result = r.get(time="2024-02-09T06:00:00Z", lst=7.5, lat=45.0, alt_km=400.0)
        print(result)  # {"density": 4.72e-12, "uncertainty": 3.5e-13}
"""

import atexit
import configparser
import ctypes
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROPE_HOLD   = 0
ROPE_INTERP = 1

_ERR_NAMES = {
    0: "ok",
    2: "no forecast cached",
    3: "time out of range",
    4: "spatial point out of range",
    5: "bad argument",
    6: "internal error",
    7: "forecast cache corrupt",
    8: "buffer too small",
}

_ERR_BUFFER_TOO_SMALL = 8


class RopeError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(f"[{_ERR_NAMES.get(code, str(code))}] {message}")
        self.code = code


class Rope:
    """
    Client for the ROPE atmospheric density service.

    Parameters
    ----------
    lib_path    : Path to librope.so / librope.dylib / librope.dll.
                  Defaults to lib/librope.so relative to the archive root.
    exe_path    : Path to the rope CLI executable.
                  Defaults to bin/rope relative to the archive root.
    cache_path  : Forecast-grid cache file path. None → platform default.
    config_path : Path to rope.conf. Defaults to config/rope.conf in the archive root.
    device      : Decoder device string (e.g. "cpu", "cuda", "cuda:1").
                  Defaults to the value in rope.conf. Only affects LibTorch builds;
                  ignored by ONNX Runtime builds.
    extrapolate_altitude : Enable/disable log-linear extrapolation past alt_max_km.
                  None (default) leaves the library default (on) in place.
                  Applied automatically on open(); see set_extrapolation().
    n_etp_pts   : Altitude bins used to fit the extrapolation slope. None (default)
                  leaves the library default (8) in place.
    """

    def __init__(
        self,
        lib_path: "str | Path | None" = None,
        exe_path: "str | Path | None" = None,
        cache_path: "str | None" = None,
        config_path: "str | Path | None" = None,
        device: "str | None" = None,
        extrapolate_altitude: "bool | None" = None,
        n_etp_pts: "int | None" = None,
    ):
        # rope.py lives in python/ inside the archive; bin/ and lib/ are one level up.
        root = Path(__file__).parent.parent

        if lib_path is None:
            candidates = [
                root / "lib" / "librope.so",
                root / "lib" / "librope.dylib",
                root / "bin" / "librope.dll",
                root / "build" / "librope.so",
            ]
            for c in candidates:
                if c.exists():
                    lib_path = c
                    break
            else:
                raise FileNotFoundError(
                    "librope not found; pass lib_path= explicitly or check your package layout"
                )

        if exe_path is None:
            for c in [root / "bin" / "rope", root / "bin" / "rope.exe", root / "build" / "rope"]:
                if c.exists():
                    exe_path = c
                    break

        resolved_conf = Path(config_path) if config_path else root / "config" / "rope.conf"
        conf_device   = _conf_get(resolved_conf, "decoder", "device", "cpu")
        self._device  = device if device is not None else conf_device

        if device is not None and device != conf_device:
            self._temp_conf_path = _write_temp_conf(resolved_conf, "decoder", "device", device)
            self._config_path    = Path(self._temp_conf_path)
        else:
            self._temp_conf_path = None
            self._config_path    = resolved_conf

        self._lib_path   = Path(lib_path)
        self._exe_path   = Path(exe_path) if exe_path else None
        self._cache_path = cache_path
        self._extrapolate_altitude = extrapolate_altitude
        self._n_etp_pts  = n_etp_pts
        self._handle: "int | None" = None
        self._lib        = self._load_lib()

        # Pre-allocated buffers for get_density() — avoids per-call allocation.
        self._qden = ctypes.c_double()
        self._qunc = ctypes.c_double()
        self._qerr = ctypes.create_string_buffer(256)

        atexit.register(self.shutdown)

    @property
    def device(self) -> str:
        """Active decoder device string."""
        return self._device

    def __del__(self):
        if getattr(self, "_temp_conf_path", None):
            try:
                os.unlink(self._temp_conf_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Context manager — opens/closes the interpolation handle
    # ------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Handle lifecycle
    # ------------------------------------------------------------------

    def open(self):
        """Memory-map the cached forecast grid and open an interpolation handle."""
        if self._handle is not None:
            return
        err   = ctypes.create_string_buffer(256)
        cache = self._cache_path.encode() if self._cache_path else None
        handle = self._lib.rope_open(cache, err, len(err))
        if not handle:
            raise RopeError(2, err.value.decode())
        self._handle = handle

        if self._extrapolate_altitude is not None or self._n_etp_pts is not None:
            self.set_extrapolation(
                extrapolate_altitude=self._extrapolate_altitude
                if self._extrapolate_altitude is not None else True,
                n_etp_pts=self._n_etp_pts,
            )

    def close(self):
        """Release the interpolation handle (unmaps the cache file)."""
        if self._handle is not None:
            self._lib.rope_close(self._handle)
            self._handle = None

    def refresh(self):
        """Re-map the current cache file (picks up a forecast written since open())."""
        self.close()
        self.open()

    def set_extrapolation(self, extrapolate_altitude: bool = True, n_etp_pts: "int | None" = None):
        """Reconfigures altitude extrapolation on the open handle; n_etp_pts=None keeps its current value."""
        if self._handle is None:
            self.open()
        err = ctypes.create_string_buffer(256)
        rc = self._lib.rope_set_extrapolation(
            self._handle,
            1 if extrapolate_altitude else 0,
            n_etp_pts if n_etp_pts is not None else 0,
            err, len(err),
        )
        if rc != 0:
            raise RopeError(rc, err.value.decode())

    def shutdown(self):
        """Alias for close(). There is no background process to stop --
        kept as a method (and registered with atexit in __init__) for
        source compatibility with existing scripts.
        """
        self.close()

    # ------------------------------------------------------------------
    # Forecast (via CLI subprocess)
    # ------------------------------------------------------------------

    def forecast(
        self,
        start: "str | datetime",
        horizon: int,
        drivers: "dict | None" = None,
        use_da: "bool | None" = None,
        use_f10_forecaster: "bool | None" = None,
    ) -> dict:
        """
        Run a forecast and atomically write the resulting grid to the cache
        file (discarding any previous forecast).

        Parameters
        ----------
        start   : Forecast start time — ISO 8601 string or datetime (UTC).
        horizon : Forecast duration in hours.
        drivers : Optional explicit driver data, e.g.
                      {"datetime": [...], "f10": [...], "kp": [...]}
                  overriding paths.driver_path/manifest.drivers.source for this
                  call only. Any raw column name is accepted, including
                  "f10_41day_avg" -- supplied directly it's used as-is;
                  omitted, it falls back to being computed from "f10" history.
                  Must cover the full contiguous hourly window the
                  model needs (history + horizon) — same requirement as an
                  explicit driver_path CSV, just supplied inline. All-or-nothing:
                  a column the model needs but this dict omits is not backfilled
                  from any other source.
        use_da  : Data-assimilation initial condition for this call (True/False);
                  None keeps rope.conf's [data_assimilation] enabled. Needs a model
                  with the add-on and a start inside its DA coverage.
        use_f10_forecaster : F10.7 from the forecaster after the start time for
                  this call (True/False); None keeps rope.conf's [f10_forecaster] enabled.

        Returns the CLI's response, e.g.:
            {"status": "ok", "window_start": "...", "window_end": "..."}
        plus an "addons" entry describing the DA state / F10 forecast used,
        when an add-on was on.
        """
        if self._exe_path is None:
            raise RuntimeError("rope executable not found; cannot run forecast")

        if isinstance(start, datetime):
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            start = start.strftime("%Y-%m-%d %H:%M:%S")

        # --cache-path is a global option and must precede the subcommand name.
        cmd = [str(self._exe_path)]
        if self._cache_path:
            cmd += ["--cache-path", self._cache_path]
        cmd += ["forecast", "--start", start, "--horizon", str(horizon)]
        if self._config_path:
            cmd += ["--config", str(self._config_path)]
        if use_da is not None:
            cmd += ["--da", "on" if use_da else "off"]
        if use_f10_forecaster is not None:
            cmd += ["--f10-forecaster", "on" if use_f10_forecaster else "off"]

        temp_driver_path = None
        try:
            if drivers is not None:
                temp_driver_path = _write_temp_driver_csv(drivers)
                cmd += ["--driver", temp_driver_path]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RopeError(6, (proc.stderr or proc.stdout).strip())

            # Take the last non-empty line — guards against any preamble lines.
            lines = [l for l in proc.stdout.splitlines() if l.strip()]
            return json.loads(lines[-1])
        finally:
            if temp_driver_path:
                try:
                    os.unlink(temp_driver_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Interpolation queries (via C API)
    # ------------------------------------------------------------------

    def get(
        self,
        time: "float | str | datetime",
        lst: float,
        lat: float,
        alt_km: float,
        mode: int = ROPE_INTERP,
    ) -> dict:
        """
        Query density and uncertainty at a single point.

        Parameters
        ----------
        time   : Query time — Unix timestamp, ISO 8601 string, or datetime (UTC).
        lst    : Local Solar Time, hours [0, 24).
        lat    : Geodetic latitude, degrees [-90, 90] (polar-cap blend beyond the grid's own range).
        alt_km : Geometric altitude, km. Below the grid's alt_min_km always throws; above alt_max_km
                 log-linearly extrapolates up to 2000 km by default (see set_extrapolation()).
        mode   : ROPE_INTERP (default) or ROPE_HOLD.

        Returns {"density": float, "uncertainty": float} in kg/m³.
        """
        if self._handle is None:
            self.open()

        density     = ctypes.c_double()
        uncertainty = ctypes.c_double()
        err         = ctypes.create_string_buffer(256)

        rc = self._lib.rope_query(
            self._handle, mode, _to_unix(time), lst, lat, alt_km,
            ctypes.byref(density), ctypes.byref(uncertainty),
            err, len(err),
        )
        if rc != 0:
            raise RopeError(rc, err.value.decode())

        return {"density": density.value, "uncertainty": uncertainty.value}

    def get_density(
        self,
        time: "float | str | datetime",
        lst: float,
        lat: float,
        alt_km: float,
        mode: int = ROPE_INTERP,
    ) -> float:
        """
        Query density at a single point, returning a bare float.

        Faster than get() for tight loops: no dict allocation, no uncertainty
        output, and reuses pre-allocated ctypes buffers across calls.
        """
        if self._handle is None:
            self.open()

        rc = self._lib.rope_query(
            self._handle, mode, _to_unix(time), lst, lat, alt_km,
            ctypes.byref(self._qden), ctypes.byref(self._qunc),
            self._qerr, len(self._qerr),
        )
        if rc != 0:
            raise RopeError(rc, self._qerr.value.decode())
        return self._qden.value

    def get_batch(
        self,
        times: list,
        lsts: list,
        lats: list,
        alts_km: list,
        mode: int = ROPE_INTERP,
    ) -> "list[dict]":
        """
        Query density and uncertainty at N points in one call.

        Each parameter is a list of length N.
        Returns a list of N dicts: [{"density": float, "uncertainty": float}, ...].
        """

        n = len(times)
        if not (len(lsts) == len(lats) == len(alts_km) == n):
            raise ValueError("all input lists must have the same length")

        if self._handle is None:
            self.open()

        DA      = ctypes.c_double * n
        t_arr   = DA(*(_to_unix(t) for t in times))
        lst_arr = DA(*lsts)
        lat_arr = DA(*lats)
        alt_arr = DA(*alts_km)
        den_arr = DA()
        unc_arr = DA()
        err     = ctypes.create_string_buffer(256)

        rc = self._lib.rope_query_batch(
            self._handle, mode, n,
            t_arr, lst_arr, lat_arr, alt_arr,
            den_arr, unc_arr,
            err, len(err),
        )
        if rc != 0:
            raise RopeError(rc, err.value.decode())

        return [{"density": den_arr[i], "uncertainty": unc_arr[i]} for i in range(n)]

    # ------------------------------------------------------------------
    # Manifest introspection (via C API)
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict:
        """
        Returns the model manifest summary: kind, latent_dim, grid, validated,
        ic.{kind, axes}, and drivers.{source, columns: [{name, description}, ...]}.

        Uses the fast ctypes path, like get()/get_batch() — this is read-only
        manifest introspection, not a forecast run, so it never shells out to
        the CLI subprocess.
        """
        exported_dir = str(self._exported_dir()).encode()
        err = ctypes.create_string_buffer(256)
        buf_len = 4096
        while True:
            buf = ctypes.create_string_buffer(buf_len)
            rc = self._lib.rope_get_manifest_info(exported_dir, buf, buf_len, err, len(err))
            if rc == 0:
                return json.loads(buf.value.decode())
            if rc == _ERR_BUFFER_TOO_SMALL:
                buf_len *= 4
                continue
            raise RopeError(rc, err.value.decode())

    def print_model(self):
        """Pretty-prints get_model_info() — what this model expects, without
        having to read model_manifest.json by hand."""
        info = self.get_model_info()
        print(f"Model kind:    {info['kind']}")
        print(f"Latent dim:    {info['latent_dim']}")
        print(f"Validated:     {info['validated']}")
        grid = info["grid"]
        print(f"Grid:          {grid['n_lst']} x {grid['n_lat']} x {grid['n_alt']}"
              f"  (lat {grid['lat_min_deg']}..{grid['lat_max_deg']} deg,"
              f" alt {grid['alt_min_km']}..{grid['alt_max_km']} km)")
        ic = info["ic"]
        print(f"IC:            kind={ic['kind']} axes={ic['axes']}")
        drivers = info["drivers"]
        print(f"Driver source: {drivers['source']}")
        print("Driver columns (in order):")
        for col in drivers["columns"]:
            print(f"  {col['name']}\t{col['description']}")

    def _exported_dir(self) -> Path:
        raw = _conf_get(self._config_path, "paths", "exported_dir", "")
        if not raw:
            raise RuntimeError(f"paths.exported_dir not set in {self._config_path}")
        p = Path(raw)
        if not p.is_absolute():
            p = self._config_path.parent / p
        return p

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_lib(self) -> ctypes.CDLL:
        lib = ctypes.CDLL(str(self._lib_path))

        lib.rope_open.restype  = ctypes.c_void_p
        lib.rope_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]

        lib.rope_query.restype  = ctypes.c_int
        lib.rope_query.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p, ctypes.c_int,
        ]

        lib.rope_query_batch.restype  = ctypes.c_int
        lib.rope_query_batch.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p, ctypes.c_int,
        ]

        lib.rope_set_extrapolation.restype  = ctypes.c_int
        lib.rope_set_extrapolation.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int,
        ]

        lib.rope_close.restype  = None
        lib.rope_close.argtypes = [ctypes.c_void_p]

        lib.rope_get_manifest_info.restype  = ctypes.c_int
        lib.rope_get_manifest_info.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int,
        ]

        return lib


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _conf_get(path: Path, section: str, key: str, fallback: str) -> str:
    cp = configparser.ConfigParser()
    if path.exists():
        cp.read(str(path))
    return cp.get(section, key, fallback=fallback)


def _write_temp_conf(base: Path, section: str, key: str, value: str) -> str:
    cp = configparser.ConfigParser()
    if base.exists():
        cp.read(str(base))
    if not cp.has_section(section):
        cp.add_section(section)
    cp.set(section, key, value)
    fd, path = tempfile.mkstemp(suffix=".conf", prefix="rope_")
    with os.fdopen(fd, "w") as f:
        cp.write(f)
    return path


def _write_temp_driver_csv(drivers: dict) -> str:
    if "datetime" not in drivers:
        raise ValueError("drivers dict must include a 'datetime' column")

    names   = list(drivers.keys())
    lengths = {len(v) for v in drivers.values()}
    if len(lengths) != 1:
        raise ValueError("all driver columns must have the same length")
    n = lengths.pop()

    fd, path = tempfile.mkstemp(suffix=".csv", prefix="rope_drivers_")
    with os.fdopen(fd, "w") as f:
        f.write(",".join(names) + "\n")
        for i in range(n):
            row = []
            for name in names:
                v = drivers[name][i]
                if name == "datetime" and isinstance(v, datetime):
                    if v.tzinfo is None:
                        v = v.replace(tzinfo=timezone.utc)
                    v = v.strftime("%Y-%m-%dT%H:%M:%S")
                row.append(str(v))
            f.write(",".join(row) + "\n")
    return path


def _to_unix(t: "float | str | datetime") -> float:
    if isinstance(t, datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.timestamp()
    if isinstance(t, str):
        # Accept ISO 8601 with or without Z/offset
        t = t.rstrip("Z")
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
            try:
                return datetime.strptime(t, fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                pass
        raise ValueError(f"cannot parse time string: {t!r}")
    return float(t)
