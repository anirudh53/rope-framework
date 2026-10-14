// rope — ROPE atmospheric density forecasting CLI.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
//
// Commands:
//   rope forecast --start <ISO8601> --horizon <h> [--config <path>] [--driver <path>]
//   rope get      --mode hold|interp --time <ISO8601> --lst <f> --lat <f> --alt <f>
//                 [--config <path>] [--no-extrapolate-altitude] [--n-etp-pts <N>]
//   rope get      --mode hold|interp --file <csv> [--output <path>]
//   rope manifest [--config <path>] [--json]
//
// `rope forecast` runs inference in this process and atomically writes the
// resulting grid to a per-user cache file; `rope get` reads that file
// (memory-mapped) and interpolates locally. No background server, no IPC.

#include "rope/core/datetime.h"
#include "rope/core/platform.h"
#include "rope/interpolate/grid_interpolator.h"
#include "rope/io/config_reader.h"
#include "rope/io/csv_reader.h"
#include "rope/io/driver_cache.h"
#include "rope/io/driver_db.h"
#include "rope/io/driver_bin.h"
#include "rope/io/forecast_grid_bin.h"
#include "rope/io/ic_table.h"
#include "rope/io/ic_bin.h"
#include "rope/io/mapped_forecast_grid.h"
#include "rope/io/model_manifest.h"

#ifdef ROPE_HAS_ZARR
#include "rope/io/forecast_zarr_writer.h"
#endif

#ifdef ROPE_HAS_FORECAST
#include "rope/forecast/pipeline.h"
#endif

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static fs::path exe_path() {
    return rope::platform::exe_path();
}

static fs::path default_config(const fs::path& exe) {
    // <exe_dir>/../config/rope.conf
    return exe.parent_path().parent_path() / "config" / "rope.conf";
}

// ---------------------------------------------------------------------------
// Batch-file processing for 'rope get --file'
// ---------------------------------------------------------------------------

static int run_batch_get(rope::interpolate::GridInterpolator<rope::io::MappedForecastGrid>& interp,
                          const std::string& mode,
                          const fs::path& file_path,
                          const fs::path& output_path) {
    rope::io::CsvReader csv{file_path};
    std::size_t N = csv.nrows();

    json out = json::array();
    for (std::size_t i = 0; i < N; ++i) {
        int yr  = csv.get_int("YYYY", i);
        int mo  = csv.get_int("MM",   i);
        int dy  = csv.get_int("DD",   i);
        int hr  = csv.get_int("HH",   i);
        int mn  = csv.get_int("MIN",  i);
        int sc  = csv.get_int("SS",   i);

        char buf[24];
        std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d",
                      yr, mo, dy, hr, mn, sc);
        std::string time_iso{buf};

        double lst    = csv.get_float("lst",    i);
        double lat    = csv.get_float("lat",    i);
        double alt_km = csv.get_float("alt_km", i);

        auto tp = rope::parse_datetime(time_iso);
        auto r  = (mode == "hold") ? interp.query_hold(tp, lst, lat, alt_km)
                                    : interp.query_interp(tp, lst, lat, alt_km);

        out.push_back({
            {"time",        time_iso},
            {"lst",         lst},
            {"lat",         lat},
            {"alt_km",      alt_km},
            {"density",     r.density},
            {"uncertainty", r.uncertainty}
        });
    }

    std::string text = out.dump(2);
    if (output_path.empty()) {
        std::cout << text << "\n";
    } else {
        std::ofstream f{output_path};
        if (!f) {
            std::cerr << "rope: cannot open output file: " << output_path << "\n";
            return 1;
        }
        f << text << "\n";
    }
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    CLI::App app{"ROPE atmospheric density forecasting service", "rope"};
    app.require_subcommand(0, 1);

    // ---- hidden --cache-path override (testing / multi-instance use) ----
    std::string cli_cache_path;
    app.add_option("--cache-path", cli_cache_path,
                   "Override the forecast-grid cache file path")->group("");

    // ---- forecast subcommand ----
    auto* fc = app.add_subcommand("forecast",
                                   "Run a forecast and cache the resulting grid");
    std::string fc_start, fc_config, fc_zarr, fc_driver, fc_da, fc_f10;
    int         fc_horizon = 0;
    fc->add_option("--start",   fc_start,   "Forecast start time (ISO 8601, UTC)")
      ->required();
    fc->add_option("--horizon", fc_horizon, "Forecast duration in hours")
      ->required();
    fc->add_option("--config",  fc_config,  "Config file path");
    fc->add_option("--zarr",    fc_zarr,
                   "Also export as a Zarr store (container directory)");
    fc->add_option("--driver",  fc_driver,
                   "Explicit driver .swbin/CSV path in ROPE's own format (overrides paths.driver_path; "
                   "raw CelesTrak CSVs need 'rope convert-sw' first)");
    fc->add_option("--da", fc_da,
                   "Data-assimilation initial condition for this run: on|off (overrides [data_assimilation] enabled)")
      ->check(CLI::IsMember({"on", "off"}));
    fc->add_option("--f10-forecaster", fc_f10,
                   "F10.7 forecaster after the start time for this run: on|off (overrides [f10_forecaster] enabled)")
      ->check(CLI::IsMember({"on", "off"}));

    // ---- manifest subcommand ----
    auto* mc = app.add_subcommand("manifest",
                                   "Print the model manifest (driver columns, grid shape, etc.)");
    std::string mc_config;
    bool        mc_json = false;
    mc->add_option("--config", mc_config, "Config file path");
    mc->add_flag("--json",     mc_json,   "Print as JSON instead of a human-readable summary");

    // ---- get subcommand ----
    auto* gc = app.add_subcommand("get",
                                   "Query the cached forecast grid");
    std::string gc_mode, gc_time, gc_file, gc_output, gc_config;
    double      gc_lst = 0, gc_lat = 0, gc_alt = 0;
    bool        gc_no_extrapolate_alt = false;
    int         gc_n_etp_pts = 0;
    gc->add_option("--mode", gc_mode, "hold or interp")->required();
    gc->add_option("--time", gc_time, "Query time (ISO 8601, UTC)");
    gc->add_option("--lst",  gc_lst,  "Local Solar Time [hours]");
    gc->add_option("--lat",  gc_lat,  "Geodetic latitude [degrees]");
    gc->add_option("--alt",  gc_alt,  "Altitude [km]");
    gc->add_option("--file", gc_file, "Batch CSV input (replaces point flags)");
    gc->add_option("--output", gc_output, "Output file for --file results");
    gc->add_option("--config", gc_config,
                   "Config file path (optional; only [interpolation] settings apply)");
    gc->add_flag("--no-extrapolate-altitude", gc_no_extrapolate_alt,
                 "Disable log-linear extrapolation past the grid's alt_max_km");
    gc->add_option("--n-etp-pts", gc_n_etp_pts,
                   "Altitude bins used to fit the extrapolation slope (<=0 = use config/default)");

    // ---- convert-sw subcommand ----
    auto* sw = app.add_subcommand("convert-sw",
                                   "Convert a space-weather CSV to .swbin binary format");
    std::string sw_input, sw_output;
    sw->add_option("--input",  sw_input,
                   "Input CSV file (ROPE canonical or raw CelesTrak; format auto-detected)")
      ->required();
    sw->add_option("--output", sw_output, "Output .swbin file")
      ->required();

    // ---- convert-ic subcommand ----
    auto* ic = app.add_subcommand("convert-ic",
                                   "Convert an IC-table CSV to .icbin binary format");
    std::string ic_input, ic_output;
    ic->add_option("--input",  ic_input,  "Input CSV file (F10,Kp,y1,...,yK)")
      ->required();
    ic->add_option("--output", ic_output, "Output .icbin file")
      ->required();

    CLI11_PARSE(app, argc, argv);

    // ---- Determine cache and config paths ----
    fs::path cache_path = cli_cache_path.empty()
        ? rope::platform::default_forecast_cache_path()
        : fs::path{cli_cache_path};

    // ---- forecast ----
    if (fc->parsed()) {
#ifndef ROPE_HAS_FORECAST
        std::cerr << "rope forecast: this build was compiled without ONNX Runtime; "
                     "forecast is unavailable\n";
        return 1;
#else
        fs::path config_path = fc_config.empty()
            ? default_config(exe_path())
            : fs::path{fc_config};
#ifndef ROPE_HAS_ZARR
        if (!fc_zarr.empty()) {
            std::cerr << "rope forecast: this build was compiled without Zarr support; "
                         "--zarr is unavailable\n";
            return 1;
        }
#endif
        try {
            rope::io::ConfigReader config{config_path};
            auto fcfg = rope::forecast::config_from_reader(config, config_path.parent_path());
            if (!fc_driver.empty())
                fcfg.driver_path = fs::path{fc_driver};
            if (!fc_da.empty())  fcfg.use_da             = fc_da  == "on";
            if (!fc_f10.empty()) fcfg.use_f10_forecaster = fc_f10 == "on";
            // The assimilated state is the estimate at the start time itself: no warm-up with DA.
            if (fcfg.use_da && fcfg.warmup_time_hours > 0) {
                std::cerr << "rope forecast: data assimilation on, warm-up ("
                          << fcfg.warmup_time_hours << "h) skipped\n";
                fcfg.warmup_time_hours = 0;
            }
            auto pipeline = rope::forecast::load(fcfg);

            // Output spans hour 0 (fc_start) through hour fc_horizon, inclusive.
            const int total_steps = fc_horizon + 1;

            // Warm-up (rope.conf's [forecast] warmup_time_hours): runs from fc_start-warmup, but only writes [fc_start, fc_start+horizon].
            if (fcfg.warmup_time_hours < 0)
                throw std::runtime_error(
                    "forecast.warmup_time_hours must be >= 0 (got " +
                    std::to_string(fcfg.warmup_time_hours) + ")");
            const int warmup_hours    = fcfg.warmup_time_hours;
            const int run_horizon     = fc_horizon + warmup_hours;
            const int run_total_steps = run_horizon + 1;
            const std::string run_start_iso = warmup_hours > 0
                ? rope::format_iso(rope::parse_datetime(fc_start) -
                                   static_cast<rope::TimePoint>(warmup_hours) * 3600)
                : fc_start;

            const long long voxels = pipeline->grid_shape().voxels();

            {
                // Grid-stitch buffers only; excludes decoder forward-pass memory (dominant, architecture-dependent, not derivable here).
                const int n_sig = fcfg.compute_uncertainty
                    ? 2 * pipeline->latent_dim() + 1 : 1;
                const int effective_chunk_hours = std::min(
                    fcfg.decode_chunk_hours > 0 ? fcfg.decode_chunk_hours : run_total_steps,
                    run_total_steps);
                const long long floats_per_chunk = fcfg.compute_uncertainty
                    ? static_cast<long long>(effective_chunk_hours) * voxels * (n_sig + 2)
                    : static_cast<long long>(effective_chunk_hours) * voxels * 2;
                const double est_mb =
                    static_cast<double>(floats_per_chunk) * 4 / (1024.0 * 1024.0);
                std::cerr << "rope forecast: decode_chunk_hours=" << fcfg.decode_chunk_hours
                          << " (effective=" << effective_chunk_hours << ")"
                          << "  estimated grid-buffer memory ~" << est_mb << " MB"
                          << " (excludes decoder network working memory)";
                if (warmup_hours > 0)
                    std::cerr << "  [+" << warmup_hours << "h warm-up, not cached]";
                std::cerr << "\n";
            }

            auto writer = rope::io::ForecastGridBinWriter::open(
                pipeline->grid_shape(), total_steps, cache_path);

#ifdef ROPE_HAS_ZARR
            std::optional<rope::io::ForecastZarrWriter> zarr;
            rope::forecast::LatentSink latent_sink;
            if (!fc_zarr.empty()) {
                latent_sink = [&](std::span<const float> latent_mean) {
                    zarr = rope::io::ForecastZarrWriter::open(
                        pipeline->grid_shape(), total_steps, pipeline->latent_dim(),
                        pipeline->model_kind(),
                        fc_start, fs::path{fc_zarr});
                    // Drops the leading warmup_hours*latent_dim floats from the (run_horizon+1)*latent_dim delivery.
                    const std::size_t skip = static_cast<std::size_t>(warmup_hours) *
                                             static_cast<std::size_t>(pipeline->latent_dim());
                    zarr->write_latent(latent_mean.subspan(skip));
                };
            }
#endif

            std::string window_start, window_end;
            pipeline->run_streaming(run_start_iso, run_horizon, fcfg.decode_chunk_hours,
                [&](int t_offset, std::span<const std::int64_t> times,
                    std::span<const float> density, std::span<const float> uncertainty) {
                    // Slices rather than skips whole chunks, since a chunk may straddle the warm-up boundary.
                    if (t_offset + static_cast<int>(times.size()) <= warmup_hours) return;
                    const int skip = std::max(0, warmup_hours - t_offset);
                    const int out_t_offset = std::max(0, t_offset - warmup_hours);
                    const auto voxels_sz = static_cast<std::size_t>(voxels);
                    auto times_trim       = times.subspan(static_cast<std::size_t>(skip));
                    auto density_trim     = density.subspan(static_cast<std::size_t>(skip) * voxels_sz);
                    auto uncertainty_trim = uncertainty.subspan(static_cast<std::size_t>(skip) * voxels_sz);

                    writer.write_chunk(out_t_offset, times_trim, density_trim, uncertainty_trim);
#ifdef ROPE_HAS_ZARR
                    if (zarr) zarr->write_chunk(out_t_offset, times_trim, density_trim, uncertainty_trim);
#endif
                    if (out_t_offset == 0) window_start = rope::format_iso(times_trim.front());
                    window_end = rope::format_iso(times_trim.back());
                }
#ifdef ROPE_HAS_ZARR
                , latent_sink
#endif
            );

            writer.close();

            json status{
                {"status",       "ok"},
                {"window_start", window_start},
                {"window_end",   window_end}
            };
            if (const std::string info = pipeline->run_info_json(); !info.empty())
                status["addons"] = json::parse(info);

#ifdef ROPE_HAS_ZARR
            if (zarr) {
                zarr->close();
                status["zarr_path"] = zarr->store_path().string();
            }
#endif

            std::cout << status.dump() << "\n";
        } catch (const std::exception& e) {
            std::cerr << "rope forecast: " << e.what() << "\n";
            std::cerr.flush();
            std::_Exit(1);
        }
        // The cache file is complete and the status printed; leave without running static
        // destructors. ONNX Runtime's own globals can crash during exit teardown when its CUDA
        // execution provider was used (heap corruption inside libonnxruntime's exit handlers).
        std::cout.flush();
        std::fflush(nullptr);
        std::_Exit(0);
#endif
    }

    // ---- manifest ----
    if (mc->parsed()) {
        try {
            fs::path config_path = mc_config.empty()
                ? default_config(exe_path())
                : fs::path{mc_config};
            rope::io::ConfigReader config{config_path};
            fs::path exported_dir{config.get("paths.exported_dir")};
            if (exported_dir.is_relative())
                exported_dir = config_path.parent_path() / exported_dir;

            auto manifest = rope::io::ModelManifest::load(exported_dir);

            if (mc_json) {
                std::cout << json::parse(manifest.to_summary_json()).dump(2) << "\n";
            } else {
                std::cout << "Model kind:    " << manifest.kind << "\n";
                std::cout << "Latent dim:    " << manifest.latent_dim << "\n";
                std::cout << "Validated:     " << (manifest.validated ? "true" : "false") << "\n";
                std::cout << "Grid:          " << manifest.grid.n_lst << " x " << manifest.grid.n_lat
                          << " x " << manifest.grid.n_alt << "  (lat " << manifest.grid.lat_min_deg
                          << ".." << manifest.grid.lat_max_deg << " deg, alt " << manifest.grid.alt_min_km
                          << ".." << manifest.grid.alt_max_km << " km)\n";
                std::cout << "IC:            kind=" << manifest.ic_kind << " axes=[";
                for (std::size_t i = 0; i < manifest.ic_grid_axes.size(); ++i)
                    std::cout << (i ? ", " : "") << manifest.ic_grid_axes[i];
                std::cout << "]\n";
                std::cout << "Driver source: " << manifest.driver_source << "\n";
                std::cout << "Driver columns (in order):\n";
                for (const auto& col : manifest.driver_column_info)
                    std::cout << "  " << col.name << "\t" << col.description << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "rope manifest: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    // ---- get ----
    if (gc->parsed()) {
        try {
            rope::io::MappedForecastGrid grid = rope::io::MappedForecastGrid::open(cache_path);

            // --config is optional here (unlike forecast/manifest): 'rope get' has never
            // required a config file, so only read one if explicitly given or the default exists.
            rope::interpolate::ExtrapolationOptions etp_opts;
            fs::path gc_config_path = gc_config.empty() ? default_config(exe_path()) : fs::path{gc_config};
            if (!gc_config.empty() || fs::exists(gc_config_path)) {
                rope::io::ConfigReader config{gc_config_path};
                etp_opts = rope::interpolate::options_from_reader(config);
            }
            if (gc_no_extrapolate_alt)
                etp_opts.extrapolate_altitude = false;
            if (gc_n_etp_pts > 0)
                etp_opts.n_etp_pts = gc_n_etp_pts;

            rope::interpolate::GridInterpolator<rope::io::MappedForecastGrid> interp{grid, etp_opts};

            if (!gc_file.empty()) {
                return run_batch_get(interp, gc_mode,
                                     fs::path{gc_file},
                                     fs::path{gc_output});
            }

            // Single-point query
            if (gc_time.empty()) {
                std::cerr << "rope get: --time is required when --file is not given\n";
                return 1;
            }
            auto tp = rope::parse_datetime(gc_time);
            auto r  = (gc_mode == "hold") ? interp.query_hold(tp, gc_lst, gc_lat, gc_alt)
                                          : interp.query_interp(tp, gc_lst, gc_lat, gc_alt);
            std::cout << json{
                {"status",      "ok"},
                {"density",     r.density},
                {"uncertainty", r.uncertainty}
            }.dump() << "\n";
        } catch (const rope::io::ForecastCacheMissingError&) {
            std::cerr << "rope get: no forecast cached; run 'rope forecast' first.\n";
            return 1;
        } catch (const std::exception& e) {
            std::cerr << "rope get: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    // ---- convert-sw ----
    if (sw->parsed()) {
        try {
            // Detect raw CelesTrak (header line starts "DATE") vs. ROPE canonical ("datetime") format.
            std::ifstream probe(sw_input);
            if (!probe) throw std::runtime_error("cannot open " + sw_input);
            std::string first_line;
            while (std::getline(probe, first_line) && first_line.empty()) {}
            probe.close();

            if (first_line.starts_with("DATE")) {
                std::cout << "Detected raw CelesTrak format\n";
                std::ifstream in(sw_input, std::ios::binary);
                if (!in) throw std::runtime_error("cannot open " + sw_input);
                std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                std::cout << "Writing " << sw_output << "…\n";
                rope::io::convert_celestrak_csv_to_swbin(raw, fs::path{sw_output});
                std::cout << "Done.\n";
            } else {
                std::cout << "Detected ROPE canonical format\n";
                std::cout << "Loading " << sw_input << "…\n";
                auto db = rope::io::SpaceWeatherDB::from_file(fs::path{sw_input});
                std::cout << "  " << db.size() << " rows  ["
                          << rope::format_iso(db.time_min()) << " → "
                          << rope::format_iso(db.time_max()) << "]\n";
                std::cout << "Writing " << sw_output << "…\n";
                rope::io::SpaceWeatherBin::save(db, fs::path{sw_output});
                std::cout << "Done.\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "rope convert-sw: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    // ---- convert-ic ----
    if (ic->parsed()) {
        try {
            std::cout << "Loading " << ic_input << "…\n";
            auto table = rope::io::ICTable::from_file(fs::path{ic_input});
            std::cout << "  latent_dim=" << table.latent_dim() << "\n";
            std::cout << "Writing " << ic_output << "…\n";
            rope::io::IcBin::save(table, fs::path{ic_output});
            std::cout << "Done.\n";
        } catch (const std::exception& e) {
            std::cerr << "rope convert-ic: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    std::cout << app.help();
    return 0;
}
