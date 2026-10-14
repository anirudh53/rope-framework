#pragma once
// forecast/pipeline.h — public interface for the ROPE forecast pipeline.
//
// Usage:
//   auto cfg = rope::forecast::Config{...};
//   auto pipe = rope::forecast::load(cfg);
//   ForecastGrid grid = pipe->run("2024-02-09 00:00:00", 120);

#include "rope/core/types.h"
#include "rope/io/config_reader.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rope::forecast {

// Paths and tuning parameters needed to construct a Pipeline.
struct Config {
    // Directory produced by export_models.py.
    std::filesystem::path exported_dir;

    // Explicit driver file (.swbin/.csv). Empty: use the manifest's drivers.source + DriverCacheManager.
    std::filesystem::path driver_path;

    // Cached .swbin directory. Empty: platform cache root.
    std::filesystem::path cache_dir;

    // Max cached driver file age before refresh.
    int cache_max_age_hours = 24;

    // ORT intra-op threads for the 15 base models.
    int intra_threads_base    = 1;

    // ORT intra-op threads for the meta model. 0 = hardware_concurrency().
    int intra_threads_meta    = 0;

    // ORT intra-op threads for the COAE decoder. 0 = hardware_concurrency().
    int intra_threads_decoder = 0;

    // LibTorch device string for the COAE decoder.
    std::string decoder_device = "cpu";

    // When false, skip the Unscented Transform; uncertainty is 0.
    bool compute_uncertainty = true;

    // Max forecast-hours of decoded voxel data held at once. <= 0: one chunk (whole horizon).
    int decode_chunk_hours = 72;

    // Caps decoder batch size (<= 0: manifest value, only ever lowered); trades decoder calls for memory; ULP-level float drift vs. other values.
    int decode_batch_size = 0;

    // Hours run before start_time to let the model settle; spin-up only, never decoded/cached. Default 0 here; rope.conf.in ships 48.
    // The forecast origin is start + warmup_time_hours (the CLI passes the warm-up start to run_streaming()).
    int warmup_time_hours = 0;

    // ---- Add-ons (docs/addons.md). Both off: the pipeline is unchanged. ----
    // [data_assimilation] enabled: seed the rollout with the assimilated state at the forecast origin.
    bool use_da = false;
    // .dabin override; empty: the manifest's addons.data_assimilation.file.
    std::filesystem::path da_states_path;
    // How the S-hour joint DA covariance becomes the covariance of the IC shift: hour_avg | mean | last.
    std::string da_ic_cov_mode = "hour_avg";
    // Multiplies the DA standard deviations.
    double da_ic_cov_scale = 1.0;

    // [f10_forecaster] enabled: F10.7 after the forecast origin comes from the F10 ensemble.
    bool use_f10_forecaster = false;
    // Ensemble type (a key of the manifest's addons.f10_forecaster.ensembles); empty: manifest default.
    std::string f10_ensemble;
    // .f10bin override; empty: from the manifest.
    std::filesystem::path f10_ensemble_path;
    // Fixed F10.7 sigma (sfu); <= 0: the ensemble spread averaged over the forecast days.
    double f10_sigma = 0.0;

    // Load-progress callback. Default (nullptr): silent.
    std::function<void(std::string_view)> log;
};

// GridChunkSink: one contiguous grid slice, increasing t_offset. LatentSink: full (H, latent_dim) trajectory, called once.
using GridChunkSink = std::function<void(int t_offset,
                                          std::span<const std::int64_t> times,
                                          std::span<const float> density,
                                          std::span<const float> uncertainty)>;
using LatentSink = std::function<void(std::span<const float> latent_mean)>;

// Abstract interface; constructed by load().
class Pipeline {
public:
    virtual ~Pipeline() = default;

    // Collects run_streaming() into one ForecastGrid; spans hour 0 through `horizon` inclusive (H_lat = horizon+1).
    ForecastGrid run(const std::string& start_iso, int horizon,
                     std::vector<float>* latent_mean_out = nullptr)
    {
        const int H_lat = horizon + 1;
        ForecastGrid grid;
        grid.shape = grid_shape();
        grid.H = H_lat;
        const std::size_t voxels = static_cast<std::size_t>(grid.shape.voxels());
        grid.times.resize(static_cast<std::size_t>(H_lat));
        grid.density.resize(static_cast<std::size_t>(H_lat) * voxels);
        grid.uncertainty.resize(static_cast<std::size_t>(H_lat) * voxels);

        run_streaming(start_iso, horizon, /*chunk_hours=*/H_lat,
            [&](int t_offset, std::span<const std::int64_t> times,
                std::span<const float> density, std::span<const float> uncertainty) {
                std::copy(times.begin(), times.end(), grid.times.begin() + t_offset);
                std::copy(density.begin(), density.end(),
                          grid.density.begin() + static_cast<std::size_t>(t_offset) * voxels);
                std::copy(uncertainty.begin(), uncertainty.end(),
                          grid.uncertainty.begin() + static_cast<std::size_t>(t_offset) * voxels);
            },
            latent_mean_out
                ? LatentSink{[latent_mean_out](std::span<const float> lm) {
                      latent_mean_out->assign(lm.begin(), lm.end());
                  }}
                : LatentSink{});
        return grid;
    }

    virtual void run_streaming(const std::string& start_iso, int horizon,
                                int chunk_hours, const GridChunkSink& sink,
                                const LatentSink& latent_sink = nullptr) = 0;

    virtual GridSpec grid_shape() const = 0;
    virtual std::string model_kind() const = 0;
    virtual int latent_dim() const = 0;

    // JSON object describing add-ons used by the last run_streaming() (empty: none / unsupported).
    virtual std::string run_info_json() const { return {}; }
};

// Constructs and initializes the pipeline from cfg. Throws on error.
std::unique_ptr<Pipeline> load(const Config& cfg);

// Builds a Config from a parsed rope.conf; relative paths resolve against config_dir.
Config config_from_reader(const io::ConfigReader& config,
                          const std::filesystem::path& config_dir);

} // namespace rope::forecast
