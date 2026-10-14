#pragma once

#include "rope/forecast/pipeline.h"
#include "rope/io/model_manifest.h"
#include "rope/io/driver_db.h"
#include "rope/io/stats.h"
#include "rope/io/da_states.h"

#include "addons/f10_ensemble.h"

#include "backends/decoder.h"
#include "backends/ic_source.h"
#include "backends/model_interface.h"
#include "ensemble_fuser.h"
#include "rollout_strategy.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rope::forecast {

class StackedEnsemblePipeline : public Pipeline {
public:
    StackedEnsemblePipeline(const Config& cfg, const io::ModelManifest& manifest);
    void run_streaming(const std::string& start_iso, int horizon, int chunk_hours,
                       const GridChunkSink& sink, const LatentSink& latent_sink) override;
    GridSpec grid_shape() const override { return grid_shape_; }
    std::string model_kind() const override { return manifest_kind_; }
    int latent_dim() const override { return K_; }
    std::string run_info_json() const override { return run_info_; }

private:
    // --- Scalars / config ---
    int  K_, S_, M_;
    bool compute_uncertainty_{true};
    GridSpec grid_shape_;
    std::string manifest_kind_;

    std::vector<std::string>               driver_cols_;
    std::string                            driver_source_;
    std::function<void(std::string_view)>  log_;

    // --- Data sources ---
    std::unique_ptr<io::SpaceWeatherDB>    sw_db_;
    std::unique_ptr<IICSource>             ic_source_;
    std::unique_ptr<io::FeatureNormalizer> ts_norm_;

    // --- Models ---
    std::vector<std::unique_ptr<IModel>>   base_models_;
    std::unique_ptr<EnsembleFuser>         meta_model_;

    std::unique_ptr<IDecoder>          decoder_;
    std::unique_ptr<IRolloutStrategy>  rollout_;

    // --- Add-ons (docs/addons.md) ---
    bool        use_da_{false}, use_f10_{false};
    int         warmup_hours_{0};
    std::string da_ic_cov_mode_{"hour_avg"};
    double      da_ic_cov_scale_{1.0};
    double      f10_sigma_fixed_{0.0};
    std::string f10_ensemble_name_;
    std::unique_ptr<io::DAStateTable> da_table_;
    std::unique_ptr<F10Ensemble>      f10_ensemble_;
    std::string run_info_;   // JSON summary of the last run's add-on state

    // F10.7 forecast for one run: daily anchors (00 UT) after the forecast origin.
    struct F10Plan {
        TimePoint           origin = 0;
        TimePoint           day0 = 0;
        std::vector<double> mean, std;   // per day from day0
        double              sigma = 0.0;
        int                 n_cover = 0;
    };

    // One deterministic pass for given IC latents (S*K, nullptr = IC table) and F10 offset.
    struct MemberRun {
        std::vector<float> mu_lat;        // (H_lat, K) denormed
        std::vector<float> init_lat;      // (K)
        std::vector<float> base_latents;  // (M, H, K) normalized
    };

    void load_addons(const Config& cfg, const io::ModelManifest& manifest,
                     const std::filesystem::path& dir);
    F10Plan plan_f10(TimePoint origin, TimePoint last) const;
    std::vector<io::DriverRow> apply_f10(const std::vector<io::DriverRow>& rows,
                                         const F10Plan& plan, double offset) const;
    MemberRun run_member(const std::vector<io::DriverRow>& all_rows, int H,
                         const float* latents) const;
    void run_streaming_addons(const std::string& start_iso, int horizon, int chunk_hours,
                              const GridChunkSink& sink, const LatentSink& latent_sink);

    // --- Constructor helpers ---
    void load_ic_source(const io::ModelManifest& manifest,
                         const std::filesystem::path& dir);
    void load_sw_db(const Config& cfg);
    void load_base_models(const Config& cfg, const io::ModelManifest& manifest,
                           const std::filesystem::path& dir);
    void load_meta_model(const Config& cfg, const io::ModelManifest& manifest,
                          const std::filesystem::path& dir, int D);

    // --- Sequence building (formerly SequenceBuilder) ---
    std::vector<float> build_X_init_norm(
        const std::vector<io::DriverRow>& hist_rows,
        const float* latents = nullptr) const;   // latents: (S*K) override of the IC lookup
    std::vector<float> build_x_chunk(
        const std::vector<float>& X_init_norm,
        const std::vector<io::DriverRow>& fcast_rows, int n_windows) const;

    // --- run() phases ---
    std::vector<float> run_base_rollout(
        const std::vector<float>& x_chunk, int H) const;

    struct MeanLatents {
        std::vector<float> mu_lat;    // (H_lat, K) — IC prepended, denormed
        std::vector<float> init_lat;  // (K) — IC latent at t=0, denormed
    };
    MeanLatents compute_mean_latents(
        const std::vector<float>& x_chunk,
        const std::vector<float>& base_latents_norm,
        const std::vector<float>& X_init, int H) const;
};

} // namespace rope::forecast
