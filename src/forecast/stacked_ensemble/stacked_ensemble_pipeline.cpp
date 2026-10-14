#include "stacked_ensemble_pipeline.h"

#include "backends/decoder_factory.h"
#include "backends/ic_source_factory.h"
#include "backends/runtime_compat.h"
#include "sliding_window_rollout.h"
#include "unscented_transform.h"

#include "rope/io/driver_bin.h"
#include "rope/io/driver_cache.h"
#include "rope/core/platform.h"
#include "rope/core/text.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef ROPE_USE_OPENMP
#  include <omp.h>
#endif

namespace rope::forecast {

namespace fs = std::filesystem;

static void fill_driver(const io::DriverRow&            row,
                        const std::vector<std::string>& cols,
                        float*                          out) {
    for (int i = 0; i < static_cast<int>(cols.size()); ++i)
        out[i] = row.get(cols[i]);
}

static std::string join(const std::vector<std::string>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += v[i];
    }
    return s + "]";
}

static ModelBackend parse_backend(const std::string& s) {
    if (s == "onnx") return ModelBackend::ONNX;
#ifdef ROPE_USE_LIBTORCH
    if (s == "libtorch") return ModelBackend::LibTorch;
#endif
    throw std::runtime_error(
        "StackedEnsemblePipeline: backend '" + s + "' is not compiled in");
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

StackedEnsemblePipeline::StackedEnsemblePipeline(
    const Config& cfg, const io::ModelManifest& manifest)
{
    log_ = cfg.log ? cfg.log : [](std::string_view){};

    check_runtime_compat(manifest.runtime_requirements);

    const auto&     spec = *manifest.stacked_ensemble;
    const fs::path& dir  = cfg.exported_dir;

    K_            = manifest.latent_dim;
    S_            = spec.seq_len;
    M_            = static_cast<int>(spec.base_models.size());
    compute_uncertainty_ = cfg.compute_uncertainty;
    driver_cols_         = manifest.driver_columns;
    driver_source_       = manifest.driver_source;
    grid_shape_          = manifest.grid;
    manifest_kind_          = manifest.kind;

    io::Stats stats_ts = io::Stats::load(dir / "stats_ts.bin");
    ts_norm_ = std::make_unique<io::FeatureNormalizer>(stats_ts, K_);

    const int D  = ts_norm_->total_dim();
    const int dd = ts_norm_->driver_dim();

    if (dd != static_cast<int>(manifest.driver_columns.size()))
        throw std::runtime_error(
            "StackedEnsemblePipeline: manifest has " +
            std::to_string(manifest.driver_columns.size()) +
            " driver_columns but stats_ts.bin expects driver_dim=" +
            std::to_string(dd));

    load_ic_source(manifest, dir);
    log_("Loading decoder\xe2\x80\xa6");
    decoder_ = make_decoder(dir, manifest, cfg);
    load_sw_db(cfg);
    load_base_models(cfg, manifest, dir);
    load_meta_model(cfg, manifest, dir, D);

    rollout_ = std::make_unique<SlidingWindowRollout>(K_, S_, D);

    warmup_hours_ = cfg.warmup_time_hours;
    load_addons(cfg, manifest, dir);

    log_("Pipeline loaded.  total_dim=" + std::to_string(D) +
         "  driver_dim=" + std::to_string(dd) +
         "  source=" + (driver_source_.empty() ? "(explicit path)" : driver_source_) +
         "  uncertainty=" + (compute_uncertainty_ ? "on" : "off") +
         "  data_assimilation=" + (use_da_ ? "on" : "off") +
         "  f10_forecaster=" + (use_f10_ ? "on (" + f10_ensemble_name_ + ")" : "off"));
}

// ---------------------------------------------------------------------------
// Constructor helpers
// ---------------------------------------------------------------------------

// True if both name lists match pairwise, ignoring ASCII case.
static bool axis_names_match(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), rope::core::iequals_ascii);
}

void StackedEnsemblePipeline::load_ic_source(
    const io::ModelManifest& manifest, const fs::path& dir)
{
    log_("Loading IC source\xe2\x80\xa6");
    ic_source_ = make_ic_source(dir, manifest.ic_kind);
    if (ic_source_->latent_dim() != K_)
        throw std::runtime_error(
            "StackedEnsemblePipeline: manifest latent_dim=" +
            std::to_string(K_) + " does not match IC source latent_dim=" +
            std::to_string(ic_source_->latent_dim()));
    if (!axis_names_match(ic_source_->axis_names(), manifest.ic_grid_axes))
        throw std::runtime_error(
            "StackedEnsemblePipeline: IC source axes " + join(ic_source_->axis_names()) +
            " do not match manifest ic.params.grid_axes " + join(manifest.ic_grid_axes));
}

void StackedEnsemblePipeline::load_sw_db(const Config& cfg) {
    log_("Loading space-weather database\xe2\x80\xa6");
    fs::path path = cfg.driver_path;
    if (path.empty()) {
        if (driver_source_.empty())
            throw std::runtime_error(
                "StackedEnsemblePipeline: no driver_path set and manifest "
                "has no 'driver_source'; cannot locate driver data.");
        fs::path cache_dir = cfg.cache_dir.empty()
            ? platform::default_cache_dir()
            : cfg.cache_dir;
        io::DriverCacheManager mgr{cache_dir, cfg.cache_max_age_hours};
        path = mgr.get_path(driver_source_);
    }
    sw_db_ = std::make_unique<io::SpaceWeatherDB>(
        io::SpaceWeatherDB::from_file(path));
}

void StackedEnsemblePipeline::load_base_models(
    const Config& cfg, const io::ModelManifest& manifest, const fs::path& dir)
{
    const auto& spec = *manifest.stacked_ensemble;
    log_("Loading " + std::to_string(M_) + " base models\xe2\x80\xa6");
    base_models_.reserve(M_);
    for (const auto& bm : spec.base_models) {
        base_models_.push_back(
            make_model((dir / bm.file).string(),
                       parse_backend(bm.backend),
                       cfg.intra_threads_base,
                       bm.inter_op_threads,
                       false, "cpu"));
    }
}

void StackedEnsemblePipeline::load_meta_model(
    const Config& cfg, const io::ModelManifest& manifest,
    const fs::path& dir, int D)
{
    const auto& spec = *manifest.stacked_ensemble;
    int meta_threads = cfg.intra_threads_meta;
    if (meta_threads <= 0)
        meta_threads = static_cast<int>(std::thread::hardware_concurrency());

    log_("Loading meta model\xe2\x80\xa6");
    meta_model_ = std::make_unique<EnsembleFuser>(
        make_model((dir / spec.meta_model.file).string(),
                   parse_backend(spec.meta_model.backend),
                   meta_threads, /*inter_op_threads=*/1,
                   false, "cpu"),
        K_, S_, D, M_);
}

// ---------------------------------------------------------------------------
// Sequence building (formerly SequenceBuilder)
// ---------------------------------------------------------------------------

std::vector<float> StackedEnsemblePipeline::build_X_init_norm(
    const std::vector<io::DriverRow>& hist_rows, const float* latents) const
{
    const int K  = K_;
    const int D  = ts_norm_->total_dim();
    const int Dd = ts_norm_->driver_dim();

    const auto&        ic_axes = ic_source_->axis_names();
    std::vector<float> axis_vals(ic_axes.size());

    std::vector<float> X(static_cast<size_t>(S_) * D, 0.0f);
    for (int s = 0; s < S_; ++s) {
        const io::DriverRow& row  = hist_rows[s];
        float*               dest = X.data() + s * D;

        if (latents) {
            // Add-on override (e.g. data assimilation): IC latents supplied, oldest row first.
            std::copy(latents + s * K, latents + (s + 1) * K, dest);
        } else {
            for (std::size_t a = 0; a < ic_axes.size(); ++a)
                axis_vals[a] = row.get(ic_axes[a]);
            std::vector<float> coeffs = ic_source_->get_latent_coeffs(axis_vals);
            for (int k = 0; k < K; ++k)
                dest[k] = coeffs[k];
        }

        fill_driver(row, driver_cols_, dest + K);
        ts_norm_->norm_full_inplace(dest);
    }
    (void)Dd;
    return X;
}

std::vector<float> StackedEnsemblePipeline::build_x_chunk(
    const std::vector<float>&        X_init_norm,
    const std::vector<io::DriverRow>& fcast_rows,
    int                              n_windows) const
{
    const int K  = K_;
    const int D  = ts_norm_->total_dim();
    const int Dd = ts_norm_->driver_dim();

    std::vector<float> chunk(static_cast<size_t>(n_windows) * S_ * D, 0.0f);
    std::copy(X_init_norm.begin(), X_init_norm.end(), chunk.begin());

    std::vector<float> drv(Dd);
    for (int t = 1; t < n_windows; ++t) {
        const float* prev = chunk.data() + (t - 1) * S_ * D;
        float*       curr = chunk.data() +  t      * S_ * D;

        std::copy(prev + D, prev + S_ * D, curr);

        float* last_row = curr + (S_ - 1) * D;
        std::fill(last_row, last_row + K, 0.0f);
        fill_driver(fcast_rows[t], driver_cols_, drv.data());
        ts_norm_->norm_driver_inplace(drv.data());
        std::copy(drv.begin(), drv.end(), last_row + K);
    }
    (void)K;
    return chunk;
}

// ---------------------------------------------------------------------------
// run() phases
// ---------------------------------------------------------------------------

std::vector<float> StackedEnsemblePipeline::run_base_rollout(
    const std::vector<float>& x_chunk, int H) const
{
    const int K = K_;
    const int M = M_;
    const int T = H;

    std::vector<float> base_latents(static_cast<size_t>(M) * T * K, 0.0f);

#ifdef ROPE_USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int m = 0; m < M; ++m) {
        float* out = base_latents.data() + m * T * K;
        rollout_->run(*base_models_[m], x_chunk.data(), H, out);
    }
    return base_latents;
}

StackedEnsemblePipeline::MeanLatents
StackedEnsemblePipeline::compute_mean_latents(
    const std::vector<float>& x_chunk,
    const std::vector<float>& base_latents_norm,
    const std::vector<float>& X_init, int H) const
{
    const int K = K_;
    const int S = S_;
    const int D = ts_norm_->total_dim();
    const int T = H;

    FusionResult meta_out = meta_model_->fuse(
        x_chunk.data(), T, base_latents_norm.data());
    auto meta_mean_norm = std::move(meta_out.mean);
    ts_norm_->denorm_latents_block(meta_mean_norm.data(), T);

    std::vector<float> init_lat(K);
    std::copy(X_init.begin() + (S - 1) * D,
              X_init.begin() + (S - 1) * D + K,
              init_lat.begin());
    ts_norm_->denorm_latents_inplace(init_lat.data());

    const int H_lat = H + 1;
    std::vector<float> mu_lat(static_cast<size_t>(H_lat) * K);
    std::copy(init_lat.begin(), init_lat.end(), mu_lat.begin());
    std::copy(meta_mean_norm.begin(), meta_mean_norm.end(), mu_lat.begin() + K);

    return {std::move(mu_lat), std::move(init_lat)};
}

// Rollout+fusion stay whole-horizon (cheap, latent-space); only decode is chunked.
void StackedEnsemblePipeline::run_streaming(
    const std::string& start_iso, int horizon, int chunk_hours,
    const GridChunkSink& sink, const LatentSink& latent_sink)
{
    if (use_da_ || use_f10_) {
        run_streaming_addons(start_iso, horizon, chunk_hours, sink, latent_sink);
        return;
    }
    run_info_.clear();

    const int H = horizon;
    const int S = S_;
    if (chunk_hours <= 0) chunk_hours = horizon + 1;

    std::vector<io::DriverRow> all_rows =
        io::DriverWindowBuilder::build(*sw_db_, start_iso, H + 1, S);

    std::vector<io::DriverRow> hist_rows(all_rows.begin(), all_rows.begin() + S);
    std::vector<io::DriverRow> fcast_rows(
        all_rows.begin() + S - 1,
        all_rows.begin() + S - 1 + H + 1);

    auto X_init  = build_X_init_norm(hist_rows);
    auto x_chunk = build_x_chunk(X_init, fcast_rows, H + 1);
    auto base_latents = run_base_rollout(x_chunk, H);
    auto [mu_lat, init_lat] = compute_mean_latents(x_chunk, base_latents, X_init, H);

    if (latent_sink)
        latent_sink(std::span(mu_lat));

    const int H_lat  = H + 1;
    const int voxels = grid_shape_.voxels();

    std::vector<std::int64_t> times(H_lat);
    for (int t = 0; t < H_lat; ++t) times[t] = fcast_rows[t].tp;

    if (compute_uncertainty_) {
        const int M = M_;
        ts_norm_->denorm_latents_block(base_latents.data(), M * H);

        auto ut = ut_sigma_points(mu_lat, base_latents, init_lat, M, K_, H_lat);
        const int N_SIG = ut.N_SIG;

        for (int t_lat = 0; t_lat < H_lat; t_lat += chunk_hours) {
            const int count_lat = std::min(chunk_hours, H_lat - t_lat);
            auto sigma_slice = std::span(ut.sigma_lat)
                .subspan(static_cast<std::size_t>(t_lat) * N_SIG * K_,
                         static_cast<std::size_t>(count_lat) * N_SIG * K_);

            std::vector<float> dens_sigmas =
                decoder_->decode(sigma_slice, count_lat * N_SIG, K_);

            const std::size_t local_voxels = static_cast<std::size_t>(count_lat) * voxels;
            std::vector<float> density_mean(local_voxels, 0.0f);
            for (int lt = 0; lt < count_lat; ++lt)
                for (int sig = 0; sig < N_SIG; ++sig) {
                    const float* src = dens_sigmas.data() +
                                       (static_cast<std::size_t>(lt) * N_SIG + sig) * voxels;
                    float* dst = density_mean.data() + static_cast<std::size_t>(lt) * voxels;
                    const float w = ut.Wm[sig];
                    for (int v = 0; v < voxels; ++v)
                        dst[v] += w * src[v];
                }

            std::vector<float> uncertainty(local_voxels, 0.0f);
            for (int lt = 0; lt < count_lat; ++lt)
                for (int sig = 0; sig < N_SIG; ++sig) {
                    const float* src = dens_sigmas.data() +
                                       (static_cast<std::size_t>(lt) * N_SIG + sig) * voxels;
                    const float* mu  = density_mean.data() +
                                       static_cast<std::size_t>(lt) * voxels;
                    float* dst = uncertainty.data() + static_cast<std::size_t>(lt) * voxels;
                    const float w = ut.Wc[sig];
                    for (int v = 0; v < voxels; ++v) {
                        float d = src[v] - mu[v];
                        dst[v] += w * d * d;
                    }
                }
            for (float& u : uncertainty)
                u = std::sqrt(std::max(u, 0.0f));

            sink(t_lat,
                 std::span(times).subspan(static_cast<std::size_t>(t_lat),
                                          static_cast<std::size_t>(count_lat)),
                 std::span(density_mean),
                 std::span(uncertainty));
        }
    } else {
        for (int t_lat = 0; t_lat < H_lat; t_lat += chunk_hours) {
            const int count_lat = std::min(chunk_hours, H_lat - t_lat);
            auto mu_slice = std::span(mu_lat)
                .subspan(static_cast<std::size_t>(t_lat) * K_, static_cast<std::size_t>(count_lat) * K_);

            std::vector<float> density = decoder_->decode(mu_slice, count_lat, K_);
            std::vector<float> uncertainty(static_cast<std::size_t>(count_lat) * voxels, 0.0f);

            sink(t_lat,
                 std::span(times).subspan(static_cast<std::size_t>(t_lat),
                                          static_cast<std::size_t>(count_lat)),
                 std::span(density),
                 std::span(uncertainty));
        }
    }
}

// ---------------------------------------------------------------------------
// Add-ons: data-assimilation initial conditions and the F10.7 forecaster (docs/addons.md)
// ---------------------------------------------------------------------------

namespace {

constexpr TimePoint HOUR = 3600;
constexpr TimePoint DAY  = 86400;
constexpr int F10_AVG_ROWS = 41 * 24;   // same trailing window as SpaceWeatherDB::compute_f10_41day_avg

// S-hour joint DA covariance (SK x SK) -> K x K covariance of the common IC shift.
std::vector<double> reduce_da_cov(const std::vector<float>& sj, int S, int K, const std::string& mode) {
    const int SK = S * K;
    std::vector<double> out(static_cast<std::size_t>(K) * K, 0.0);
    auto at = [&](int r, int c) { return static_cast<double>(sj[static_cast<std::size_t>(r) * SK + c]); };
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j) {
            double v = 0.0;
            if (mode == "hour_avg") {          // mean of the per-hour blocks
                for (int s = 0; s < S; ++s) v += at(s * K + i, s * K + j);
                v /= S;
            } else if (mode == "mean") {       // covariance of the mean state: A Sigma A^T, A = [I..I]/S
                for (int s = 0; s < S; ++s)
                    for (int r = 0; r < S; ++r) v += at(s * K + i, r * K + j);
                v /= static_cast<double>(S) * S;
            } else {                           // "last": the block at the forecast origin
                v = at((S - 1) * K + i, (S - 1) * K + j);
            }
            out[static_cast<std::size_t>(i) * K + j] = v;
        }
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < i; ++j) {
            const double m = 0.5 * (out[i * K + j] + out[j * K + i]);
            out[i * K + j] = out[j * K + i] = m;
        }
    return out;
}

// Lower Cholesky factor of a (n x n, row-major), double; false if not positive definite.
bool cholesky(std::vector<double> a, int n, std::vector<double>& L) {
    L.assign(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = a[i * n + j];
            for (int k = 0; k < j; ++k) s -= L[i * n + k] * L[j * n + k];
            if (i == j) {
                if (!(s > 0.0)) return false;
                L[i * n + i] = std::sqrt(s);
            } else {
                L[i * n + j] = s / L[j * n + j];
            }
        }
    return true;
}

void set_raw(io::DriverRow& r, const std::string& name, float v) {
    for (auto& [n, val] : r.raw)
        if (n == name) { val = v; return; }
}

}  // namespace

void StackedEnsemblePipeline::load_addons(
    const Config& cfg, const io::ModelManifest& manifest, const fs::path& dir)
{
    use_da_          = cfg.use_da;
    use_f10_         = cfg.use_f10_forecaster;
    da_ic_cov_mode_  = cfg.da_ic_cov_mode;
    da_ic_cov_scale_ = cfg.da_ic_cov_scale;
    f10_sigma_fixed_ = cfg.f10_sigma;

    if (use_da_) {
        if (da_ic_cov_mode_ != "hour_avg" && da_ic_cov_mode_ != "mean" && da_ic_cov_mode_ != "last")
            throw std::runtime_error("data_assimilation.ic_cov_mode must be hour_avg, mean or last (got '" +
                                     da_ic_cov_mode_ + "')");
        if (!(da_ic_cov_scale_ >= 0.0))
            throw std::runtime_error("data_assimilation.ic_cov_scale must be >= 0");
        fs::path p = cfg.da_states_path;
        if (p.empty()) {
            if (manifest.addons.da_file.empty())
                throw std::runtime_error(
                    "data_assimilation is enabled, but this model's manifest has no "
                    "addons.data_assimilation and data_assimilation.states_path is not set");
            p = dir / manifest.addons.da_file;
        }
        log_("Loading data-assimilation states\xe2\x80\xa6");
        da_table_ = std::make_unique<io::DAStateTable>(io::DAStateTable::load(p));
        if (da_table_->latent_dim() != K_ || da_table_->seq_len() != S_)
            throw std::runtime_error(
                "data assimilation: " + p.string() + " has seq_len=" + std::to_string(da_table_->seq_len()) +
                " latent_dim=" + std::to_string(da_table_->latent_dim()) + ", the model has seq_len=" +
                std::to_string(S_) + " latent_dim=" + std::to_string(K_));
    }

    if (use_f10_) {
        if (std::find(driver_cols_.begin(), driver_cols_.end(), "f10") == driver_cols_.end())
            throw std::runtime_error("f10_forecaster is enabled, but this model has no 'f10' driver");
        fs::path    p    = cfg.f10_ensemble_path;
        std::string name = cfg.f10_ensemble.empty() ? manifest.addons.f10_default : cfg.f10_ensemble;
        if (p.empty()) {
            if (manifest.addons.f10_files.empty())
                throw std::runtime_error(
                    "f10_forecaster is enabled, but this model's manifest has no "
                    "addons.f10_forecaster and f10_forecaster.ensemble_path is not set");
            if (name.empty()) name = manifest.addons.f10_files.begin()->first;
            auto it = manifest.addons.f10_files.find(name);
            if (it == manifest.addons.f10_files.end()) {
                std::string known;
                for (const auto& [k, _] : manifest.addons.f10_files) known += (known.empty() ? "" : ", ") + k;
                throw std::runtime_error("f10_forecaster.ensemble '" + name + "' not in the manifest (" + known + ")");
            }
            p = dir / it->second;
        }
        f10_ensemble_name_ = name.empty() ? p.stem().string() : name;
        log_("Loading F10.7 forecaster (" + f10_ensemble_name_ + ")\xe2\x80\xa6");
        f10_ensemble_ = std::make_unique<F10Ensemble>(F10Ensemble::load(p));
    }
}

StackedEnsemblePipeline::F10Plan StackedEnsemblePipeline::plan_f10(TimePoint origin, TimePoint last) const {
    F10Plan pl;
    pl.origin  = origin;
    pl.day0    = origin - origin % DAY;
    pl.n_cover = static_cast<int>((last - last % DAY - pl.day0) / DAY) + 1;
    const int n_days = pl.n_cover + 1;   // + the next day's 00 UT anchor for interpolation

    // Daily history = F10.7 at 00 UT on each of the preceding days (the daily value in both the
    // linear and the PCHIP hourly conversions).
    const int nh = f10_ensemble_->history_days();
    std::vector<double> hist(static_cast<std::size_t>(nh));
    const TimePoint first = pl.day0 - static_cast<TimePoint>(nh) * DAY;
    if (first < sw_db_->time_min())
        throw std::runtime_error("f10_forecaster: needs daily F10.7 from " + format_iso(first) +
                                 "; the driver data starts at " + format_iso(sw_db_->time_min()));
    for (int i = 0; i < nh; ++i)
        hist[i] = sw_db_->lookup(first + static_cast<TimePoint>(i) * DAY).get("f10");

    F10Forecast fc = f10_ensemble_->forecast(hist, n_days);
    pl.mean = std::move(fc.mean);
    pl.std  = std::move(fc.std);
    if (f10_sigma_fixed_ > 0.0) {
        pl.sigma = f10_sigma_fixed_;
    } else {
        double s = 0.0;
        for (int d = 0; d < pl.n_cover; ++d) s += pl.std[d];
        pl.sigma = s / pl.n_cover;
    }
    return pl;
}

// Hours after the origin take the forecast F10.7 (linear between daily 00 UT anchors) + offset;
// their f10_41day_avg is recomputed over the trailing 984 hours including those replacements.
std::vector<io::DriverRow> StackedEnsemblePipeline::apply_f10(
    const std::vector<io::DriverRow>& rows, const F10Plan& plan, double offset) const
{
    auto fc_at = [&](TimePoint t) {
        const double x = static_cast<double>(t - plan.day0) / DAY;
        const int last = static_cast<int>(plan.mean.size()) - 1;
        if (x <= 0.0) return plan.mean.front();
        if (x >= last) return plan.mean.back();
        const int i = static_cast<int>(x);
        return plan.mean[i] + (x - i) * (plan.mean[i + 1] - plan.mean[i]);
    };

    std::vector<io::DriverRow> out = rows;
    if (out.empty() || out.back().tp <= plan.origin) return out;

    const TimePoint t_first = std::max(out.front().tp - static_cast<TimePoint>(F10_AVG_ROWS - 1) * HOUR,
                                       sw_db_->time_min());
    const std::size_t n = static_cast<std::size_t>((out.back().tp - t_first) / HOUR) + 1;
    std::vector<double> csum(n + 1, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const TimePoint t = t_first + static_cast<TimePoint>(i) * HOUR;
        const double v = t > plan.origin ? fc_at(t) + offset : sw_db_->lookup(t).get("f10");
        csum[i + 1] = csum[i] + v;
    }

    for (auto& r : out) {
        if (r.tp <= plan.origin) continue;
        const float f10 = static_cast<float>(fc_at(r.tp) + offset);
        set_raw(r, "f10", f10);
        const std::size_t idx = static_cast<std::size_t>((r.tp - t_first) / HOUR);
        const std::size_t lo  = idx + 1 >= static_cast<std::size_t>(F10_AVG_ROWS) ? idx + 1 - F10_AVG_ROWS : 0;
        const float avg = static_cast<float>((csum[idx + 1] - csum[lo]) / static_cast<double>(idx + 1 - lo));
        r.f10_41day_avg = avg;
        set_raw(r, "f10_41day_avg", avg);   // only if the source supplied the column itself
    }
    return out;
}

StackedEnsemblePipeline::MemberRun StackedEnsemblePipeline::run_member(
    const std::vector<io::DriverRow>& all_rows, int H, const float* latents) const
{
    std::vector<io::DriverRow> hist_rows(all_rows.begin(), all_rows.begin() + S_);
    std::vector<io::DriverRow> fcast_rows(all_rows.begin() + S_ - 1, all_rows.begin() + S_ - 1 + H + 1);
    auto X_init  = build_X_init_norm(hist_rows, latents);
    auto x_chunk = build_x_chunk(X_init, fcast_rows, H + 1);
    MemberRun m;
    m.base_latents = run_base_rollout(x_chunk, H);
    auto [mu, init] = compute_mean_latents(x_chunk, m.base_latents, X_init, H);
    m.mu_lat   = std::move(mu);
    m.init_lat = std::move(init);
    return m;
}

void StackedEnsemblePipeline::run_streaming_addons(
    const std::string& start_iso, int horizon, int chunk_hours,
    const GridChunkSink& sink, const LatentSink& latent_sink)
{
    const int H     = horizon;
    const int H_lat = H + 1;
    const int K     = K_;
    if (chunk_hours <= 0) chunk_hours = H_lat;

    const TimePoint run_start = floor_hour(parse_datetime(start_iso));
    const TimePoint origin    = run_start + static_cast<TimePoint>(warmup_hours_) * HOUR;
    if (use_da_ && warmup_hours_ != 0)
        throw std::runtime_error("data assimilation seeds the forecast at its origin; "
                                 "warm-up must be 0 when it is enabled");

    std::vector<io::DriverRow> all_rows = io::DriverWindowBuilder::build(*sw_db_, start_iso, H + 1, S_);
    std::vector<std::int64_t> times(H_lat);
    for (int t = 0; t < H_lat; ++t) times[t] = all_rows[S_ - 1 + t].tp;

    nlohmann::json info;

    // ---- F10.7 forecast after the origin
    F10Plan plan;
    std::vector<io::DriverRow> rows0 = all_rows;
    if (use_f10_) {
        plan  = plan_f10(origin, all_rows.back().tp);
        rows0 = apply_f10(all_rows, plan, 0.0);
        auto daily = nlohmann::json::array();
        for (std::size_t d = 0; d < plan.mean.size(); ++d)
            daily.push_back({{"date", format_iso(plan.day0 + static_cast<TimePoint>(d) * DAY).substr(0, 10)},
                             {"mean", plan.mean[d]}, {"std", plan.std[d]}});
        info["f10_forecaster"] = {{"enabled", true}, {"ensemble", f10_ensemble_name_},
                                  {"members", f10_ensemble_->n_members()},
                                  {"origin", format_iso(origin)}, {"sigma_sfu", plan.sigma},
                                  {"sigma_from", f10_sigma_fixed_ > 0.0 ? "config" : "ensemble spread"},
                                  {"daily", daily}};
    }

    // ---- data-assimilation IC at the origin
    std::vector<float>  z_rows;
    std::vector<double> S_ic;
    if (use_da_) {
        io::DAState st = da_table_->at(origin);
        z_rows = st.z;
        S_ic   = reduce_da_cov(st.sigma_joint, S_, K, da_ic_cov_mode_);
        for (double& v : S_ic) v *= da_ic_cov_scale_ * da_ic_cov_scale_;
        double sd = 0.0;
        for (int k = 0; k < K; ++k) sd += std::sqrt(std::max(S_ic[k * K + k], 0.0));
        info["data_assimilation"] = {{"enabled", true}, {"state_time", format_iso(origin)},
                                     {"n_obs_span", st.n_obs_span},
                                     {"ic_cov_mode", da_ic_cov_mode_}, {"ic_cov_scale", da_ic_cov_scale_},
                                     {"ic_sigma_mean", sd / K}};
    }
    const float* z0 = use_da_ ? z_rows.data() : nullptr;

    const int n_ic = (use_da_  && compute_uncertainty_) ? K : 0;
    const int n_f  = (use_f10_ && compute_uncertainty_) ? 1 : 0;
    const int n    = n_ic + n_f;
    info["uncertainty"] = {{"model", compute_uncertainty_}, {"input_dims", n},
                           {"input_sigma_points", n > 0 ? 2 * n + 1 : 0}};
    run_info_ = info.dump();

    const int voxels = grid_shape_.voxels();

    // ---- no uncertainty: one deterministic pass with the add-on inputs
    if (!compute_uncertainty_) {
        MemberRun c = run_member(rows0, H, z0);
        if (latent_sink) latent_sink(std::span(c.mu_lat));
        for (int t_lat = 0; t_lat < H_lat; t_lat += chunk_hours) {
            const int count_lat = std::min(chunk_hours, H_lat - t_lat);
            auto mu_slice = std::span(c.mu_lat).subspan(static_cast<std::size_t>(t_lat) * K,
                                                        static_cast<std::size_t>(count_lat) * K);
            std::vector<float> density = decoder_->decode(mu_slice, count_lat, K);
            std::vector<float> uncertainty(static_cast<std::size_t>(count_lat) * voxels, 0.0f);
            sink(t_lat, std::span(times).subspan(static_cast<std::size_t>(t_lat), static_cast<std::size_t>(count_lat)),
                 std::span(density), std::span(uncertainty));
        }
        return;
    }

    // ---- input unscented transform over [IC shift (K, DA) | F10 offset (1)], alpha=1 beta=2 kappa=0
    std::vector<double> m(static_cast<std::size_t>(n), 0.0), cov(static_cast<std::size_t>(n) * n, 0.0);
    std::vector<double> z_bar(K, 0.0);
    if (use_da_)
        for (int s = 0; s < S_; ++s)
            for (int k = 0; k < K; ++k) z_bar[k] += z_rows[s * K + k] / static_cast<double>(S_);
    for (int i = 0; i < n_ic; ++i) {
        m[i] = z_bar[i];
        for (int j = 0; j < n_ic; ++j) cov[i * n + j] = S_ic[i * K + j];
    }
    if (n_f) cov[n_ic * n + n_ic] = plan.sigma * plan.sigma;

    const int    B = 2 * n + 1;
    const double c = static_cast<double>(n);
    std::vector<double> Wm(B, 1.0 / (2.0 * c)), Wc(B, 1.0 / (2.0 * c));
    Wm[0] = 0.0;
    Wc[0] = 2.0;
    std::vector<double> L, ccov(cov);
    for (double& v : ccov) v *= c;
    if (!cholesky(ccov, n, L)) {
        for (int i = 0; i < n; ++i) ccov[i * n + i] += 1e-12;
        if (!cholesky(ccov, n, L))
            throw std::runtime_error("input UT: covariance is not positive definite");
    }

    std::vector<MemberRun> runs(B);
    for (int b = 0; b < B; ++b) {
        std::vector<double> p(m);
        if (b > 0) {
            const int j = (b - 1) % n;
            const double sgn = b <= n ? 1.0 : -1.0;
            for (int i = 0; i < n; ++i) p[i] += sgn * L[i * n + j];
        }
        std::vector<float> lat;
        if (n_ic) {
            lat.resize(static_cast<std::size_t>(S_) * K);
            for (int s = 0; s < S_; ++s)
                for (int k = 0; k < K; ++k)
                    lat[s * K + k] = static_cast<float>(z_rows[s * K + k] + (p[k] - z_bar[k]));
        }
        const float* lat_ptr = n_ic ? lat.data() : z0;
        const double offset  = n_f ? p[n_ic] : 0.0;
        runs[b] = run_member(b == 0 || !n_f ? rows0 : apply_f10(all_rows, plan, offset), H, lat_ptr);
    }
    const MemberRun& central = runs[0];
    if (latent_sink) latent_sink(std::span(central.mu_lat));

    // Model (ensemble) UT at the central input, as in the base pipeline.
    std::vector<float> base_phys = central.base_latents;
    ts_norm_->denorm_latents_block(base_phys.data(), M_ * H);
    auto ut = ut_sigma_points(central.mu_lat, base_phys, central.init_lat, M_, K, H_lat);
    const int N_SIG = ut.N_SIG;

    double sum_wc = 0.0;
    for (double w : Wc) sum_wc += w;

    for (int t_lat = 0; t_lat < H_lat; t_lat += chunk_hours) {
        const int count_lat = std::min(chunk_hours, H_lat - t_lat);
        const std::size_t local_voxels = static_cast<std::size_t>(count_lat) * voxels;

        // model variance about the model-UT mean
        auto sigma_slice = std::span(ut.sigma_lat).subspan(static_cast<std::size_t>(t_lat) * N_SIG * K,
                                                           static_cast<std::size_t>(count_lat) * N_SIG * K);
        std::vector<float> dens_sigmas = decoder_->decode(sigma_slice, count_lat * N_SIG, K);
        std::vector<float> model_mean(local_voxels, 0.0f), model_var(local_voxels, 0.0f);
        for (int lt = 0; lt < count_lat; ++lt)
            for (int sig = 0; sig < N_SIG; ++sig) {
                const float* src = dens_sigmas.data() + (static_cast<std::size_t>(lt) * N_SIG + sig) * voxels;
                float* dst = model_mean.data() + static_cast<std::size_t>(lt) * voxels;
                for (int v = 0; v < voxels; ++v) dst[v] += ut.Wm[sig] * src[v];
            }
        for (int lt = 0; lt < count_lat; ++lt)
            for (int sig = 0; sig < N_SIG; ++sig) {
                const float* src = dens_sigmas.data() + (static_cast<std::size_t>(lt) * N_SIG + sig) * voxels;
                const float* mu  = model_mean.data() + static_cast<std::size_t>(lt) * voxels;
                float* dst = model_var.data() + static_cast<std::size_t>(lt) * voxels;
                for (int v = 0; v < voxels; ++v) { const float d = src[v] - mu[v]; dst[v] += ut.Wc[sig] * d * d; }
            }
        dens_sigmas.clear();
        dens_sigmas.shrink_to_fit();

        // input UT moments: mean = sum Wm x; var = sum Wc (x - mean)^2 = C - 2 mean Bw + mean^2 sum(Wc)
        std::vector<double> A(local_voxels, 0.0), Bw(local_voxels, 0.0), C(local_voxels, 0.0);
        for (int b = 0; b < B; ++b) {
            auto mu_slice = std::span(runs[b].mu_lat).subspan(static_cast<std::size_t>(t_lat) * K,
                                                              static_cast<std::size_t>(count_lat) * K);
            std::vector<float> dens = decoder_->decode(mu_slice, count_lat, K);
            for (std::size_t v = 0; v < local_voxels; ++v) {
                const double x = dens[v];
                A[v]  += Wm[b] * x;
                Bw[v] += Wc[b] * x;
                C[v]  += Wc[b] * x * x;
            }
        }
        std::vector<float> density(local_voxels), uncertainty(local_voxels);
        for (std::size_t v = 0; v < local_voxels; ++v) {
            const double in_var = std::max(C[v] - 2.0 * A[v] * Bw[v] + A[v] * A[v] * sum_wc, 0.0);
            density[v]     = static_cast<float>(A[v]);
            uncertainty[v] = static_cast<float>(std::sqrt(in_var + std::max(static_cast<double>(model_var[v]), 0.0)));
        }
        sink(t_lat, std::span(times).subspan(static_cast<std::size_t>(t_lat), static_cast<std::size_t>(count_lat)),
             std::span(density), std::span(uncertainty));
    }
}

} // namespace rope::forecast
