// Add-ons: data-assimilation initial conditions and the F10.7 forecaster (docs/addons.md).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "rope/forecast/pipeline.h"
#include "rope/io/da_states.h"
#include "rope/io/driver_db.h"
#include "rope/io/model_manifest.h"
#include "addons/f10_ensemble.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

namespace {

constexpr int K = 10, S = 3;

// Unique per test run and per call, so parallel ctest invocations never share a directory.
fs::path unique_temp_dir(const std::string& tag) {
    static std::atomic<int> counter{0};
    std::random_device rd;
    fs::path d = fs::temp_directory_path() /
                 ("rope_addons_" + tag + "_" + std::to_string(rd()) + "_" + std::to_string(counter++));
    fs::create_directories(d);
    return d;
}

template <typename T> void put(std::ofstream& f, T v) { f.write(reinterpret_cast<const char*>(&v), sizeof(T)); }

// .dabin with one record per hour from t0 (inclusive), n hours. z = fixed small latents
// (distinct per row), covariance = diag(var).
void write_dabin(const fs::path& p, rope::TimePoint t0, int n, float var) {
    std::ofstream f(p, std::ios::binary);
    const std::string src = "unit test";
    put<std::uint32_t>(f, 0x52504441u); put<std::uint32_t>(f, 1u);
    put<std::uint32_t>(f, static_cast<std::uint32_t>(n));
    put<std::uint32_t>(f, S); put<std::uint32_t>(f, K);
    put<std::uint32_t>(f, static_cast<std::uint32_t>(src.size())); f.write(src.data(), static_cast<std::streamsize>(src.size()));
    const int SK = S * K;
    for (int i = 0; i < n; ++i) {
        put<std::int64_t>(f, t0 + static_cast<rope::TimePoint>(i) * 3600);
        put<std::int32_t>(f, 7);
        for (int s = 0; s < S; ++s)
            for (int k = 0; k < K; ++k) put<float>(f, 0.01f * static_cast<float>(k - 4) + 0.002f * static_cast<float>(s));
        for (int r = 0; r < SK; ++r)
            for (int c = 0; c < SK; ++c) put<float>(f, r == c ? var : 0.0f);
    }
}

// 60 days of hourly f10/kp ending after 2024-01-04 — enough history for the F10 forecaster.
fs::path write_long_drivers(const fs::path& dir) {
    const fs::path p = dir / "sw_60d.csv";
    std::ofstream f(p);
    f << "datetime,f10,kp\n";
    const rope::TimePoint t0 = rope::parse_datetime("2023-11-06 00:00:00");
    for (int h = 0; h < 60 * 24; ++h) {
        const rope::TimePoint t = t0 + static_cast<rope::TimePoint>(h) * 3600;
        const double day = h / 24.0;
        f << rope::format_iso(t) << "," << 150.0 + 20.0 * std::sin(2 * 3.141592653589793 * day / 27.0) << ","
          << 2.0 + std::sin(h / 7.0) << "\n";
    }
    return p;
}

// Copy of the synthetic model with an "addons" block pointing at a DA table and an F10 ensemble.
fs::path model_with_addons(const fs::path& dir, float da_var) {
    const fs::path m = dir / "model";
    fs::create_directories(m / "addons");
    for (const auto& e : fs::directory_iterator(fs::path(ROPE_FIXTURE_MODELS)))
        if (e.is_regular_file()) fs::copy_file(e.path(), m / e.path().filename());
    write_dabin(m / "addons" / "da.dabin", rope::parse_datetime("2024-01-01 00:00:00"), 72, da_var);
    fs::copy_file(fs::path(ROPE_CPP_FIXTURE_DIR) / "addons" / "f10_tiny.f10bin", m / "addons" / "f10_tiny.f10bin");

    nlohmann::json j;
    std::ifstream(m / "model_manifest.json") >> j;
    j["addons"] = {
        {"data_assimilation", {{"kind", "da_state_table"}, {"file", "addons/da.dabin"}}},
        {"f10_forecaster", {{"default", "tiny"},
                            {"ensembles", {{"tiny", {{"kind", "f10_ensemble"}, {"file", "addons/f10_tiny.f10bin"}}}}}}}};
    std::ofstream(m / "model_manifest.json") << j.dump(2);
    return m;
}

rope::forecast::Config base_cfg(const fs::path& model, const fs::path& drivers) {
    rope::forecast::Config cfg;
    cfg.exported_dir          = model;
    cfg.driver_path           = drivers;
    cfg.intra_threads_base    = 1;
    cfg.intra_threads_meta    = 1;
    cfg.intra_threads_decoder = 1;
    cfg.compute_uncertainty   = true;
    return cfg;
}

const char* START = "2024-01-02 06:00:00";
constexpr int HORIZON = 6;

}  // namespace

// ---------------------------------------------------------------------------
// File formats / manifest
// ---------------------------------------------------------------------------

TEST_CASE("Addons: .dabin round-trips and refuses hours it does not cover") {
    const auto dir = unique_temp_dir("dabin");
    const auto t0 = rope::parse_datetime("2024-01-01 00:00:00");
    write_dabin(dir / "t.dabin", t0, 5, 1e-4f);
    auto t = rope::io::DAStateTable::load(dir / "t.dabin");
    CHECK(t.size() == 5);
    CHECK(t.seq_len() == S);
    CHECK(t.latent_dim() == K);
    CHECK(t.source() == "unit test");
    auto st = t.at(t0 + 2 * 3600);
    CHECK(st.n_obs_span == 7);
    REQUIRE(st.z.size() == static_cast<std::size_t>(S * K));
    CHECK_THAT(st.z[2 * K + 9], WithinAbs(0.01 * 5 + 0.004, 1e-7));
    CHECK_THAT(st.sigma_joint[0], WithinAbs(1e-4, 1e-10));
    CHECK_THROWS_WITH(t.at(t0 + 99 * 3600), ContainsSubstring("DA table covers"));
    fs::remove_all(dir);
}

TEST_CASE("Addons: .dabin with a bad magic fails loudly") {
    const auto dir = unique_temp_dir("dabin_bad");
    std::ofstream(dir / "bad.dabin", std::ios::binary) << std::string(64, 'x');
    CHECK_THROWS_WITH(rope::io::DAStateTable::load(dir / "bad.dabin"), ContainsSubstring("bad magic"));
    fs::remove_all(dir);
}

TEST_CASE("Addons: manifest without an addons block has none") {
    auto m = rope::io::ModelManifest::load(fs::path(ROPE_FIXTURE_MODELS));
    CHECK(m.addons.da_file.empty());
    CHECK(m.addons.f10_files.empty());
}

TEST_CASE("Addons: manifest addons block is parsed") {
    const auto dir = unique_temp_dir("manifest");
    auto m = rope::io::ModelManifest::load(model_with_addons(dir, 1e-4f));
    CHECK(m.addons.da_file == "addons/da.dabin");
    CHECK(m.addons.f10_default == "tiny");
    REQUIRE(m.addons.f10_files.count("tiny") == 1);
    CHECK(m.addons.f10_files.at("tiny") == "addons/f10_tiny.f10bin");
    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// F10.7 ensemble: golden values from the reference implementation (Keras predict)
// ---------------------------------------------------------------------------

TEST_CASE("Addons: F10 ensemble reproduces the reference forecaster") {
    const fs::path fx = fs::path(ROPE_CPP_FIXTURE_DIR) / "addons";
    auto e = rope::forecast::F10Ensemble::load(fx / "f10_tiny.f10bin");
    CHECK(e.n_members() == 2);
    nlohmann::json ref;
    std::ifstream(fx / "f10_tiny_expected.json") >> ref;
    const auto hist = ref["history"].get<std::vector<double>>();
    const int  n    = ref["n_days"].get<int>();
    auto fc = e.forecast(hist, n);
    REQUIRE(static_cast<int>(fc.mean.size()) == n);
    for (int d = 0; d < n; ++d) {
        CHECK_THAT(fc.mean[d], WithinAbs(ref["mean"][d].get<double>(), 1e-3));   // Keras runs in float32
        CHECK_THAT(fc.std[d],  WithinAbs(ref["std"][d].get<double>(),  1e-3));
    }
    CHECK_THROWS_WITH(e.forecast(std::vector<double>(5, 150.0), 3), ContainsSubstring("history"));
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

TEST_CASE("Addons: DA seeds the rollout with the assimilated state") {
    const auto dir = unique_temp_dir("da_ic");
    auto cfg = base_cfg(model_with_addons(dir, 1e-4f), write_long_drivers(dir));
    cfg.compute_uncertainty = false;
    cfg.use_da = true;
    auto pipe = rope::forecast::load(cfg);
    std::vector<float> latent;
    pipe->run(START, HORIZON, &latent);
    REQUIRE(latent.size() == static_cast<std::size_t>((HORIZON + 1) * K));
    for (int k = 0; k < K; ++k)   // hour 0 = the last DA row (the start hour)
        CHECK_THAT(latent[k], WithinAbs(0.01 * (k - 4) + 0.004, 1e-6));
    auto info = nlohmann::json::parse(pipe->run_info_json());
    CHECK(info["data_assimilation"]["state_time"] == "2024-01-02T06:00:00");
    CHECK(info["data_assimilation"]["n_obs_span"] == 7);
    fs::remove_all(dir);
}

TEST_CASE("Addons: DA outside the table's coverage fails loudly") {
    const auto dir = unique_temp_dir("da_range");
    auto cfg = base_cfg(model_with_addons(dir, 1e-4f), write_long_drivers(dir));
    cfg.use_da = true;
    auto pipe = rope::forecast::load(cfg);
    CHECK_THROWS_WITH(pipe->run("2023-12-20 00:00:00", HORIZON), ContainsSubstring("DA table covers"));
    fs::remove_all(dir);
}

TEST_CASE("Addons: DA with zero spread adds no input uncertainty") {
    const auto dir = unique_temp_dir("da_zero");
    const auto model = model_with_addons(dir, 1e-4f);
    const auto drv   = write_long_drivers(dir);

    auto cfg_det = base_cfg(model, drv);
    cfg_det.use_da = true;
    cfg_det.compute_uncertainty = false;
    auto det = rope::forecast::load(cfg_det)->run(START, HORIZON);

    auto cfg0 = base_cfg(model, drv);
    cfg0.use_da = true;
    cfg0.da_ic_cov_scale = 0.0;            // all sigma points collapse onto the central run
    auto g0 = rope::forecast::load(cfg0)->run(START, HORIZON);

    auto cfg1 = base_cfg(model, drv);
    cfg1.use_da = true;
    cfg1.da_ic_cov_scale = 30.0;
    auto g1 = rope::forecast::load(cfg1)->run(START, HORIZON);

    // (The synthetic fixture networks are constant, so only the bookkeeping is observable here;
    //  sensitivity is checked against the Python reference with a real model, docs/addons.md.)
    REQUIRE(g0.density.size() == det.density.size());
    for (std::size_t i = 0; i < det.density.size(); ++i) {
        CHECK_THAT(g0.density[i], WithinAbs(det.density[i], 1e-6 * std::abs(det.density[i]) + 1e-30));
        CHECK(g1.uncertainty[i] + 1e-6f * g0.uncertainty[i] >= g0.uncertainty[i]);
    }
    fs::remove_all(dir);
}

TEST_CASE("Addons: F10 forecaster changes only the hours after the start") {
    const auto dir = unique_temp_dir("f10");
    const auto model = model_with_addons(dir, 1e-4f);
    const auto drv   = write_long_drivers(dir);

    auto cfg_off = base_cfg(model, drv);
    cfg_off.compute_uncertainty = false;
    auto off = rope::forecast::load(cfg_off)->run(START, HORIZON);

    auto cfg_on = cfg_off;
    cfg_on.use_f10_forecaster = true;
    auto pipe = rope::forecast::load(cfg_on);
    auto on = pipe->run(START, HORIZON);

    const std::size_t vox = static_cast<std::size_t>(on.shape.voxels());
    for (std::size_t v = 0; v < vox; ++v)                       // hour 0: same IC, same drivers
        CHECK(on.density[v] == off.density[v]);

    // The pipeline's daily forecast = the ensemble run on the 00 UT F10.7 of the preceding days.
    auto e = rope::forecast::F10Ensemble::load(model / "addons" / "f10_tiny.f10bin");
    auto db = rope::io::SpaceWeatherDB::from_file(drv);
    const rope::TimePoint day0 = rope::parse_datetime("2024-01-02 00:00:00");
    std::vector<double> hist;
    for (int i = e.history_days(); i >= 1; --i)
        hist.push_back(db.lookup(day0 - static_cast<rope::TimePoint>(i) * 86400).get("f10"));
    auto info = nlohmann::json::parse(pipe->run_info_json());
    const auto& daily = info["f10_forecaster"]["daily"];
    REQUIRE(daily.size() == 2);                                 // 06..12 UT on 01-02, + next-day anchor
    auto ref = e.forecast(hist, 2);
    for (int d = 0; d < 2; ++d) {
        CHECK_THAT(daily[d]["mean"].get<double>(), WithinAbs(ref.mean[d], 1e-9));
        CHECK_THAT(daily[d]["std"].get<double>(),  WithinAbs(ref.std[d],  1e-9));
    }
    CHECK(daily[0]["date"] == "2024-01-02");
    CHECK(info["f10_forecaster"]["ensemble"] == "tiny");
    CHECK(info["f10_forecaster"]["members"] == 2);
    CHECK_THAT(info["f10_forecaster"]["sigma_sfu"].get<double>(), WithinAbs(ref.std[0], 1e-9));   // 1 covered day
    fs::remove_all(dir);
}

TEST_CASE("Addons: both add-ons with uncertainty give 2(10+1)+1 sigma points") {
    const auto dir = unique_temp_dir("both");
    auto cfg = base_cfg(model_with_addons(dir, 1e-4f), write_long_drivers(dir));
    cfg.use_da = true;
    cfg.use_f10_forecaster = true;
    auto pipe = rope::forecast::load(cfg);
    auto g = pipe->run(START, HORIZON);
    for (float u : g.uncertainty) CHECK(u >= 0.0f);
    for (float d : g.density)     CHECK(d > 0.0f);
    auto info = nlohmann::json::parse(pipe->run_info_json());
    CHECK(info["uncertainty"]["input_dims"] == 11);
    CHECK(info["uncertainty"]["input_sigma_points"] == 23);
    fs::remove_all(dir);
}

TEST_CASE("Addons: add-ons off leaves the pipeline output unchanged") {
    const auto dir = unique_temp_dir("off");
    const auto drv = write_long_drivers(dir);
    auto plain  = rope::forecast::load(base_cfg(fs::path(ROPE_FIXTURE_MODELS), drv))->run(START, HORIZON);
    auto with   = rope::forecast::load(base_cfg(model_with_addons(dir, 1e-4f), drv));
    auto g      = with->run(START, HORIZON);
    CHECK(g.density == plain.density);
    CHECK(g.uncertainty == plain.uncertainty);
    CHECK(with->run_info_json().empty());
    fs::remove_all(dir);
}
