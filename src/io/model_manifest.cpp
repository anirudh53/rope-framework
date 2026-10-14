#include "rope/io/model_manifest.h"
#include "rope/core/types.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace rope::io {

static std::string missing_field(const char* field, const std::string& path) {
    return std::string("ModelManifest::load: missing required field '") +
           field + "' in " + path;
}

ModelManifest ModelManifest::load(const std::filesystem::path& exported_dir) {
    auto path = exported_dir / "model_manifest.json";

    if (!std::filesystem::exists(path))
        throw std::runtime_error(
            "ModelManifest::load: no model_manifest.json in " +
            exported_dir.string());

    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error(
            "ModelManifest::load: cannot open " + path.string());
    std::ostringstream raw_stream;
    raw_stream << f.rdbuf();
    const std::string raw = raw_stream.str();

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(raw);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(
            "ModelManifest::load: malformed JSON in " +
            path.string() + ": " + e.what());
    }

    ModelManifest m;
    const std::string ps = path.string();

    // schema_version
    if (!j.contains("schema_version") || !j["schema_version"].is_number_integer())
        throw std::runtime_error(missing_field("schema_version", ps));
    m.schema_version = j["schema_version"].get<int>();
    if (m.schema_version != 1)
        throw std::runtime_error(
            "ModelManifest::load: unsupported schema_version " +
            std::to_string(m.schema_version) + " in " + ps);

    // kind
    if (!j.contains("kind") || !j["kind"].is_string())
        throw std::runtime_error(missing_field("kind", ps));
    m.kind = j["kind"].get<std::string>();
    if (m.kind != "stacked_ensemble")
        throw std::runtime_error(
            "ModelManifest::load: unsupported kind '" + m.kind + "' in " + ps);

    // runtime_requirements (object with optional keys)
    if (j.contains("runtime_requirements") && j["runtime_requirements"].is_object()) {
        const auto& rr = j["runtime_requirements"];
        if (rr.contains("onnxruntime") && rr["onnxruntime"].is_string())
            m.runtime_requirements.onnxruntime = rr["onnxruntime"].get<std::string>();
        if (rr.contains("libtorch") && rr["libtorch"].is_string())
            m.runtime_requirements.libtorch = rr["libtorch"].get<std::string>();
    }

    // latent_dim
    if (!j.contains("latent_dim") || !j["latent_dim"].is_number_integer())
        throw std::runtime_error(missing_field("latent_dim", ps));
    m.latent_dim = j["latent_dim"].get<int>();
    if (m.latent_dim <= 0)
        throw std::runtime_error(
            "ModelManifest::load: 'latent_dim' must be positive in " + ps);

    // drivers
    if (!j.contains("drivers") || !j["drivers"].is_object())
        throw std::runtime_error(missing_field("drivers", ps));
    const auto& jdrv = j["drivers"];

    if (!jdrv.contains("source") || !jdrv["source"].is_string())
        throw std::runtime_error(missing_field("drivers.source", ps));
    m.driver_source = jdrv["source"].get<std::string>();
    if (m.driver_source.empty())
        throw std::runtime_error(
            "ModelManifest::load: 'drivers.source' must not be empty in " + ps);

    if (!jdrv.contains("columns") || !jdrv["columns"].is_array() || jdrv["columns"].empty())
        throw std::runtime_error(missing_field("drivers.columns", ps));
    for (const auto& jc : jdrv["columns"]) {
        if (!jc.contains("name") || !jc["name"].is_string() || jc["name"].get<std::string>().empty())
            throw std::runtime_error(
                "ModelManifest::load: drivers.columns entry missing 'name' in " + ps);
        if (!jc.contains("description") || !jc["description"].is_string() ||
                jc["description"].get<std::string>().empty())
            throw std::runtime_error(
                "ModelManifest::load: drivers.columns entry missing 'description' in " + ps);
        DriverColumnInfo info{jc["name"].get<std::string>(), jc["description"].get<std::string>()};
        m.driver_columns.push_back(info.name);
        m.driver_column_info.push_back(std::move(info));
    }

    // validated (lenient — defaults to false if absent)
    if (j.contains("validated") && j["validated"].is_boolean())
        m.validated = j["validated"].get<bool>();

    // grid — kind-agnostic physical shape of this model's output grid.
    if (!j.contains("grid") || !j["grid"].is_object())
        throw std::runtime_error(missing_field("grid", ps));
    {
        const auto& jg = j["grid"];
        for (const char* f : {"n_lst", "n_lat", "n_alt"})
            if (!jg.contains(f) || !jg[f].is_number_integer())
                throw std::runtime_error(missing_field((std::string("grid.") + f).c_str(), ps));
        for (const char* f : {"lat_min_deg", "lat_max_deg", "alt_min_km", "alt_max_km"})
            if (!jg.contains(f) || !jg[f].is_number())
                throw std::runtime_error(missing_field((std::string("grid.") + f).c_str(), ps));

        m.grid.n_lst       = jg["n_lst"].get<int>();
        m.grid.n_lat       = jg["n_lat"].get<int>();
        m.grid.n_alt       = jg["n_alt"].get<int>();
        m.grid.lat_min_deg = jg["lat_min_deg"].get<double>();
        m.grid.lat_max_deg = jg["lat_max_deg"].get<double>();
        m.grid.alt_min_km  = jg["alt_min_km"].get<double>();
        m.grid.alt_max_km  = jg["alt_max_km"].get<double>();

        if (m.grid.n_lst <= 0 || m.grid.n_lat <= 1 || m.grid.n_alt <= 1)
            throw std::runtime_error(
                "ModelManifest::load: 'grid' counts must be positive, with n_lat/n_alt >= 2, in " + ps);
        if (m.grid.lat_min_deg >= m.grid.lat_max_deg)
            throw std::runtime_error(
                "ModelManifest::load: 'grid.lat_min_deg' must be < 'grid.lat_max_deg' in " + ps);
        if (m.grid.alt_min_km >= m.grid.alt_max_km)
            throw std::runtime_error(
                "ModelManifest::load: 'grid.alt_min_km' must be < 'grid.alt_max_km' in " + ps);
    }

    // ic
    if (!j.contains("ic") || !j["ic"].is_object())
        throw std::runtime_error(missing_field("ic", ps));
    const auto& jic = j["ic"];

    if (!jic.contains("kind") || !jic["kind"].is_string())
        throw std::runtime_error(missing_field("ic.kind", ps));
    m.ic_kind = jic["kind"].get<std::string>();
    if (m.ic_kind.empty())
        throw std::runtime_error("ModelManifest::load: 'ic.kind' must not be empty in " + ps);

    if (!jic.contains("params") || !jic["params"].is_object())
        throw std::runtime_error(missing_field("ic.params", ps));
    const auto& jic_params = jic["params"];
    if (!jic_params.contains("grid_axes") || !jic_params["grid_axes"].is_array())
        throw std::runtime_error(missing_field("ic.params.grid_axes", ps));
    m.ic_grid_axes = jic_params["grid_axes"].get<std::vector<std::string>>();
    if (m.ic_grid_axes.size() != 2)
        throw std::runtime_error(
            "ModelManifest::load: 'ic.params.grid_axes' must have exactly 2 entries "
            "(ic_lookup_table is a 2D interpolator), got " +
            std::to_string(m.ic_grid_axes.size()) + " in " + ps);

    // decoder — kind-agnostic, sibling to ic/drivers/grid, like ic's own kind/params split.
    if (!j.contains("decoder") || !j["decoder"].is_object())
        throw std::runtime_error(missing_field("decoder", ps));
    const auto& jdec = j["decoder"];

    if (!jdec.contains("kind") || !jdec["kind"].is_string())
        throw std::runtime_error(missing_field("decoder.kind", ps));
    m.decoder_kind = jdec["kind"].get<std::string>();
    if (m.decoder_kind.empty())
        throw std::runtime_error("ModelManifest::load: 'decoder.kind' must not be empty in " + ps);

    if (!jdec.contains("params") || !jdec["params"].is_object())
        throw std::runtime_error(missing_field("decoder.params", ps));
    const auto& jdec_params = jdec["params"];

    if (!jdec_params.contains("decode_batch_size") || !jdec_params["decode_batch_size"].is_number_integer())
        throw std::runtime_error(missing_field("decoder.params.decode_batch_size", ps));
    m.decode_batch_size = jdec_params["decode_batch_size"].get<int>();
    if (m.decode_batch_size <= 0)
        throw std::runtime_error(
            "ModelManifest::load: 'decoder.params.decode_batch_size' must be positive in " + ps);

    if (!jdec_params.contains("stages") || !jdec_params["stages"].is_array() ||
            jdec_params["stages"].empty())
        throw std::runtime_error(missing_field("decoder.params.stages", ps));
    for (const auto& jd : jdec_params["stages"]) {
        DecoderStageSpec ds;
        if (!jd.contains("backends") || !jd["backends"].is_object())
            throw std::runtime_error(
                "ModelManifest::load: decoder stage missing 'backends' in " + ps);
        for (const auto& [key, val] : jd["backends"].items()) {
            if (!val.is_string())
                throw std::runtime_error(
                    "ModelManifest::load: decoder backends values must be strings in " + ps);
            ds.backends[key] = val.get<std::string>();
        }
        if (ds.backends.empty())
            throw std::runtime_error(
                "ModelManifest::load: decoder 'backends' must not be empty in " + ps);
        if (!jd.contains("stats") || !jd["stats"].is_string())
            throw std::runtime_error(
                "ModelManifest::load: decoder stage missing 'stats' in " + ps);
        ds.stats = jd["stats"].get<std::string>();
        if (!jd.contains("alt_start") || !jd["alt_start"].is_number_integer())
            throw std::runtime_error(
                "ModelManifest::load: decoder stage missing 'alt_start' in " + ps);
        ds.alt_start = jd["alt_start"].get<int>();
        if (!jd.contains("alt_end") || !jd["alt_end"].is_number_integer())
            throw std::runtime_error(
                "ModelManifest::load: decoder stage missing 'alt_end' in " + ps);
        ds.alt_end = jd["alt_end"].get<int>();
        m.decoder_stages.push_back(std::move(ds));
    }

    // Validate decoder altitude ranges: sort by alt_start, verify exact tiling of [0, grid.n_alt).
    std::sort(m.decoder_stages.begin(), m.decoder_stages.end(),
        [](const DecoderStageSpec& a, const DecoderStageSpec& b) {
            return a.alt_start < b.alt_start;
        });
    if (m.decoder_stages.front().alt_start != 0)
        throw std::runtime_error(
            "ModelManifest::load: first decoder must have alt_start=0 in " + ps);
    for (std::size_t i = 1; i < m.decoder_stages.size(); ++i) {
        if (m.decoder_stages[i].alt_start != m.decoder_stages[i - 1].alt_end)
            throw std::runtime_error(
                "ModelManifest::load: decoder altitude ranges have a gap or overlap in " + ps);
    }
    if (m.decoder_stages.back().alt_end != m.grid.n_alt)
        throw std::runtime_error(
            "ModelManifest::load: last decoder must have alt_end=" +
            std::to_string(m.grid.n_alt) + " in " + ps);

    // Kind-specific block
    if (!j.contains("stacked_ensemble") || !j["stacked_ensemble"].is_object())
        throw std::runtime_error(missing_field("stacked_ensemble", ps));

    const auto& jk = j["stacked_ensemble"];
    StackedEnsembleSpec spec;

    if (!jk.contains("seq_len") || !jk["seq_len"].is_number_integer())
        throw std::runtime_error(missing_field("stacked_ensemble.seq_len", ps));
    spec.seq_len = jk["seq_len"].get<int>();
    if (spec.seq_len <= 0)
        throw std::runtime_error(
            "ModelManifest::load: 'seq_len' must be positive in " + ps);

    // base_models
    if (!jk.contains("base_models") || !jk["base_models"].is_array() ||
            jk["base_models"].empty())
        throw std::runtime_error(missing_field("stacked_ensemble.base_models", ps));
    for (const auto& jbm : jk["base_models"]) {
        BaseModelSpec bm;
        if (!jbm.contains("file") || !jbm["file"].is_string())
            throw std::runtime_error(
                "ModelManifest::load: base_model entry missing 'file' in " + ps);
        bm.file = jbm["file"].get<std::string>();
        if (!jbm.contains("backend") || !jbm["backend"].is_string())
            throw std::runtime_error(
                "ModelManifest::load: base_model entry missing 'backend' in " + ps);
        bm.backend = jbm["backend"].get<std::string>();
        bm.architecture    = jbm.value("architecture", std::string{});
        bm.inter_op_threads = jbm.value("inter_op_threads", 1);
        spec.base_models.push_back(std::move(bm));
    }

    // meta_model
    if (!jk.contains("meta_model") || !jk["meta_model"].is_object())
        throw std::runtime_error(missing_field("stacked_ensemble.meta_model", ps));
    {
        const auto& jmm = jk["meta_model"];
        if (!jmm.contains("file") || !jmm["file"].is_string())
            throw std::runtime_error(
                "ModelManifest::load: meta_model missing 'file' in " + ps);
        spec.meta_model.file = jmm["file"].get<std::string>();
        if (!jmm.contains("backend") || !jmm["backend"].is_string())
            throw std::runtime_error(
                "ModelManifest::load: meta_model missing 'backend' in " + ps);
        spec.meta_model.backend = jmm["backend"].get<std::string>();
    }

    m.stacked_ensemble = std::move(spec);

    // Optional add-ons (forward-compatible: older builds ignore this block entirely).
    if (j.contains("addons") && j["addons"].is_object()) {
        const auto& ja = j["addons"];
        if (ja.contains("data_assimilation") && ja["data_assimilation"].is_object()) {
            const auto& jd = ja["data_assimilation"];
            if (!jd.contains("file") || !jd["file"].is_string())
                throw std::runtime_error(
                    "ModelManifest::load: addons.data_assimilation missing 'file' in " + ps);
            m.addons.da_file = jd["file"].get<std::string>();
        }
        if (ja.contains("f10_forecaster") && ja["f10_forecaster"].is_object()) {
            const auto& jf = ja["f10_forecaster"];
            if (jf.contains("ensembles") && jf["ensembles"].is_object())
                for (const auto& [name, je] : jf["ensembles"].items()) {
                    if (!je.is_object() || !je.contains("file") || !je["file"].is_string())
                        throw std::runtime_error(
                            "ModelManifest::load: addons.f10_forecaster.ensembles." + name +
                            " missing 'file' in " + ps);
                    m.addons.f10_files[name] = je["file"].get<std::string>();
                }
            m.addons.f10_default = jf.value("default", std::string{});
            if (!m.addons.f10_default.empty() && !m.addons.f10_files.count(m.addons.f10_default))
                throw std::runtime_error(
                    "ModelManifest::load: addons.f10_forecaster.default '" + m.addons.f10_default +
                    "' is not one of its ensembles in " + ps);
        }
    }

    // Cross-check: every backend referenced must have a version in runtime_requirements.
    std::set<std::string> used_backends;
    for (const auto& bm : m.stacked_ensemble->base_models)
        used_backends.insert(bm.backend);
    used_backends.insert(m.stacked_ensemble->meta_model.backend);
    for (const auto& ds : m.decoder_stages)
        for (const auto& [b, _] : ds.backends)
            used_backends.insert(b);

    for (const auto& b : used_backends) {
        if (b == "onnx" && m.runtime_requirements.onnxruntime.empty())
            throw std::runtime_error(
                "ModelManifest::load: backend 'onnx' is used but "
                "runtime_requirements.onnxruntime is not set in " + ps);
        if (b == "libtorch" && m.runtime_requirements.libtorch.empty())
            throw std::runtime_error(
                "ModelManifest::load: backend 'libtorch' is used but "
                "runtime_requirements.libtorch is not set in " + ps);
    }

    return m;
}

std::string ModelManifest::to_summary_json() const {
    nlohmann::json j;
    j["kind"]       = kind;
    j["latent_dim"] = latent_dim;
    j["validated"]  = validated;
    j["grid"] = {
        {"n_lst", grid.n_lst}, {"n_lat", grid.n_lat}, {"n_alt", grid.n_alt},
        {"lat_min_deg", grid.lat_min_deg}, {"lat_max_deg", grid.lat_max_deg},
        {"alt_min_km", grid.alt_min_km}, {"alt_max_km", grid.alt_max_km}
    };
    j["ic"] = {
        {"kind", ic_kind},
        {"axes", ic_grid_axes}
    };
    if (!addons.da_file.empty() || !addons.f10_files.empty()) {
        auto& ja = j["addons"] = nlohmann::json::object();
        if (!addons.da_file.empty())
            ja["data_assimilation"] = {{"file", addons.da_file}};
        if (!addons.f10_files.empty()) {
            auto names = nlohmann::json::array();
            for (const auto& [name, _] : addons.f10_files) names.push_back(name);
            ja["f10_forecaster"] = {{"default", addons.f10_default}, {"ensembles", names}};
        }
    }
    j["drivers"]["source"] = driver_source;
    auto& jcols = j["drivers"]["columns"] = nlohmann::json::array();
    for (const auto& col : driver_column_info)
        jcols.push_back({{"name", col.name}, {"description", col.description}});

    return j.dump();
}

} // namespace rope::io
