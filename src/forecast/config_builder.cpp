#include "rope/forecast/pipeline.h"

namespace rope::forecast {

Config config_from_reader(const io::ConfigReader& config,
                          const std::filesystem::path& config_dir)
{
    auto resolve = [&](const std::string& p) -> std::filesystem::path {
        std::filesystem::path fp{p};
        return fp.is_absolute() ? fp : config_dir / fp;
    };

    Config fcfg;
    fcfg.exported_dir        = resolve(config.get("paths.exported_dir"));
    if (config.has("paths.driver_path"))
        fcfg.driver_path = resolve(config.get("paths.driver_path"));
    if (config.has("driver_cache.cache_dir"))
        fcfg.cache_dir = resolve(config.get("driver_cache.cache_dir"));
    fcfg.cache_max_age_hours   = config.get_int("driver_cache.max_age_hours", 24);
    fcfg.intra_threads_base    = config.get_int("threads.intra_threads_base", 1);
    fcfg.intra_threads_meta    = config.get_int("threads.intra_threads_meta", 0);
    fcfg.intra_threads_decoder = config.get_int("threads.intra_threads_decoder", 0);
    fcfg.decoder_device        = config.get("decoder.device", "cpu");
    fcfg.compute_uncertainty   = config.get("forecast.compute_uncertainty", "true") == "true";
    fcfg.decode_chunk_hours    = config.get_int("forecast.decode_chunk_hours", 72);
    fcfg.decode_batch_size     = config.get_int("forecast.decode_batch_size", 0);
    // Fallback is 0, not rope.conf.in's shipped 48; a config that omits this key keeps unchanged behavior.
    fcfg.warmup_time_hours     = config.get_int("forecast.warmup_time_hours", 0);

    // Add-ons: absent sections/keys keep them off.
    fcfg.use_da           = config.get("data_assimilation.enabled", "false") == "true";
    if (config.has("data_assimilation.states_path"))
        fcfg.da_states_path = resolve(config.get("data_assimilation.states_path"));
    fcfg.da_ic_cov_mode   = config.get("data_assimilation.ic_cov_mode", "hour_avg");
    fcfg.da_ic_cov_scale  = config.get_double("data_assimilation.ic_cov_scale", 1.0);
    fcfg.use_f10_forecaster = config.get("f10_forecaster.enabled", "false") == "true";
    fcfg.f10_ensemble     = config.get("f10_forecaster.ensemble", "");
    if (config.has("f10_forecaster.ensemble_path"))
        fcfg.f10_ensemble_path = resolve(config.get("f10_forecaster.ensemble_path"));
    fcfg.f10_sigma        = config.get_double("f10_forecaster.sigma", 0.0);

    return fcfg;
}

} // namespace rope::forecast
