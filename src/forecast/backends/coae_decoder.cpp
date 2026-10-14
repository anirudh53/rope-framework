#include "coae_decoder.h"

#include "grid_stitch.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace rope::forecast {

CoaeDecoder::CoaeDecoder(
    const std::filesystem::path& dir,
    const io::ModelManifest&     manifest,
    const Config&                cfg)
    : grid_shape_(manifest.grid)
{
    auto log = cfg.log ? cfg.log : [](std::string_view){};

    int dec_threads = cfg.intra_threads_decoder;
    if (dec_threads <= 0)
        dec_threads = static_cast<int>(std::thread::hardware_concurrency());

    const int decode_batch = (cfg.decode_batch_size > 0)
        ? std::min(cfg.decode_batch_size, manifest.decode_batch_size)
        : manifest.decode_batch_size;

    log("Loading " + std::to_string(manifest.decoder_stages.size()) + " decoder stage(s)\xe2\x80\xa6"
        "  decode_batch_size=" + std::to_string(decode_batch));
    stages_.reserve(manifest.decoder_stages.size());

    for (const auto& d : manifest.decoder_stages) {
#ifdef ROPE_USE_LIBTORCH
        std::string  dec_key    = "libtorch";
        ModelBackend dec_be     = ModelBackend::LibTorch;
        std::string  dec_device = cfg.decoder_device;
        if (d.backends.find(dec_key) == d.backends.end()) {
            dec_key    = "onnx";
            dec_be     = ModelBackend::ONNX;
            dec_device = "cpu";
        }
#else
        std::string  dec_key    = "onnx";
        ModelBackend dec_be     = ModelBackend::ONNX;
#  ifdef ROPE_ORT_CUDA
        std::string  dec_device = cfg.decoder_device;   // CUDA builds: ORT CUDA EP (see onnx_model.h)
#  else
        std::string  dec_device = "cpu";
#  endif
#endif
        auto it = d.backends.find(dec_key);
        if (it == d.backends.end())
            throw std::runtime_error(
                "CoaeDecoder: decoder stage '" + d.stats +
                "': no usable backend key in manifest (tried '" + dec_key + "')");

#ifdef ROPE_USE_LIBTORCH
        log("  stage [" + std::to_string(d.alt_start) + ", " +
            std::to_string(d.alt_end) + "): backend=libtorch  device=" + dec_device);
#else
        log("  stage [" + std::to_string(d.alt_start) + ", " +
            std::to_string(d.alt_end) + "): backend=onnx  device=" + dec_device + "  threads=" +
            std::to_string(dec_threads));
#endif

        io::Stats stats_cae = io::Stats::load(dir / d.stats);

        auto& stage     = stages_.emplace_back();
        stage.alt_start = d.alt_start;
        stage.alt_end   = d.alt_end;
        stage.model     = make_model((dir / it->second).string(),
                                     dec_be, dec_threads, /*inter_op_threads=*/1,
                                     false, dec_device);
        stage.denorm    = std::make_unique<io::CoaeDenormalizer>(stats_cae);
        stage.decoder   = std::make_unique<LatentDecoder>(
            *stage.model, *stage.denorm, decode_batch,
            d.alt_end - d.alt_start, grid_shape_.n_lst, grid_shape_.n_lat);
    }
}

std::vector<float> CoaeDecoder::decode(std::span<const float> latents, int H, int K) {
    const int voxels = grid_shape_.voxels();
    std::vector<float> density(static_cast<std::size_t>(H) * voxels, 0.0f);
    for (auto& stage : stages_) {
        auto stage_dens = stage.decoder->decode(latents, H, K);
        stitch_altitude_range(density.data(), stage_dens.data(), H,
                              stage.alt_start, stage.alt_end, grid_shape_);
    }
    return density;
}

} // namespace rope::forecast
