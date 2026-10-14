#include "model_interface.h"
#include "onnx_model.h"
#ifdef ROPE_USE_LIBTORCH
#  include "libtorch_model.h"
#endif

#include <memory>
#include <stdexcept>
#include <string>

namespace rope::forecast {

std::unique_ptr<IModel> make_model(
    const std::string& path,
    ModelBackend       backend,
    int                intra_op_threads,
    int                inter_op_threads,
    bool               use_dnnl,
    const std::string& device)
{
    switch (backend) {
    case ModelBackend::ONNX:
        return std::make_unique<OnnxModel>(
            path, intra_op_threads, inter_op_threads, use_dnnl, device);
#ifdef ROPE_USE_LIBTORCH
    case ModelBackend::LibTorch:
        return std::make_unique<LibTorchModel>(path, device, intra_op_threads);
#endif
    }
    throw std::runtime_error("make_model: unknown backend");
}

} // namespace rope::forecast
