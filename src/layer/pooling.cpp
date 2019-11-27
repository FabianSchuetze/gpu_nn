#include "../../include/layer/pooling.h"
#include "../../include/cuda_math.h"
#include "../../include/math.h"
Pooling::Pooling(Window window, Stride stride, ImageShape imageshape,
                 Channels channels)
    : Layer("Pooling"),
      _window(window),
      _stride(stride),
      _inp(imageshape),
      _channels(channels),
      batch_size(0) {
    initialize_masking();
}

void Pooling::initialize_masking() { mask = std::make_shared<Storage>(); }

void Pooling::check_masking(const SharedStorage& in) {
    if (!same_size(in, mask)) {
        mask = std::make_shared<Storage>(
            Matrix::Zero(in->get_rows(), in->get_cols()));
    }
}

void Pooling::forward_gpu(const std::shared_ptr<Storage>& in,
                          std::shared_ptr<Storage>& out, const std::string&) {
    check_masking(in);
    pooling_gpu(in->gpu_pointer_const(), _window.get(), _stride.get(),
                _inp.first(), _inp.second(), _channels.get(), batch_size,
                out->gpu_pointer(), mask->gpu_pointer());
}

void Pooling::forward_cpu(const std::shared_ptr<Storage>& in,
                          std::shared_ptr<Storage>& out, const std::string&) {
    check_masking(in);
    pooling_cpu(in->gpu_pointer_const(), _window.get(), _stride.get(),
                _inp.first(), _inp.second(), _channels.get(), batch_size,
                out->gpu_pointer(), mask->gpu_pointer());
}
void Pooling::backward_gpu(const SharedStorage&, const SharedStorage&,
                           SharedStorage&) {
    ;
};
void Pooling::backward_cpu(const SharedStorage&, const SharedStorage&,
                           SharedStorage&) {
    ;
};
