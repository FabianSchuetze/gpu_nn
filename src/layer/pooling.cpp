#include "../../include/layer/pooling.h"
#include <iostream>
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

void inline Pooling::check_input_size(const SharedStorage& in) {
    int rows = _channels.get() * _inp.first() * _inp.second();
    if ((rows != in->get_rows()) or (batch_size != in->get_cols()))  {
            std::stringstream ss;
            ss << "Dimension do not fit, in:\n"
               << __PRETTY_FUNCTION__ << "\ncalled from "
               << __FILE__ << " at " << __LINE__;
            throw std::invalid_argument(ss.str());
    }
}

void Pooling::check_masking(const SharedStorage& in) {
    if (!same_size(in, mask)) {
        mask = std::make_shared<Storage>(
            Matrix::Zero(in->get_rows(), in->get_cols()));
        batch_size = in->get_cols();
    }
}

void Pooling::forward_gpu(const std::shared_ptr<Storage>& in,
                          std::shared_ptr<Storage>& out, const std::string&) {
    check_masking(in);
    check_input_size(in);
    pooling_gpu(in->gpu_pointer_const(), _window.get(), _stride.get(),
                _inp.first(), _inp.second(), _channels.get(), batch_size,
                out->gpu_pointer(), mask->gpu_pointer());
}

void Pooling::forward_cpu(const std::shared_ptr<Storage>& in,
                          std::shared_ptr<Storage>& out, const std::string&) {
    check_masking(in);
    check_input_size(in);
    pooling_cpu(in->cpu_pointer_const(), _window.get(), _stride.get(),
                _inp.first(), _inp.second(), _channels.get(), batch_size,
                out->cpu_pointer(), mask->cpu_pointer());
}
void Pooling::backward_gpu(const SharedStorage&,
                           const SharedStorage& gradient_in,
                           SharedStorage& gradient_out) {
    std::cout << "pooling backwards\n" << std::endl;
    pooling_backward_gpu2(
        gradient_in->gpu_pointer_const(), mask->gpu_pointer_const(),
        _window.get(), _stride.get(), _inp.first(), _inp.second(),
        _channels.get(), batch_size, gradient_out->gpu_pointer());
};

void Pooling::backward_cpu(const SharedStorage&,
                           const SharedStorage& gradient_in,
                           SharedStorage& gradient_out) {
    pooling_backward_cpu(
        gradient_in->cpu_pointer_const(), mask->cpu_pointer_const(),
        _window.get(), _stride.get(), _inp.first(), _inp.second(),
        _channels.get(), batch_size, gradient_out->cpu_pointer());
};
