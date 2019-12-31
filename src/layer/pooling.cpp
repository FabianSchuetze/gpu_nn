#include "../../include/layer/pooling.h"
#include <float.h>
#include <sys/time.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "../../include/cuda_math.h"
#include "../../include/math.h"

Pooling::Pooling(Window window, Stride stride, ImageShape imageshape,
                 Channels channels)
    : Layer("Pooling"),
      _window(window),
      _stride(stride),
      _inp(imageshape),
      _channels(channels),
      _out(0, 0),
      batch_size(0) {
    initialize_masking();
    initialize_output_dimension();
}

Pooling::Pooling(Window window, Stride stride,
                 const std::shared_ptr<Layer>& previous)
    : Layer("Pooling"),
      _window(window),
      _stride(stride),
      _inp(0, 0),
      _channels(0),
      _out(0, 0),
      batch_size(0) {
    initialize_from_previous(previous);
    initialize_masking();
    initialize_output_dimension();
    _previous = previous;
}

void Pooling::initialize_from_previous(const std::shared_ptr<Layer>& previous) {
    if (previous->name() == "Convolution") {
        std::shared_ptr<Convolution> conv =
            std::dynamic_pointer_cast<Convolution>(previous);
        _inp = conv->_inp;
        _channels.get() = conv->_filters.get();
        _out = conv->_out;
    } else {
        throw std::runtime_error("Can only convert conv\n");
    }
}

void Pooling::initialize_output_dimension() {
    int out_height =
        static_cast<int>(ceil(static_cast<float>(_inp.first() - _window.get()) /
                              _stride.get())) +
        1;
    int out_width = static_cast<int>(
                        ceil(static_cast<float>(_inp.second() - _window.get()) /
                             _stride.get())) +
                    1;
    _out = ImageShape(out_height, out_width);
    _out_dim[0] = _channels.get();
    _out_dim.push_back(out_height);
    _out_dim.push_back(out_width);
}

void Pooling::initialize_masking() {
    mask = std::make_shared<Storage>();
    mask2 = std::make_shared<Storage>();
}

void inline Pooling::check_input_size(const SharedStorage& in) {
    int rows = _channels.get() * _inp.first() * _inp.second();
    if ((rows != in->get_rows()) or (batch_size != in->get_cols())) {
        std::stringstream ss;
        ss << "Dimension do not fit, in:\n"
           << __PRETTY_FUNCTION__ << "\ncalled from " << __FILE__ << " at "
           << __LINE__;
        throw std::invalid_argument(ss.str());
    }
}

void Pooling::check_masking(const SharedStorage& out) {
    if (!same_size(out, mask)) {
        mask = std::make_shared<Storage>(
            Matrix::Zero(out->get_rows(), out->get_cols()));
        mask2 = std::make_shared<Storage>(
            Matrix::Zero(out->get_rows(), out->get_cols()));
        batch_size = out->get_cols();
    }
}

void dump_file2(const dtype* val, int size, const char* name) {
    std::ofstream file(name);
    for (int i = 0; i < size; ++i) {
        file << val[i];
        file << " ";
    }
}
void Pooling::forward_gpu(const std::shared_ptr<Storage>& in,
                          std::shared_ptr<Storage>& out, const std::string&) {
    check_masking(out);
    check_input_size(in);
    pooling_gpu(in->gpu_pointer_const(), _window.get(), _stride.get(),
                _inp.first(), _inp.second(), _channels.get(), _out.first(),
                _out.second(), batch_size, out->gpu_pointer(),
                mask->gpu_pointer());
}

void Pooling::forward_cpu(const std::shared_ptr<Storage>& in,
                          std::shared_ptr<Storage>& out, const std::string&) {
    check_masking(in);
    check_input_size(in);
    dtype min = -FLT_MAX;
    //out->return_data().fill(min);
    out->update_cpu_data(min);
    pooling_cpu(in->cpu_pointer_const(), _window.get(), _stride.get(),
                _inp.first(), _inp.second(), _channels.get(), _out.first(),
                _out.second(), batch_size, out->cpu_pointer(),
                mask->cpu_pointer());
}

void Pooling::backward_gpu(const SharedStorage&,
                           const SharedStorage& gradient_in,
                           SharedStorage& gradient_out) {
    pooling_backward_gpu(gradient_in->gpu_pointer_const(),
                         mask->gpu_pointer_const(), _window.get(),
                         _stride.get(), _inp.first(), _inp.second(),
                         _channels.get(), _out.first(), _out.second(),
                         batch_size, gradient_out->gpu_pointer());
    //int size = gradient_out->get_rows() * gradient_out->get_cols();
    //dump_file2(gradient_out->cpu_pointer(), size, "grad_out.txt");
};

void Pooling::backward_cpu(const SharedStorage&,
                           const SharedStorage& gradient_in,
                           SharedStorage& gradient_out) {
    pooling_backward_cpu(gradient_in->cpu_pointer_const(),
                         mask->cpu_pointer_const(), _window.get(),
                         _stride.get(), _inp.first(), _inp.second(),
                         _channels.get(), _out.first(), _out.second(),
                         batch_size, gradient_out->cpu_pointer());
};
