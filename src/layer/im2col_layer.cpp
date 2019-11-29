#include "../../include/layer/im2col_layer.h"
#include <iostream>
#include "../../include/cuda_math.h"
#include "../../include/math.h"
Im2ColLayer::Im2ColLayer(FilterShape filtershape, Pad pad, Stride stride,
                         ImageShape imageshape, Channels channels)
    : Layer("Im2ColLayer"),
      _kernel(filtershape),
      _pad(pad),
      _stride(stride),
      _inp(imageshape),
      _out(0, 0),
      _channels(channels) {
    output_shape();
};

int Im2ColLayer::output_dimension() {
    return _out.first() * _out.second();
}

int Im2ColLayer::output_dimension() const {
    return  _out.first() * _out.second();
}

int Im2ColLayer::n_cols() {
    return _channels.get() * _kernel.first() * _kernel.second();
}
int Im2ColLayer::n_cols() const {
    return _channels.get() * _kernel.first() * _kernel.second();
}

void Im2ColLayer::output_shape() {
    int out_height =
        (_inp.first() + 2 * _pad.get() - _kernel.first()) / _stride.get() + 1;
    int out_width =
        (_inp.second() + 2 * _pad.get() - _kernel.second()) / _stride.get() + 1;
    _out = ImageShape(out_height, out_width);
}

void Im2ColLayer::check_size(const SharedStorage& in,
                             const SharedStorage& out) {
    int col_rows = _out.first() * _out.second();
    int col_cols =
        _kernel.first() * _kernel.second() * _channels.get() * in->get_cols();
    if (col_rows != out->get_rows()) {
        std::stringstream ss;
        ss << "Dimension do not fit, in:\n"
           << __PRETTY_FUNCTION__ << "\n expected rows: " << col_rows
           << " received rows " << out->get_rows() << "\ncalled from "
           << __FILE__ << " at " << __LINE__;
        throw std::invalid_argument(ss.str());
    }
    if (col_cols != out->get_cols()) {
        std::stringstream ss;
        ss << "Dimension do not fit, in:\n"
           << __PRETTY_FUNCTION__ << "\n expected cols: " << col_cols
           << " received cols " << out->get_cols() << "\ncalled from "
           << __FILE__ << " at " << __LINE__;
        throw std::invalid_argument(ss.str());
    }
}

void Im2ColLayer::advance_pointers_forward(const float*& input, float*& column,
                                           int input_rows) {
    input += (input_rows);
    column += (_out.first() * _out.second() * _kernel.first() *
               _kernel.second() * _channels.get());
}
void Im2ColLayer::forward_gpu(const SharedStorage& in, SharedStorage& out,
                              const std::string&) {
    check_size(in, out);
    const float* inpp = in->gpu_pointer_const();
    float* colp = out->gpu_pointer();
    for (int n = 0; n < in->get_cols(); n++) {
        im2col_gpu(inpp, _channels.get(), _inp.first(), _inp.second(),
                   _kernel.first(), _kernel.second(), _pad.get(), _stride.get(),
                   colp);
        advance_pointers_forward(inpp, colp, in->get_rows());
    }
}

void Im2ColLayer::forward_cpu(const SharedStorage& in, SharedStorage& out,
                              const std::string&) {
    check_size(in, out);
    const float* inpp = in->cpu_pointer_const();
    float* colp = out->cpu_pointer();
    for (int n = 0; n < in->get_cols(); n++) {
        im2col_cpu(inpp, _channels.get(), _inp.first(), _inp.second(),
                   _kernel.first(), _kernel.second(), _pad.get(), _stride.get(),
                   colp);
        advance_pointers_forward(inpp, colp, in->get_rows());
    }
}
void Im2ColLayer::advance_pointers_backward(const float*& input,
                                            float*& output) {
    input += _out.first() * _out.second() * _kernel.first() * _kernel.second() *
             _channels.get();
    output += _inp.first() * _inp.second() * _channels.get();
}
void Im2ColLayer::backward_cpu(const SharedStorage&,
                               const SharedStorage& gradient_in,
                               SharedStorage& gradient_out) {
    const float* colp = gradient_in->cpu_pointer_const();
    float* imgp = gradient_out->cpu_pointer();
    for (int n = 0; n < gradient_out->get_cols(); n++) {
        col2im_cpu(colp, _channels.get(), _inp.first(), _inp.second(),
                   _kernel.first(), _kernel.second(), _pad.get(), _stride.get(),
                   imgp);
        advance_pointers_backward(colp, imgp);
    }
}

void Im2ColLayer::backward_gpu(const SharedStorage&,
                               const SharedStorage& gradient_in,
                               SharedStorage& gradient_out) {
    const float* colp = gradient_in->gpu_pointer_const();
    float* imgp = gradient_out->gpu_pointer();
    for (int n = 0; n < gradient_out->get_cols(); n++) {
        col2im_gpu(colp, _channels.get(), _inp.first(), _inp.second(),
                   _kernel.first(), _kernel.second(), _pad.get(), _stride.get(),
                   imgp);
        advance_pointers_backward(colp, imgp);
    }
}
