#include "../../include/layer/im2col_layer.h"
#include <iostream>
#include <memory>
#include "../../include/cuda_math.h"
#include "../../include/math.h"

Im2ColLayer::Im2ColLayer(const std::shared_ptr<Convolution>& convolution_layer)
    : Layer("Im2ColLayer"),
      _kernel(convolution_layer->_kernel),
      _pad(convolution_layer->_pad),
      _stride(convolution_layer->_stride),
      _inp(convolution_layer->_inp),
      _out(convolution_layer->_out),
      _channels(convolution_layer->_channels) {
    _previous = convolution_layer->_previous;
    // convolution_layer->reset_previous(std::make_shared<Layer>(this));
    initialize_output_dimension();
    ;
}

void Im2ColLayer::initialize_output_dimension() {
    _out_dim[0] = _out.first() * _out.second();
}

int Im2ColLayer::input_dimension() {
    return _channels.get() * _kernel.first() * _kernel.second();
}

int Im2ColLayer::input_dimension() const {
    return _channels.get() * _kernel.first() * _kernel.second();
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
