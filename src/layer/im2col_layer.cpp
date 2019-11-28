#include "../../include/layer/im2col_layer.h"
#include "../../include/math.h"
#include "../../include/cuda_math.h"
#include <iostream>
Im2ColLayer::Im2ColLayer(FilterShape filtershape, Pad pad, Stride stride,
                         Filters filters, ImageShape imageshape,
                         Channels channels)
    : Layer("Im2ColLayer"),
      _kernel(filtershape),
      _pad(pad),
      _stride(stride),
      _filters(filters),
      _inp(imageshape),
      _out(0, 0),
      _channels(channels) {
    output_shape();
};

void Im2ColLayer::output_shape() {
    int out_height =
        (_inp.first() + 2 * _pad.get() - _kernel.first()) / _stride.get() + 1;
    int out_width =
        (_inp.second() + 2 * _pad.get() - _kernel.second()) / _stride.get() + 1;
    _out = ImageShape(out_height, out_width);
}

void Im2ColLayer::check_size(const SharedStorage& in, const SharedStorage& out) {
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

void Im2ColLayer::advance_pointers(const float*& input, float*& column,
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
        advance_pointers(inpp, colp, in->get_rows());
    }
}

void Im2ColLayer::forward_cpu(const SharedStorage& in, SharedStorage& out,
                              const std::string&) {
    check_size(in, out);
    const float* inpp = in->cpu_pointer_const();
    float* colp = out->cpu_pointer();
    for (int n = 0; n < in->get_cols(); n++) {
        im2col_cpu(inpp, _channels.get(), _inp.first(), _inp.second(),
                     _kernel.first(), _kernel.second(), _pad.get(),
                     _stride.get(), colp);
        advance_pointers(inpp, colp, in->get_rows());
    }
}
void Im2ColLayer::backward_gpu(const SharedStorage&, const SharedStorage&,
                               SharedStorage&){};
void Im2ColLayer::backward_cpu(const SharedStorage&, const SharedStorage&,
                               SharedStorage&){};
