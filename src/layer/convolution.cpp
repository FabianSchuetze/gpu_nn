#include "../../include/layer/convolution.h"
#include <iostream>
#include "../../include/math.h"

Convolution::Convolution(FilterShape filtershape, Pad pad, Stride stride,
                         Filters filters, ImageShape imageshape,
                         Channels channels)
    : Layer("Convolution"),
      _filter_shape(filtershape),
      _pad(pad),
      _stride(stride),
      _filters(filters),
      _inp(imageshape),
      _out(0, 0),
      _channels(channels),
      batch_size(0) {
    initialize_weight();
    initialize_grad();
}

int Convolution::output_dimension() {
    return _filters.get() * _out.first() * _out.second();
}

int Convolution::output_dimension() const {
    return _filters.get() * _out.first() * _out.second();
}

void Convolution::initialize_grad() {
    int rows = _filters.get();
    int cols = _channels.get() * _filter_shape.get().first *
               _filter_shape.get().second;
    Matrix tmp = Matrix(rows, cols).setZero();
    gradients.push_back(std::make_shared<Storage>(tmp));
}

void Convolution::initialize_weight() {
    srand((unsigned int)time(0));
    int rows = _filters.get();
    int cols = _channels.get() * _filter_shape.get().first *
               _filter_shape.get().second;
    Matrix tmp = Matrix::Random(rows, cols);
    dtype glorot_scale = std::sqrt(6.) / std::sqrt(rows + cols);
    tmp *= glorot_scale;
    parameters.push_back(std::make_shared<Storage>(tmp));
}

void Convolution::forward_gpu(const SharedStorage& in, SharedStorage& out,
                              const std::string&) {
    resize_gpu(in->get_cols());
    const float alpha = 1.0f, beta = 0.f;
    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn, &alpha, input_des, in->gpu_pointer_const(), kernel_des,
        parameters[0]->gpu_pointer_const(), convolution_des,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        // convolution_algorithm,
        ffw.gpu_pointer(), ffw.size(), &beta, output_des, out->gpu_pointer()));
}

void Convolution::im2col(const SharedStorage& image) {
    dtype* colp = col->cpu_pointer();
    dtype const* imagep = image->cpu_pointer_const();
    for (int t = 0; t < image->get_cols(); ++t) {
        ::im2col(imagep, _channels.get(), _inp.get().first, _inp.get().second,
                 _filter_shape.get().first, _filter_shape.get().second,
                 _pad.get(), _pad.get(), _pad.get(), _pad.get(), _stride.get(),
                 _stride.get(), colp);
        imagep += _inp.get().first * _inp.get().second * _channels.get();
        colp += _filter_shape.get().first * _filter_shape.get().second *
                _channels.get() * _out.get().first * _out.get().second;
    }
}


void Convolution::forward_cpu(const SharedStorage& in, SharedStorage& out,
                              const std::string&) {
    resize_cpu(in->get_cols());
    im2col(in);
    out->return_data() =
        parameters[0]->return_data_const() * col->return_data_const();
}

void Convolution::backward_gpu(const SharedStorage& values,
                               const SharedStorage& gradient_in,
                               SharedStorage& gradient_out) {
    const float alpha = 1.0f, beta = 0.f;
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(
        cudnn, &alpha, input_des, values->gpu_pointer_const(), output_des,
        gradient_in->gpu_pointer_const(), convolution_des,
        convolution_bwd_algorithm, bwd.gpu_pointer(), bwd.size(), &beta,
        kernel_des, gradients[0]->gpu_pointer()));
    CHECK_CUDNN(cudnnConvolutionBackwardData(
        cudnn, &alpha, kernel_des, parameters[0]->gpu_pointer_const(),
        output_des, gradient_in->gpu_pointer_const(), convolution_des,
        convolution_bwd_data_algo, bwd_data.gpu_pointer(), bwd_data.size(),
        &beta, input_des, gradient_out->gpu_pointer()));
};

void Convolution::backward_cpu(const SharedStorage&, const SharedStorage&,
                               SharedStorage&) {
    ;
};
