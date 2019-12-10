#include "../../include/layer/convolution.h"
#include <cblas.h>
//#include "/usr/lib/x86_64-linux-gnu/cblas_atlas.h>
//c
#include <iostream>
#include <memory>
#include <random>
#include "../../include/cuda_math.h"
#include "../../include/math.h"

Convolution::Convolution(FilterShape filtershape, Pad pad, Stride stride,
                         Filters filters, ImageShape imageshape,
                         Channels channels, Init* init)
    : Layer("Convolution"),
      _kernel(filtershape),
      _pad(pad),
      _stride(stride),
      _filters(filters),
      _inp(imageshape),
      _out(0, 0),
      _channels(channels) {
    cublasStatus_t stat = cublasCreate(&_handle);
    CHECK_CUBLAS(stat);
    output_shape();
    initialize_weight(init);
    initialize_bias();
    initialize_grad();
}
Convolution::~Convolution() { CHECK_CUBLAS(cublasDestroy(_handle)); };

void Convolution::output_shape() {
    int out_height =
        (_inp.first() + 2 * _pad.get() - _kernel.first()) / _stride.get() + 1;
    int out_width =
        (_inp.second() + 2 * _pad.get() - _kernel.second()) / _stride.get() + 1;
    _out = ImageShape(out_height, out_width);
}

int Convolution::output_dimension() {
    return _filters.get() * _out.first() * _out.second();
}

int Convolution::output_dimension() const {
    return _filters.get() * _out.first() * _out.second();
}

void Convolution::initialize_grad() {
    int cols = _filters.get();
    int rows = _channels.get() * _kernel.first() * _kernel.second();
    Matrix tmp = Matrix(rows, cols).setZero();
    int bias_rows = _filters.get() * _out.first() * _out.second();
    Matrix bias_tmp = Matrix(bias_rows, 1).setZero();
    Matrix assistance = Matrix(32, 1);
    gradients.push_back(std::make_shared<Storage>(tmp));
    gradients.push_back(std::make_shared<Storage>(bias_tmp));
    assistance_parameters.push_back(std::make_shared<Storage>(assistance));
}

void Convolution::initialize_weight(Init* init) {
    int cols = _filters.get();
    int rows = _channels.get() * _kernel.first() * _kernel.second();
    Matrix weights = init->weights(rows, cols);
    parameters.push_back(std::make_shared<Storage>(weights));
}

void Convolution::initialize_bias() {
    int rows = _filters.get() * _out.first() * _out.second();
    Matrix mat = Matrix(rows, 1).setZero();
    parameters.push_back(std::make_shared<Storage>(mat));
}

int Convolution::n_batches(const SharedStorage& in) {
    int ind_width = _channels.get() * _kernel.first() * _kernel.second();
    if (in->get_cols() % ind_width) {
        std::stringstream ss;
        ss << "Dimension do not fit, in:\n"
           << __PRETTY_FUNCTION__
           << "\n expected cols to be a multiple of: " << ind_width
           << " received cols " << in->get_cols() << "\ncalled from "
           << __FILE__ << " at " << __LINE__;
        throw std::invalid_argument(ss.str());
    }
    return in->get_cols() / ind_width;
}

void Convolution::check_size(const SharedStorage& out) {
    int expected_rows = _out.first() * _out.second() * _filters.get();
    if (expected_rows != out->get_rows()) {
        std::stringstream ss;
        ss << "Dimension do not fit, in:\n"
           << __PRETTY_FUNCTION__ << "\n expected rows: " << expected_rows
           << " received rows " << out->get_rows() << "\ncalled from "
           << __FILE__ << " at " << __LINE__;
        throw std::invalid_argument(ss.str());
    }
}
void Convolution::advance_pointers_forward(const float*& input,
                                           float*& output) {
    input += (_out.first() * _out.second() * _kernel.first() *
              _kernel.second() * _channels.get());
    output += _out.first() * _out.second() * _filters.get();
}

void Convolution::forward_gpu(const SharedStorage& in, SharedStorage& out,
                              const std::string&) {
    const float* inpp = in->gpu_pointer_const();
    const float* wp = parameters[0]->gpu_pointer_const();
    float* outp = out->gpu_pointer();
    int M = _out.first() * _out.second();
    int N = _filters.get();
    int K = _channels.get() * _kernel.first() * _kernel.second();
    float alpha = 1.0f;
    float beta = 0.0f;
    float* alphap = &alpha;
    for (int n = 0; n < n_batches(in); ++n) {
        my_cuda_Dgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, alphap, inpp,
                      M, wp, K, &beta, outp, M);
        advance_pointers_forward(inpp, outp);
    }
    my_add_vec_to_mat_colwise(out, parameters[1], 1.0f);
}

void Convolution::forward_cpu(const SharedStorage& in, SharedStorage& out,
                              const std::string&) {
    const float* inpp = in->cpu_pointer_const();
    float* outp = out->cpu_pointer();
    int M = _out.first() * _out.second();
    int N = _filters.get();
    int K = _channels.get() * _kernel.first() * _kernel.second();
    for (int n = 0; n < n_batches(in); ++n) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                    inpp, M, parameters[0]->cpu_pointer_const(), K, 0.0f, outp,
                    M);
        advance_pointers_forward(inpp, outp);
    }
}

void Convolution::advance_pointers_backward(const float*& grad_in,
                                            const float*& values,
                                            float*& grad_out) {
    grad_in += _out.first() * _out.second() * _filters.get();
    values += (_out.first() * _out.second() * _kernel.first() *
               _kernel.second() * _channels.get());
    grad_out += (_out.first() * _out.second() * _kernel.first() *
                 _kernel.second() * _channels.get());
}

void Convolution::backwards_weight_grad_para(int& M, int& N, int& K) {
    M = _channels.get() * _kernel.first() * _kernel.second();
    N = _filters.get();
    K = _out.first() * _out.second();
}

void Convolution::check_size_backwards(const SharedStorage& values,
                                       const SharedStorage& grad_out) {
    if ((values->get_cols() != grad_out->get_cols()) or
        (values->get_rows() != grad_out->get_rows())) {
        std::stringstream ss;
        ss << "values and grad_out must have same dimension, in:\n"
           << __PRETTY_FUNCTION__ << "\ncalled from " << __FILE__ << " at "
           << __LINE__;
        throw std::invalid_argument(ss.str());
    }
}

void Convolution::resize_assistance(const SharedStorage& in) {
    if (in->get_cols() != assistance_parameters[0]->get_rows()) {
        Matrix ones = Matrix::Ones(in->get_cols(), 1);
        assistance_parameters[0] = std::make_shared<Storage>(ones);
    }
}

void Convolution::backward_gpu(const SharedStorage& values,
                               const SharedStorage& gradient_in,
                               SharedStorage& gradient_out) {
    check_size_backwards(values, gradient_out);
    resize_assistance(gradient_in);
    int M, N, K;
    backwards_weight_grad_para(M, N, K);
    const float* valp = values->gpu_pointer_const();
    const float* grad_inp = gradient_in->gpu_pointer_const();
    const float* wp = parameters[0]->gpu_pointer_const();
    float* weight_gradp = gradients[0]->gpu_pointer();
    float* grad_outp = gradient_out->gpu_pointer();
    float beta = 0.0f;
    float alpha = 1.0f;
    float* alphap = &alpha;
    my_Dgemv(_handle, CUBLAS_OP_N, gradient_in, assistance_parameters[0],
             gradients[1], 2, 0);
    for (int n = 0; n < gradient_in->get_cols(); ++n) {
        my_cuda_Dgemm(_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, alphap,
                    valp, K, grad_inp, K, &beta, weight_gradp, M);
        beta = 0.0f;
        my_cuda_Dgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_T, K, M, N, alphap,
                    grad_inp, K, wp, M, &beta, grad_outp, K);
        beta = 1.0f;
        advance_pointers_backward(grad_inp, valp, grad_outp);
    }
}

void Convolution::backward_cpu(const SharedStorage& values,
                               const SharedStorage& gradient_in,
                               SharedStorage& gradient_out) {
    check_size_backwards(values, gradient_out);
    int M, N, K;
    backwards_weight_grad_para(M, N, K);
    const float* valp = values->cpu_pointer_const();
    const float* grad_inp = gradient_in->cpu_pointer_const();
    const float* wp = parameters[0]->cpu_pointer_const();
    float* weight_gradp = gradients[0]->cpu_pointer();
    float* grad_outp = gradient_out->cpu_pointer();
    float beta = 0.0f;
    for (int n = 0; n < gradient_in->get_cols(); ++n) {
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0f,
                    valp, K, grad_inp, K, beta, weight_gradp, M);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, K, M, N, 1.0f,
                    grad_inp, K, wp, M, 0.0f, grad_outp, K);
        beta = 1.0f;
        advance_pointers_backward(grad_inp, valp, grad_outp);
    }
}
