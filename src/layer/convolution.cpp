#include "../../include/layer/convolution.h"
#include <iostream>
#include "../../include/math.h"
#include <cblas.h>

Convolution::Convolution(FilterShape filtershape, Pad pad, Stride stride,
                         Filters filters, ImageShape imageshape,
                         Channels channels)
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
    initialize_weight();
    initialize_grad();
    output_shape();
}
Convolution::~Convolution() {CHECK_CUBLAS(cublasDestroy(_handle));};

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
    gradients.push_back(std::make_shared<Storage>(tmp));
}

void Convolution::initialize_weight() {
    srand((unsigned int)time(0));
    int cols = _filters.get();
    int rows = _channels.get() * _kernel.first() * _kernel.second();
    Matrix tmp = Matrix::Random(rows, cols);
    dtype glorot_scale = std::sqrt(6.) / std::sqrt(rows + cols);
    tmp *= glorot_scale;
    parameters.push_back(std::make_shared<Storage>(tmp));
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
void Convolution::advance_pointers_forward(const float*& input, float*& output) {
    //input += (input_rows);
    input += (_out.first() * _out.second() * _kernel.first() *
               _kernel.second() * _channels.get());
    output += _out.first() * _out.second() * _filters.get();
}

void Convolution::forward_gpu(const SharedStorage& in, SharedStorage& out,
                              const std::string&) {}

void Convolution::forward_cpu(const SharedStorage& in, SharedStorage& out,
                              const std::string&) {
    const float* inpp = in->cpu_pointer_const();
    float* outp = out->cpu_pointer();
    int M = _out.first() * _out.second();
    int N = _filters.get();
    int K = _channels.get() * _kernel.first() * _kernel.second();
    for (int n = 0; n < n_batches(in); ++n) {
        std::cout << "finised\n" << std::endl;
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                    inpp, M, parameters[0]->cpu_pointer_const(), K, 0.0f, outp,
                    M);
        std::cout << out->cpu_pointer();
        advance_pointers_forward(inpp, outp);
    }
}

void Convolution::backward_gpu(const SharedStorage& values,
                               const SharedStorage& gradient_in,
                               SharedStorage& gradient_out){};

void Convolution::backward_cpu(const SharedStorage&, const SharedStorage&,
                               SharedStorage&){};
