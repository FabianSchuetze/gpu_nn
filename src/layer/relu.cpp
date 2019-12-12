#include "../../include/layer/relu.h"
#include <eigen-git-mirror/Eigen/Dense>
//#include <iostream>
//#include <memory>
#include <iostream>
#include <memory>
#include "../../include/layer/layer.h"
#include "../../include/math.h"

using std::vector;
Relu::Relu() : Layer("Activation") {
    cublasStatus_t stat = cublasCreate(&_handle);
    CHECK_CUBLAS(stat);
    //_name = "Activation";
}

Relu::Relu(const std::shared_ptr<Layer>& previous) : Layer("Activation") {
    cublasStatus_t stat = cublasCreate(&_handle);
    CHECK_CUBLAS(stat);
    initialize_output_dimension(previous);
    _previous = previous;
}

void Relu::forward_cpu(const SharedStorage& in, SharedStorage& out,
                       const std::string&) {
    out->return_data() = in->return_data_const().cwiseMax(0.);
}

void Relu::forward_gpu(const SharedStorage& in, SharedStorage& out,
                       const std::string&) {
    my_relu(out, in);
}

void Relu::backward_gpu(const SharedStorage& values,
                        const SharedStorage& gradient_in,
                        SharedStorage& gradient_out) {
    my_relu_backwards(values, gradient_in, gradient_out);
}

void Relu::backward_cpu(const SharedStorage& values,
                        const SharedStorage& gradient_in,
                        SharedStorage& gradient_out) {
    const Matrix& value_ref = values->return_data_const();
    const Matrix& grad_in = gradient_in->return_data_const();
    Matrix& grad_out = gradient_out->return_data();
    Matrix tmp(values->get_rows(), values->get_cols());
    for (int i = 0; i < values->get_cols(); i++) {
        for (int j = 0; j < values->get_rows(); j++)
            tmp(j, i) = (value_ref(j, i) > 0) ? 1. : 0.;
    }
    Matrix out = grad_in.array() * tmp.array();
    if ((tmp.rows() != gradient_out->get_rows()) or
        (tmp.cols() != gradient_out->get_cols())) {
        throw std::runtime_error("Doesn work");
    } else {
        grad_out = out;
    }
}
