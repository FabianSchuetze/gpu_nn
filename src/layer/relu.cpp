#include "../../include/layer/relu.h"
//#include <cuda_runtime.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <iostream>
#include <memory>
//#include "../../include/cuda_math.h"
#include "../../include/layer/layer.h"
#include "../../include/math.h"
#include <iostream>

using std::vector;
Relu::Relu(cublasHandle_t& handle)
    : Layer(), parameters(), gradients(), _handle(handle) {}

void Relu::forward_cpu(const SharedStorage& in, SharedStorage& out) {
    out->return_data() = in->return_data_const().cwiseMax(0.);
}

void Relu::forward_gpu(const SharedStorage& in, SharedStorage& out) {
    my_relu(out, in);
}

void Relu::backward_gpu(int& idx, const SharedStorage& values,
                        vector<SharedStorage>& gradient) {
    SharedStorage& grad_in = gradient[idx--];
    SharedStorage& grad_out = gradient[idx];
    my_relu_backwards(values, grad_in, grad_out);
}

void Relu::backward_cpu(int& idx, const SharedStorage& values,
                        vector<SharedStorage>& gradient) {
    const Matrix& value_ref = values->return_data_const();
    const Matrix& grad_in = gradient[idx--]->return_data_const();
    Matrix& grad_out = gradient[idx]->return_data();
    Matrix tmp(values->get_rows(), values->get_cols());
    for (int i = 0; i < values->get_cols(); i++) {
        for (int j = 0; j < values->get_rows(); j++)
            tmp(j, i) = (value_ref(j, i) > 0) ? 1. : 0.;
    }
    Matrix out = grad_in.array() * tmp.array();
    if ((tmp.rows() != gradient[idx]->get_rows()) or 
            (tmp.cols() != gradient[idx]->get_cols())) {
        throw std::runtime_error("Doesn work");
    } else {
        grad_out = out;
    }
}
