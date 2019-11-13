#include "../../include/layer/softmax.h"
#include <cuda_runtime.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <iostream>
#include <memory>
#include "../../include/layer/layer.h"
#include "../../include/math.h"

using Eigen::all;
// using Eigen::MatrixXd;
using std::make_shared;
using std::vector;

typedef std::shared_ptr<Storage> SharedStorage;

Softmax::Softmax() : Layer() {
    cublasStatus_t stat = cublasCreate(&_handle);
    CHECK_CUBLAS(stat);
    _name = "Activation";
}

void Softmax::forward_cpu(const SharedStorage& in, SharedStorage& out) {
    int cols = in->get_cols();
    const Matrix& in_ref = in->return_data_const();
    Matrix& out_ref = out->return_data();
    for (int i = 0; i < cols; i++)
        out_ref(all, i) = in_ref(all, i).array() - in_ref(all, i).sum();
    out_ref = out_ref.array().exp();
    Vector summation = out_ref.colwise().sum();
    for (int i = 0; i < cols; i++)
        out_ref(all, i) = out_ref(all, i).array() / summation(i);
}

void Softmax::forward_gpu(const SharedStorage& in, SharedStorage& out) {
    int rows = in->get_rows();
    int cols = in->get_cols();
    SharedStorage ones = make_shared<Storage>(Matrix::Ones(rows, 1));
    SharedStorage tmp = make_shared<Storage>(Matrix::Zero(cols, 1));
    my_Dgemv(_handle, CUBLAS_OP_T, in, ones, tmp, 1, 1);
    my_add_vec_to_mat_colwise(in, tmp, out, -1.0f);
    my_Exponential(out);
    my_Dgemv(_handle, CUBLAS_OP_T, out, ones, tmp, 1, 0.0f);
    my_Divide_colwise(out, tmp);
}

void Softmax::backward_gpu(const SharedStorage&, const SharedStorage&,
                           SharedStorage&) {}
void Softmax::backward_cpu(const SharedStorage&, const SharedStorage&,
                           SharedStorage&) {}
void Softmax::clear_gradients_cpu(){};
void Softmax::clear_gradients_gpu(){};
