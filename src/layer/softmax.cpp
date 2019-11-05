#include "../../include/layer/softmax.h"
#include <cuda_runtime.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <iostream>
#include <memory>
#include "../../include/common.h"
#include "../../include/cuda_math.h"
#include "../../include/layer/layer.h"
#include "../../include/math.h"
//#include <filesystem>
// namespace fs = std::filesystem

using Eigen::MatrixXd;
using std::make_shared;
using std::vector;

typedef std::shared_ptr<Storage> SharedStorage;

Softmax::Softmax(cublasHandle_t& handle)
    : Layer(), parameters(), gradients(), _handle(handle) {}

//Softmax::~Softmax() {};

void Softmax::forward_cpu(const SharedStorage& in, SharedStorage& out) {
    return;
}
void Softmax::forward_gpu(const SharedStorage& in, SharedStorage& out) {
    int rows = in->get_rows();
    int cols = in->get_cols();
    SharedStorage ones = make_shared<Storage>(Eigen::MatrixXd::Ones(rows, 1));
    SharedStorage tmp = make_shared<Storage>(Eigen::MatrixXd::Zero(cols, 1));
    my_Dgemv(_handle, CUBLAS_OP_T, in, ones, tmp, 1, 1);
    my_add_vec_to_mat_colwise(in, tmp, out, -1.0f);  // Cannot do inplace
    my_Exponential(out);
    my_Dgemv(_handle, CUBLAS_OP_T, out, ones, tmp, 1, 0.0f);
    my_Divide_colwise(out, tmp);  // can be done inplace
}
void Softmax::backward_gpu(int, const std::vector<std::shared_ptr<Storage>>&,
                           std::vector<std::shared_ptr<Storage>>&) {
    return;
}
void Softmax::backward_cpu(int, const std::vector<std::shared_ptr<Storage>>&,
                           std::vector<std::shared_ptr<Storage>>&) {
    return;
}
