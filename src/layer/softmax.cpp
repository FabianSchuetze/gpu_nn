#include "../../include/layer/softmax.h"
#include <cuda_runtime.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <iostream>
#include <memory>
#include "../../include/layer/layer.h"
#include "../../include/math.h"
#include "../../include/cuda_math.h"

using Eigen::all;
// using Eigen::MatrixXd;
using std::make_shared;
using std::vector;

typedef std::shared_ptr<Storage> SharedStorage;

Softmax::Softmax() : Layer("Softmax") {
    cublasStatus_t stat = cublasCreate(&_handle);
    CHECK_CUBLAS(stat);
    create_storage();
}

Softmax::Softmax(const std::shared_ptr<Layer>& previous) : Layer("Softmax") {
    cublasStatus_t stat = cublasCreate(&_handle);
    CHECK_CUBLAS(stat);
    initialize_output_dimension(previous);
    _previous = previous;
    create_storage();
}


void Softmax::create_storage() {
    max = make_shared<Storage>(Matrix::Zero(32, 1));
    ones = make_shared<Storage>(Matrix::Ones(_out_dim[0], 1));
}

void Softmax::forward_cpu(const SharedStorage& in, SharedStorage& out,
                          const std::string&) {
    int cols = in->get_cols();
    const Matrix& in_ref = in->return_data_const();
    Matrix& out_ref = out->return_data();
    for (int i = 0; i < cols; i++)
        out_ref(all, i) = in_ref(all, i).array() - in_ref(all, i).maxCoeff();
    out_ref = out_ref.array().exp();
    Vector summation = out_ref.colwise().sum();
    for (int i = 0; i < cols; i++)
        out_ref(all, i) = out_ref(all, i).array() / summation(i);
}

void Softmax::resize_maybe(int features, int obs) {
    if (obs != max->get_rows()) {
        max = make_shared<Storage>(Matrix::Zero(obs, 1));
    }
    if (_out_dim[0] != features) {
        _out_dim[0] = features;
        ones = make_shared<Storage>(Matrix::Ones(features, 1));
    }
}

void Softmax::forward_gpu(const SharedStorage& in, SharedStorage& out,
                          const std::string&) {
    int rows = in->get_rows();
    int obs = in->get_cols();
    resize_maybe(rows, obs);
    cuda_colwise_max(in->gpu_pointer_const(), rows, obs, max->gpu_pointer());
    my_add_vec_to_mat_colwise(in, max, out, -1.0f);
    my_Exponential(out);
    my_Dgemv(_handle, CUBLAS_OP_T, out, ones, max, 1, 0.0f);
    my_Divide_colwise(out, max);
}

void Softmax::backward_gpu(const SharedStorage&, const SharedStorage&,
                           SharedStorage&) {}
void Softmax::backward_cpu(const SharedStorage&, const SharedStorage&,
                           SharedStorage&) {}
