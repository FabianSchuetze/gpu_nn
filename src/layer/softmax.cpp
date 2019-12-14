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

void Softmax::resize_maybe(int cols) {
    if (cols != max->get_rows()) {
        max = make_shared<Storage>(Matrix::Zero(cols, 1));
    }
}

void Softmax::forward_gpu(const SharedStorage& in, SharedStorage& out,
                          const std::string&) {
    int rows = in->get_rows();
    int cols = in->get_cols();
    resize_maybe(cols);
    // Ones could be part of the class definition
    //SharedStorage ones = make_shared<Storage>(Matrix::Ones(rows, 1));
    //SharedStorage tmp = make_shared<Storage>(Matrix::Zero(cols, 1));
    //SharedStorage tmp2 = make_shared<Storage>(Matrix::Zero(cols, 1));
    cuda_colwise_max(in->gpu_pointer_const(), rows, cols, max->gpu_pointer());
    // I NEED TO ADD THE PROPER MAX REDUCTION ALGORITHM!
    //my_Dgemv(_handle, CUBLAS_OP_T, in, ones, tmp, 1, 1);
    my_add_vec_to_mat_colwise(in, max, out, -1.0f);
    my_Exponential(out);
    my_Dgemv(_handle, CUBLAS_OP_T, out, ones, max, 1, 0.0f);
    my_Divide_colwise(out, max);
}

void Softmax::backward_gpu(const SharedStorage&, const SharedStorage&,
                           SharedStorage&) {}
void Softmax::backward_cpu(const SharedStorage&, const SharedStorage&,
                           SharedStorage&) {}
