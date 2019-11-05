#include "../../include/layer/dense.h"
//#include <cuda_runtime.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
//#include "../../include/common.h"
//#include "../../include/cuda_math.h"
#include "../../include/layer/layer.h"
#include "../../include/math.h"

using Eigen::all;
using Eigen::MatrixXd;
using std::vector;

typedef std::shared_ptr<Storage> SharedStorage;

void print_Matrix_to_stdout(const Eigen::MatrixXd& val, std::string loc) {
    int rows(val.rows()), cols(val.cols());
    std::ofstream myfile(loc);
    myfile << "dimensions: rows, cols: " << rows << ", " << cols << std::endl;
    myfile << std::fixed;
    myfile << std::setprecision(2);
    for (int row = 0; row < rows; ++row) {
        myfile << val(row, 0);
        for (int col = 1; col < cols; ++col) {
            myfile << ", " << val(row, col);
        }
        myfile << std::endl;
    }
}

Dense::Dense(int rows, int cols, cublasHandle_t& handle)
    : Layer(), _handle(handle) {
    initialize_weight(rows, cols), initialize_bias(rows, cols),
        initialize_grad(rows, cols);
}

void Dense::forward_cpu(const SharedStorage& in, SharedStorage& out) {
    const Eigen::MatrixXd& in_ref = in->return_data_const();
    out->return_data() = parameters[0]->return_data_const() * in_ref;
    for (int i = 0; i < out->get_cols(); i++)
        out->return_data()(all, i) += parameters[1]->return_data_const();
}

void Dense::forward_gpu(const SharedStorage& in, SharedStorage& out) {
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    my_Dgemm(_handle, transA, transB, parameters[0], in, out, 1, 1);
    my_add_vec_to_mat_colwise(out, parameters[1], 1.0f);
}

void Dense::backward_gpu(int& idx, const SharedStorage& values,
                         vector<SharedStorage>& gradient) {
    my_Dgemv(_handle, CUBLAS_OP_N, gradient[idx--], parameters[2], gradients[1],
             1, 1);
    my_Dgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_T, gradient[idx], values,
             gradients[0], 1, 1);
    my_Dgemm(_handle, CUBLAS_OP_T, CUBLAS_OP_N, parameters[0], gradient[idx],
             gradient[0], 1, 1);
}

void Dense::backward_cpu(int& idx, const SharedStorage& values,
                         vector<SharedStorage>& gradient) {
    Eigen::MatrixXd& bias_ref = parameters[1]->return_data();
    Eigen::MatrixXd& weight_ref = parameters[0]->return_data();
    const Eigen::MatrixXd& grad_in = gradient[idx--]->return_data_const();
    Eigen::MatrixXd& grad_out = gradient[idx]->return_data();
    bias_ref += grad_in.rowwise().sum();
    weight_ref += grad_in * values->return_data_const().transpose();
    MatrixXd tmp = parameters[0]->return_data_const().transpose() * grad_in;
    if ((tmp.rows() != grad_out.rows()) or (tmp.cols() != grad_out.cols())) {
        throw std::runtime_error("Doesn work");
    } else {
        grad_out = tmp;
    }
}

void Dense::initialize_grad(int rows, int cols) {
    MatrixXd tmp = MatrixXd(rows, cols).setZero();
    MatrixXd bias_tmp = MatrixXd(rows, 1).setZero();
    MatrixXd ones = MatrixXd::Ones(rows, 1);
    gradients.push_back(std::make_shared<Storage>(tmp));
    gradients.push_back(std::make_shared<Storage>(bias_tmp));
    parameters.push_back(std::make_shared<Storage>(ones));
}
void Dense::initialize_weight(int rows, int cols) {
    MatrixXd mat = MatrixXd::Random(rows, cols);
    double glorot_scale = std::sqrt(6.) / std::sqrt(rows + cols);
    mat *= glorot_scale;
    parameters.push_back(std::make_shared<Storage>(mat));
}

void Dense::initialize_bias(int rows, int cols) {
    MatrixXd mat = MatrixXd(rows, 1).setZero();
    parameters.push_back(std::make_shared<Storage>(mat));
}
