#include "../../include/layer/dense.h"
//#include <cuda_runtime.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "../../include/common.h"
#include "../../include/layer/layer.h"
#include "../../include/math.h"

using Eigen::all;
// using Eigen::MatrixXd;
using std::vector;

typedef std::shared_ptr<Storage> SharedStorage;

//void print_Matrix_to_stdout(const Eigen::MatrixXd& val, std::string loc) {
    //int rows(val.rows()), cols(val.cols());
    //std::ofstream myfile(loc);
    //myfile << "dimensions: rows, cols: " << rows << ", " << cols << std::endl;
    //myfile << std::fixed;
    //myfile << std::setprecision(2);
    //for (int row = 0; row < rows; ++row) {
        //myfile << val(row, 0);
        //for (int col = 1; col < cols; ++col) {
            //myfile << ", " << val(row, col);
        //}
        //myfile << std::endl;
    //}
//}

Dense::Dense(int rows, int cols, cublasHandle_t& handle)
    : Layer(),
      _handle(handle),
      _input_dimension(cols),
      _output_dimension(rows) {
    initialize_weight(rows, cols);
    initialize_bias(rows, cols);
    initialize_grad(rows, cols);
    _name = "Dense";
}

void Dense::forward_cpu(const SharedStorage& in, SharedStorage& out) {
    const Matrix& in_ref = in->return_data_const();
    out->return_data() = parameters[0]->return_data_const() * in_ref;
    for (int i = 0; i < out->get_cols(); i++)
        out->return_data()(all, i) += parameters[1]->return_data_const();
}

void Dense::forward_gpu(const SharedStorage& in, SharedStorage& out) {
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    dtype alpha = 1;
    dtype beta = 1;
    my_Dgemm(_handle, transA, transB, parameters[0], in, out, alpha, beta);
    my_add_vec_to_mat_colwise(out, parameters[1], 1.0f);
}

void Dense::backward_gpu(int& idx, const SharedStorage& values,
                         vector<SharedStorage>& gradient) {
    my_Dgemv(_handle, CUBLAS_OP_N, gradient[idx], parameters[2], gradients[1],
             1, 1);
    my_Dgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_T, gradient[idx], values,
             gradients[0], 1, 1);
    my_Dgemm(_handle, CUBLAS_OP_T, CUBLAS_OP_N, parameters[0], gradient[idx],
             gradient[idx - 1], 1, 1);
    idx--;
}

void Dense::backward_cpu(int& idx, const SharedStorage& values,
                         vector<SharedStorage>& gradient) {
    Matrix& bias_ref = gradients[1]->return_data();
    Matrix& weight_ref = gradients[0]->return_data();
    const Matrix& grad_in = gradient[idx--]->return_data_const();
    Matrix& grad_out = gradient[idx]->return_data();
    bias_ref += grad_in.rowwise().sum();
    weight_ref += grad_in * values->return_data_const().transpose();
    Matrix tmp = parameters[0]->return_data_const().transpose() * grad_in;
    if ((tmp.rows() != grad_out.rows()) or (tmp.cols() != grad_out.cols())) {
        throw std::runtime_error("Doesn work");
    } else {
        grad_out = tmp;
    }
}

void Dense::initialize_grad(int rows, int cols) {
    Matrix tmp = Matrix(rows, cols).setZero();
    Matrix bias_tmp = Matrix(rows, 1).setZero();
    Matrix ones = Matrix::Ones(rows, 1);
    gradients.push_back(std::make_shared<Storage>(tmp));
    gradients.push_back(std::make_shared<Storage>(bias_tmp));
    parameters.push_back(std::make_shared<Storage>(ones));
}
void Dense::initialize_weight(int rows, int cols) {
    Matrix mat = Matrix::Random(rows, cols);
    dtype glorot_scale = std::sqrt(6.) / std::sqrt(rows + cols);
    mat *= glorot_scale;
    parameters.push_back(std::make_shared<Storage>(mat));
}

void Dense::initialize_bias(int rows, int cols) {
    Matrix mat = Matrix(rows, 1).setZero();
    parameters.push_back(std::make_shared<Storage>(mat));
}
