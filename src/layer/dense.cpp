#include "../../include/layer/dense.h"
#include <cuda_runtime.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <iostream>
#include <memory>
#include "../../include/common.h"
#include "../../include/layer/layer.h"
#include "../../include/math.h"
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

void Dense::forward_cpu(const SharedStorage& in, SharedStorage& out) {}
void Dense::forward_gpu(const SharedStorage& in, SharedStorage& out) {
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    // print_Matrix_to_stdout(in->return_data_const(), "../debug/in.txt");
    // print_Matrix_to_stdout(parameters[0]->return_data_const(),
    //"../debug/weight.txt");
    // print_Matrix_to_stdout(out->return_data_const(), "../debug/out.txt");
    my_Dgemm(_handle, transA, transB, parameters[0], in, out, 1, 1);
}

void Dense::backward_gpu(int, const vector<SharedStorage>& values,
                         vector<SharedStorage>& gradient) {
    // I need a matrix vector product here
    my_Dgemv(_handle, CUBLAS_OP_N, gradient[1], parameters[2], gradients[1], 1,
             1);
    my_Dgemm(_handle, CUBLAS_OP_T, CUBLAS_OP_N, gradient[1], values[0],
             gradients[0], 1, 1);
    // gradients[0] += gradient_in * values[values_idx--].transpose();
    my_Dgemm(_handle, CUBLAS_OP_T, CUBLAS_OP_N, parameters[0], gradient[1],
             gradient[0], 1, 1);
    // gradient = parameters[paras_idx--].transpose() * gradient;
}
void Dense::backward_cpu(int, const vector<SharedStorage>& values,
       vector<SharedStorage>& gradient) {}

void Dense::initialize_grad(int rows, int cols) {
    MatrixXd tmp = MatrixXd(rows, cols).setZero();
    MatrixXd bias_tmp = MatrixXd(rows, 1).setZero();
    gradients.push_back(std::make_shared<Storage>(tmp));
    gradients.push_back(std::make_shared<Storage>(bias_tmp));
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