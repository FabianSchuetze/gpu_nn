#include "../../include/layer/dense.h"
#include <cuda_runtime.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <memory>
#include "../../include/common.h"
#include "../../include/layer/layer.h"
#include "../../include/math.h"
#include <iostream>
using Eigen::MatrixXd;

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
    print_Matrix_to_stdout(in->return_data_const(), "../debug/in.txt");
    print_Matrix_to_stdout(parameters[0]->return_data_const(),
                           "../debug/weight.txt");
    print_Matrix_to_stdout(out->return_data_const(), "../debug/out.txt");
    multonGPU(_handle, parameters[0], in, out, 1, 1);
}

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
