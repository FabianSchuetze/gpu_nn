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
using std::vector;

typedef std::shared_ptr<Storage> SharedStorage;

// void print_Matrix_to_stdout2(const Matrix& val, std::string loc) {
// int rows(val.rows()), cols(val.cols());
// std::ofstream myfile(loc);
// myfile << "dimensions: rows, cols: " << rows << ", " << cols << std::endl;
// myfile << std::fixed;
// myfile << std::setprecision(2);
// for (int row = 0; row < rows; ++row) {
// myfile << val(row, 0);
// for (int col = 1; col < cols; ++col) {
// myfile << ", " << val(row, col);
//}
// myfile << std::endl;
//}
//}

Dense::Dense(Features out, Features in, Init* init)
    : Layer("Dense"), _out(out), _in(in) {
    _previous = NULL;
    cublasStatus_t stat = cublasCreate(&_handle);
    CHECK_CUBLAS(stat);
    initialize_weight(_out.get(), _in.get(), init);
    initialize_bias(_out.get(), _in.get());
    initialize_grad(_out.get(), _in.get());
    initialize_output_dimension();
}

Dense::Dense(Features out, const std::shared_ptr<Layer>& previous, Init* init)
    : Layer("Dense"), _out(out), _in(0) {
    _previous = previous;
    initialize_input_dimension(previous);
    cublasStatus_t stat = cublasCreate(&_handle);
    CHECK_CUBLAS(stat);
    initialize_weight(_out.get(), _in.get(), init);
    initialize_bias(_out.get(), _in.get());
    initialize_grad(_out.get(), _in.get());
    initialize_output_dimension();
}

void Dense::initialize_input_dimension(const std::shared_ptr<Layer>& previous) {
    std::vector<int> in = previous->output_dimension();
    int i = 1;
    if ((in.size() == 1) and (in[0] > 0)) {
        i = in[0];
    } else if (in.size() == 3) {
        for (int shape : previous->output_dimension()) i *= shape;
    } else {
        std::stringstream ss;
        ss << "Dimension do not fit, in:\n"
           << __PRETTY_FUNCTION__ << "\ncalled with layer " << previous->name()
           << " from\n"
           << __FILE__ << " at " << __LINE__;
        throw std::invalid_argument(ss.str());
    }
    _in = Features(i);
}

void Dense::initialize_output_dimension() { 
    _out_dim[0] =_out.get(); }

void Dense::forward_cpu(const SharedStorage& in, SharedStorage& out,
                        const std::string&) {
    const Matrix& in_ref = in->return_data_const();
    out->return_data() = parameters[0]->return_data_const() * in_ref;
    for (int i = 0; i < out->get_cols(); i++)
        out->return_data()(all, i) += parameters[1]->return_data_const();
}

void Dense::forward_gpu(const SharedStorage& in, SharedStorage& out,
                        const std::string&) {
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    dtype alpha = 1;
    dtype beta = 0;
    my_Dgemm(_handle, transA, transB, parameters[0], in, out, alpha, beta);
    my_add_vec_to_mat_colwise(out, parameters[1], 1.0f);
}

void Dense::resize_assistance(int batches) {
    if (batches != assistance_parameters[0]->get_rows()) {
        Matrix ones = Matrix::Ones(batches, 1);
        assistance_parameters[0] = std::make_shared<Storage>(ones);
    }
}

void Dense::backward_gpu(const SharedStorage& values,
                         const SharedStorage& gradient_in,
                         SharedStorage& gradient_out) {
    resize_assistance(gradient_in->get_cols());
    Matrix test = gradient_in->return_data_const().rowwise().sum();
    my_Dgemv(_handle, CUBLAS_OP_N, gradient_in, assistance_parameters[0],
             gradients[1], 1., 0); //bias
    my_Dgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_T, gradient_in, values,
             gradients[0], 1, 0); // weights
    my_Dgemm(_handle, CUBLAS_OP_T, CUBLAS_OP_N, parameters[0], gradient_in,
             gradient_out, 1, 0); //out
}

void Dense::backward_cpu(const SharedStorage& values,
                         const SharedStorage& gradient_in,
                         SharedStorage& gradient_out) {
    Matrix& bias_ref = gradients[1]->return_data();
    Matrix& weight_ref = gradients[0]->return_data();
    bias_ref = gradient_in->return_data_const().rowwise().sum();
    weight_ref = gradient_in->return_data_const() *
                 values->return_data_const().transpose();
    Matrix tmp = parameters[0]->return_data_const().transpose() *
                 gradient_in->return_data_const();
    if ((tmp.rows() != gradient_out->get_rows()) or
        (tmp.cols() != gradient_out->get_cols())) {
        std::string m("The gradient sizes don't fit, in:\n");
        throw std::runtime_error(m + __PRETTY_FUNCTION__);
    } else {
        gradient_out->update_cpu_data(tmp);
    }
}

void Dense::initialize_grad(int rows, int cols) {
    Matrix tmp = Matrix(rows, cols).setZero();
    Matrix bias_tmp = Matrix(rows, 1).setZero();
    Matrix ones = Matrix::Ones(32, 1); // guess
    gradients.push_back(std::make_shared<Storage>(tmp));
    gradients.push_back(std::make_shared<Storage>(bias_tmp));
    assistance_parameters.push_back(std::make_shared<Storage>(ones));
}

void Dense::initialize_weight(int rows, int cols, Init* init) {
    Matrix weights = init->weights(rows, cols);
    parameters.push_back(std::make_shared<Storage>(weights));
}

void Dense::initialize_bias(int rows, int cols) {
    Matrix mat = Matrix(rows, 1).setZero();
    parameters.push_back(std::make_shared<Storage>(mat));
}
