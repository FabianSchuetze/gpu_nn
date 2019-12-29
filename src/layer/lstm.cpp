#include "../../include/layer/lstm.hpp"
#include <sys/time.h>
#include <iostream>
#include <memory>
#include "../../include/cuda_math.h"
#include "../../include/math.h"
using Eigen::all;
double cpuSecond2() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

LSTM::LSTM(Features out, Features in, Init* init)
    : Layer("LSTM"), _out(out), _in(in), states(6), assistance_parameters(0) {
    _previous = NULL;
    cublasStatus_t stat = cublasCreate(&_handle);
    CHECK_CUBLAS(stat);
    initialize_weight(init);
    initialize_grad();
    initialize_output_dimension();
    initialize_states();
}

LSTM::LSTM(Features out, const std::shared_ptr<Layer>& previous, Init* init)
    : Layer("LSTM"), _out(out), _in(0), states(6), assistance_parameters(0) {
    _previous = previous;
    initialize_input_dimension(previous);
    cublasStatus_t stat = cublasCreate(&_handle);
    CHECK_CUBLAS(stat);
    initialize_weight(init);
    initialize_grad();
    initialize_output_dimension();
    initialize_states();
}

void LSTM::initialize_states() {
    states[0] = std::make_shared<Storage>(Matrix::Zero(4 * _out.get(), 32));
    states[1] = std::make_shared<Storage>(Matrix::Zero(_out.get(), 32 + 1));
    states[2] = std::make_shared<Storage>(Matrix::Zero(_out.get(), 32 + 1));
    states[3] = std::make_shared<Storage>(Matrix::Zero(4 * _out.get(), 32));
    states[4] = std::make_shared<Storage>(Matrix::Zero(4 * _out.get(), 32));
    states[5] = std::make_shared<Storage>(Matrix::Zero(_out.get(), 32 + 1));
}

void LSTM::initialize_input_dimension(const std::shared_ptr<Layer>& previous) {
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

void LSTM::initialize_output_dimension() { _out_dim[0] = _out.get(); }

void LSTM::multiply_one_col_fwd(const SharedStorage& in, int col) {
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    int M = assistance_parameters[0]->get_rows();
    int N = 1;
    int K = in->get_rows();
    int LDA = M;
    int LDB = K;
    int LDC = M;
    dtype alpha = 1;
    dtype beta = 0;
    const float* d_A = parameters[0]->gpu_pointer_const();
    const float* d_B = in->gpu_pointer_const() + col * in->get_rows();
    float* d_C = assistance_parameters[0]->gpu_pointer();
    my_cuda_Dgemm(_handle, transA, transB, M, N, K, &alpha, d_A, LDA, d_B, LDB,
                  &beta, d_C, LDC);
    beta = 1;
    d_A = parameters[1]->gpu_pointer_const();
    d_B = states[2]->gpu_pointer_const() + col * states[2]->get_rows();
    K = states[2]->get_rows();
    LDB = K;
    my_cuda_Dgemm(_handle, transA, transB, M, N, K, &alpha, d_A, LDA, d_B, LDB,
                  &beta, d_C, LDC);
    my_add_vec_to_mat_colwise(assistance_parameters[0], parameters[2], 1.0f);
}

void LSTM::multiply_one_col_bwd(SharedStorage& out, int col) {
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    int M = parameters[1]->get_cols();
    int N = 1;
    int K = parameters[1]->get_rows();
    int LDA = K;
    int LDB = K;
    int LDC = M;
    dtype alpha = 1;
    dtype beta = 0;
    const float* d_A = parameters[1]->gpu_pointer_const();
    const float* d_B =
        states[3]->gpu_pointer_const() + col * states[3]->get_rows();
    float* d_C = assistance_parameters[1]->gpu_pointer();
    my_cuda_Dgemm(_handle, transA, transB, M, N, K, &alpha, d_A, LDA, d_B, LDB,
                  &beta, d_C, LDC);
    d_A = parameters[0]->gpu_pointer_const();
    M = parameters[0]->get_cols();
    K = parameters[0]->get_rows();
    LDA = K;
    LDB = K;
    LDC = M;
    d_C = out->gpu_pointer() + col * out->get_rows();
    my_cuda_Dgemm(_handle, transA, transB, M, N, K, &alpha, d_A, LDA, d_B, LDB,
                  &beta, d_C, LDC);
    multiply_elementwise(
        assistance_parameters[2]->get_rows(), 1,
        assistance_parameters[4]->gpu_pointer_const(),
        states[0]->gpu_pointer_const() + col * 4 * _out.get() + _out.get(),
        assistance_parameters[2]->gpu_pointer());
}

void LSTM::nonlinear_transformations(int t) {
    int n_sig = _out.get() * 3;
    int rows = states[0]->get_rows();
    cuda_sigmoid(n_sig, 1, assistance_parameters[0]->gpu_pointer_const(),
                 states[0]->gpu_pointer() + t * rows);
    cuda_tanh(_out.get(), 1,
              assistance_parameters[0]->gpu_pointer_const() + n_sig,
              states[0]->gpu_pointer() + t * rows + n_sig);
}

void LSTM::compute_next_state(int t) {
    next_lstm_cell(states[1]->get_rows(),
                   states[0]->gpu_pointer_const() + t * states[0]->get_rows(),
                   states[1]->gpu_pointer() + t * states[1]->get_rows());
    next_lstm_state(states[2]->get_rows(),
                    states[0]->gpu_pointer_const() + t * states[0]->get_rows(),
                    states[1]->gpu_pointer_const() + t * states[1]->get_rows(),
                    states[2]->gpu_pointer() + t * states[2]->get_rows());
}

void LSTM::forward_gpu(const SharedStorage& in, SharedStorage& out,
                       const std::string&) {
    maybe_resize_state(in->get_cols());
    for (int t = 0; t < in->get_cols(); ++t) {
        multiply_one_col_fwd(in, t);
        nonlinear_transformations(t);
        compute_next_state(t);
    }
    copy_data(out->get_rows(), out->get_cols(),
              states[2]->gpu_pointer_const() + states[2]->get_rows(),
              out->gpu_pointer());
};

void LSTM::forward_cpu(const SharedStorage& in, SharedStorage& out,
                       const std::string&) {
    maybe_resize_state(in->get_cols());
    int cols = states[2]->get_cols();
    Matrix& state = states[2]->return_data();
    Matrix& cell = states[1]->return_data();
    Matrix& funcs = states[0]->return_data();
    state(all, 0) = state(all, cols - 1);
    cell(all, 0) = states[1]->return_data()(all, cols - 1);
    Matrix& tmp = assistance_parameters[0]->return_data();
    const int sigmoid_rows = _out.get() * 3;
    for (int t = 0; t < in->get_cols(); ++t) {
        tmp = parameters[0]->return_data_const() *
                  in->return_data_const()(all, t) +
              parameters[1]->return_data_const() * state(all, t) +
              parameters[2]->return_data_const();
        funcs.block(0, t, sigmoid_rows, 1) = sigmoid(tmp.topRows(sigmoid_rows));
        funcs.block(sigmoid_rows, t, _out.get(), 1) =
            tmp.bottomRows(_out.get()).array().tanh();
        const Matrix& i = funcs.block(0, t, _out.get(), 1);
        const Matrix& f = funcs.block(_out.get(), t, _out.get(), 1);
        const Matrix& o = funcs.block(2 * _out.get(), t, _out.get(), 1);
        const Matrix& g = funcs.block(3 * _out.get(), t, _out.get(), 1);
        cell(all, t + 1) =
            f.array() * cell(all, t).array() + i.array() * g.array();
        state(all, t + 1) = o.array() * cell(all, t + 1).array().tanh();
    }
    out->return_data() = state.rightCols(cols - 1);
}

void LSTM::construct_sigma_cpu() {
    Matrix& sigma = states[4]->return_data();
    const Matrix& state = states[0]->return_data_const();
    int n_sig = 3 * _out.get();
    sigma.topRows(n_sig) = state.topRows(n_sig).array() *
                           (Matrix::Ones(n_sig, sigma.cols()).array() -
                            state.topRows(n_sig).array());
    sigma.bottomRows(_out.get()) =
        Matrix::Ones(_out.get(), sigma.cols()).array() -
        state.bottomRows(_out.get()).array().pow(2);
}

void LSTM::construct_sigma_gpu() {
    int n_sig = 3 * _out.get();
    ::sigmoid_deriv(n_sig, states[0]->get_rows(), states[0]->get_cols(),
                    states[0]->gpu_pointer_const(), states[4]->gpu_pointer());
    ::tanh_deriv(_out.get(), states[0]->get_rows(), states[0]->get_cols(),
                 states[0]->gpu_pointer_const(), states[4]->gpu_pointer());
}

void LSTM::new_hidden_state(int t, const SharedStorage& grad_in) {
    const dtype* in = grad_in->gpu_pointer_const() + t * grad_in->get_rows();
    const dtype* last_state = assistance_parameters[1]->gpu_pointer_const();
    dtype* next_state = assistance_parameters[3]->gpu_pointer();
    int rows = assistance_parameters[3]->get_rows();
    int cols = 1;
    matrix_addition(rows, cols, in, last_state, next_state, 1.0f, 1.0f);
}

void LSTM::new_cell_state(int t, const SharedStorage& sigma_c) {
    int rows = assistance_parameters[2]->get_rows();
    ::new_cell_state(rows, assistance_parameters[2]->gpu_pointer_const(),
                     assistance_parameters[3]->gpu_pointer_const(),
                     states[0]->gpu_pointer_const() + t * 4 * rows + 2 * rows,
                     sigma_c->gpu_pointer_const() + (t + 1) * rows,
                     assistance_parameters[4]->gpu_pointer());
}

void LSTM::internal_deriv(int t) {
    int nh = states[1]->get_rows();
    ::internal_deriv(nh, assistance_parameters[3]->gpu_pointer_const(),
                     assistance_parameters[4]->gpu_pointer_const(),
                     states[1]->gpu_pointer_const() + t * nh,
                     states[0]->gpu_pointer_const() + t * 4 * nh,
                     assistance_parameters[5]->gpu_pointer());
    ::multiply_elementwise(4 * nh, 1,
                           states[4]->gpu_pointer_const() + t * 4 * nh,
                           assistance_parameters[5]->gpu_pointer_const(),
                           states[3]->gpu_pointer() + t * 4 * nh);
}

void LSTM::clip_gradients_gpu() {
    for (SharedStorage& grad : gradients) {
        ::clip_gradients_gpu(grad->get_rows(), grad->get_cols(), 5.0f,
                             grad->gpu_pointer());
    }
}

void LSTM::para_gradients(const SharedStorage& values) {
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_T;
    my_Dgemm(_handle, transA, transB, states[3], values, gradients[0], 1.0,
             0.0);
    my_Dgemm(_handle, transA, transB, states[3], states[2], gradients[1], 1.0,
             0.0);
    my_Dgemv(_handle, CUBLAS_OP_N, states[3], assistance_parameters[6],
             gradients[2], 1., 0);  // bias
    clip_gradients_gpu();
}

void LSTM::backward_gpu(const SharedStorage& values,
                        const SharedStorage& grad_in, SharedStorage& grad_out) {
    int sz = grad_in->get_cols();
    compute_deriv_cell(states[5]->get_rows(), states[5]->get_cols(),
                       states[1]->gpu_pointer_const(),
                       states[5]->gpu_pointer());
    construct_sigma_gpu();
    for (int t = sz - 1; t >= 0; --t) {
        new_hidden_state(t, grad_in);
        new_cell_state(t, states[5]);
        internal_deriv(t);
        multiply_one_col_bwd(grad_out, t);
    };
    para_gradients(values);
}
// assistance_parameters1: dcum_s, dcum_c, dh, dc, d_tmp;
void LSTM::backward_cpu(const SharedStorage& values,
                        const SharedStorage& grad_in, SharedStorage& grad_out) {
    int nh = _out.get();
    int sz = grad_in->get_cols();
    const Matrix& state = states[2]->return_data_const();
    const Matrix& cell = states[1]->return_data_const();
    const Matrix& funcs = states[0]->return_data_const();
    Matrix& sigma = states[4]->return_data();
    Matrix sigma_c =
        Matrix::Ones(nh, sz + 1).array() - cell.array().tanh().pow(2);
    construct_sigma_cpu();
    Matrix& dcum_s = assistance_parameters[1]->return_data();
    Matrix& dcum_c = assistance_parameters[2]->return_data();
    Matrix& dh = assistance_parameters[3]->return_data();
    Matrix& dc = assistance_parameters[4]->return_data();
    Matrix& d_tmp = assistance_parameters[5]->return_data();
    Matrix& d_all = states[3]->return_data();
    dcum_s.setZero();
    dcum_c.setZero();
    for (int t = sz - 1; t >= 0; --t) {
        const Matrix& i = funcs.block(0, t, nh, 1);
        const Matrix& f = funcs.block(nh, t, nh, 1);
        const Matrix& o = funcs.block(2 * nh, t, nh, 1);
        const Matrix& g = funcs.block(3 * nh, t, nh, 1);
        dh = grad_in->return_data_const()(all, t) + dcum_s;
        dc = dcum_c.array() +
             dh.array() * o.array() * sigma_c(all, t + 1).array();
        d_tmp.block(2 * nh, 0, nh, 1) =
            dh.array() * cell(all, t + 1).array().tanh();
        d_tmp.block(0, 0, nh, 1) = dc.array() * g.array();
        d_tmp.block(nh, 0, nh, 1) = dc.array() * cell(all, t).array();
        d_tmp.block(3 * nh, 0, nh, 1) = dc.array() * i.array();
        d_all(all, t) = sigma(all, t).array() * d_tmp.array();
        dcum_s = parameters[1]->return_data_const().transpose() * d_all(all, t);
        grad_out->return_data()(all, t) =
            parameters[0]->return_data_const().transpose() * d_all(all, t);
        dcum_c = dc.array() * f.array();
    }
    gradients[0]->return_data() =
        d_all * values->return_data_const().transpose();
    gradients[1]->return_data() = d_all * state.leftCols(sz).transpose();
    gradients[2]->return_data() = d_all.rowwise().sum();
    clip_gradients();
}

void LSTM::expand_states(int cols) {
    states[0] = std::make_shared<Storage>(Matrix::Zero(4 * _out.get(), cols));
    states[1] = std::make_shared<Storage>(Matrix::Zero(_out.get(), cols + 1));
    states[2] = std::make_shared<Storage>(Matrix::Zero(_out.get(), cols + 1));
    states[3] = std::make_shared<Storage>(Matrix::Zero(4 * _out.get(), cols));
    states[4] = std::make_shared<Storage>(Matrix::Zero(4 * _out.get(), cols));
    states[5] = std::make_shared<Storage>(Matrix::Zero(_out.get(), cols + 1));
}

void LSTM::maybe_resize_state(int cols) {
    if (cols != states[0]->get_cols()) reorganize_states(cols);
}

void LSTM::clip_gradients() {
    for (SharedStorage& sgrad : gradients) {
        Matrix& grad = sgrad->return_data();
        for (int col = 0; col < grad.cols(); ++col) {
            for (int row = 0; row < grad.rows(); ++row) {
                if (grad(row, col) > 5)
                    grad(row, col) = 5;
                else if (grad(row, col) < -5)
                    grad(row, col) = -5;
            }
        }
    }
}

void LSTM::reorganize_states(int cols) {
    Matrix initial_cell =
        states[1]->return_data_const()(all, states[1]->get_cols() - 1);
    Matrix initial_state =
        states[2]->return_data_const()(all, states[2]->get_cols() - 1);
    expand_states(cols);
    states[1]->return_data()(all, states[1]->get_cols() - 1) =
        initial_cell(all, 0);
    states[2]->return_data()(all, states[2]->get_cols() - 1) =
        initial_state(all, 0);
}

void LSTM::initialize_grad() {
    Matrix wx = Matrix::Zero(4 * _out.get(), _in.get());
    Matrix wh = Matrix::Zero(4 * _out.get(), _out.get());
    Matrix bias_tmp = Matrix::Zero(4 * _out.get(), 1);
    gradients.push_back(std::make_shared<Storage>(wx));
    gradients.push_back(std::make_shared<Storage>(wh));
    gradients.push_back(std::make_shared<Storage>(bias_tmp));
}

void LSTM::initialize_weight(Init* init) {
    Matrix wx = init->weights(4 * _out.get(), _in.get());
    Matrix wh = init->weights(4 * _out.get(), _out.get());
    Matrix b = init->weights(4 * _out.get(), 1);
    parameters.push_back(std::make_shared<Storage>(wx));
    parameters.push_back(std::make_shared<Storage>(wh));
    parameters.push_back(std::make_shared<Storage>(b));
    Matrix tmp(Matrix::Zero(4 * _out.get(), 1));
    // Matrix bias(Matrix::Ones(4 * _out.get(), 1));
    assistance_parameters.push_back(std::make_shared<Storage>(tmp));
    assistance_parameters.push_back(
        std::make_shared<Storage>(Matrix::Zero(_out.get(), 1)));  // dcum_s;
    assistance_parameters.push_back(
        std::make_shared<Storage>(Matrix::Zero(_out.get(), 1)));  // dcum_c;
    assistance_parameters.push_back(
        std::make_shared<Storage>(Matrix::Zero(_out.get(), 1)));  // dh;
    assistance_parameters.push_back(
        std::make_shared<Storage>(Matrix::Zero(_out.get(), 1)));  // dc;
    assistance_parameters.push_back(
        std::make_shared<Storage>(Matrix::Zero(4 * _out.get(), 1)));  // d_tmp;
    assistance_parameters.push_back(
        std::make_shared<Storage>(Matrix::Ones(4 * _out.get(), 1)));  // d_tmp;
}
