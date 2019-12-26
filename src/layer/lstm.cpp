#include "../../include/layer/lstm.hpp"
#include <memory>
#include "../../include/math.h"
using Eigen::all;

LSTM::LSTM(Features out, Features in, Init* init)
    : Layer("LSTM"), _out(out), _in(in), states(3), assistance_parameters(0) {
    _previous = NULL;
    cublasStatus_t stat = cublasCreate(&_handle);
    CHECK_CUBLAS(stat);
    initialize_weight(init);
    initialize_grad();
    initialize_output_dimension();
    initialize_states();
}

LSTM::LSTM(Features out, const std::shared_ptr<Layer>& previous, Init* init)
    : Layer("LSTM"), _out(out), _in(0), states(3), assistance_parameters(0) {
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
    // batch size guess
    // states[1]->update_cpu_data(Matrix::Zero(_out.get(), cols + 1));
    // states[2]->update_cpu_data(Matrix::Zero(_out.get(), cols + 1));
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

void LSTM::forward_gpu(const SharedStorage& in, SharedStorage& out,
                       const std::string&){};

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
        // funcs.block(0, t, sigmoid_rows, 1));
        funcs.block(sigmoid_rows, t, _out.get(), 1) =
            tmp.bottomRows(_out.get()).array().tanh();
        const Matrix& i = funcs.block(0, t, _out.get(), 1);
        const Matrix& f = funcs.block(_out.get(), t, _out.get(), 1);
        const Matrix& o = funcs.block(2 * _out.get(), t, _out.get(), 1);
        const Matrix& g = funcs.block(3 * _out.get(), t, _out.get(), 1);
        cell(all, t + 1) =
            f.array() * cell(all, t).array() + i.array() * g.array();
        state(all, t + 1) = o.array() * state(all, t + 1).array().tanh();
    }
    out->return_data() = state.rightCols(cols - 1);
}

void LSTM::construct_sigma_cpu(Matrix& sigma) {
    const Matrix& state = states[0]->return_data_const();
    int n_sig = 3 * _out.get();
    sigma.topRows(n_sig) = state.topRows(n_sig).array() *
                           (Matrix::Ones(n_sig, sigma.cols()).array() -
                            state.topRows(n_sig).array());
    sigma.bottomRows(_out.get()) =
        Matrix::Ones(_out.get(), sigma.cols()).array() -
        state.bottomRows(_out.get()).array().pow(2);
}

void LSTM::backward_gpu(const SharedStorage& values,
                        const SharedStorage& grad_in,
                        SharedStorage& grad_out){};

void LSTM::backward_cpu(const SharedStorage& values,
                        const SharedStorage& grad_in, SharedStorage& grad_out) {
    int nh = _out.get();
    int sz = grad_in->get_cols();
    const Matrix& state = states[2]->return_data_const();
    const Matrix& cell = states[1]->return_data_const();
    const Matrix& funcs = states[0]->return_data_const();
    Matrix sigma(Matrix::Zero(_out.get() * 4, sz));
    Matrix sigma_c =
        Matrix::Ones(_out.get(), sz + 1).array() - cell.array().tanh().pow(2);
    construct_sigma_cpu(sigma);
    // grad_out.setZero();
    Vector dcum_s = Vector::Zero(_out.get());
    Vector dcum_c = Vector::Zero(_out.get());
    Vector dh = Vector::Zero(_out.get());
    Matrix dc = Vector::Zero(_out.get());
    Matrix d_tmp = Matrix::Zero(4 * _out.get(), 1);
    Matrix d_all = Matrix::Zero(4 * _out.get(), sz);
    for (int t = sz - 1; t >= 0; --t) {
        const Matrix& i = funcs.block(0, t, _out.get(), 1);
        const Matrix& f = funcs.block(nh, t, nh, 1);
        const Matrix& o = funcs.block(2 * nh, t, nh, 1);
        const Matrix& g = funcs.block(3 * nh, t, nh, 1);
        dh = grad_in->return_data_const()(all, t) + dcum_s;
        d_tmp.block(2 * nh, 0, nh, 1) =
            dh.array() * cell(all, t + 1).array().tanh();
        dc = dcum_c.array() +
             dh.array() * o.array() * sigma_c(all, t + 1).array();
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
}

void LSTM::expand_states(int cols) {
    states[0] = std::make_shared<Storage>(Matrix::Zero(4 * _out.get(), cols));
    states[1] = std::make_shared<Storage>(Matrix::Zero(_out.get(), cols + 1));
    states[2] = std::make_shared<Storage>(Matrix::Zero(_out.get(), cols + 1));
}

void LSTM::maybe_resize_state(int cols) {
    if (cols != states[0]->get_cols()) {
        // if (states[2]->get_cols() > 0)
        reorganize_states(cols);
        // else
        // expand_states(cols);
        //_batch_size = cols;
    }
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
    assistance_parameters.push_back(std::make_shared<Storage>(tmp));
}
