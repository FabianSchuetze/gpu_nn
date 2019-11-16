#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include "../include/loss/cross_entropy.h"
#include "../include/network.h"

using Eigen::all;
using std::vector;
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

void NeuralNetwork::backwards(std::vector<SharedStorage>& gradients,
                              const std::vector<SharedStorage>& values) {
    (this->*fun_backward)(gradients, values);
}

void NeuralNetwork::backward_cpu(std::vector<SharedStorage>& gradients,
                                 const std::vector<SharedStorage>& values) {
    int idx = gradients.size() - 1;
    for (int i = layers.size() - 2; i > 0; i--) {
        const SharedStorage& gradient_in = gradients[idx];
        SharedStorage& gradient_out = gradients[idx - 1];
        const SharedStorage& vals = values[idx - 1];
        layers[i]->backward_cpu(vals, gradient_in, gradient_out);
        idx--;
    }
}

void NeuralNetwork::backward_gpu(vector<SharedStorage>& gradients,
                                 const vector<SharedStorage>& values) {
    int idx = gradients.size() - 1;
    for (int i = layers.size() - 2; i > 0; i--) {
        const SharedStorage& gradient_in = gradients[idx];
        SharedStorage& gradient_out = gradients[idx - 1];
        const SharedStorage& vals = values[idx - 1];
        layers[i]->backward_gpu(vals, gradient_in, gradient_out);
        idx--;
    }
}

void NeuralNetwork::update_weights(std::shared_ptr<GradientDescent> opt,
                                   int batch_size) {
    (this->*fun_update)(opt, batch_size);
}

void NeuralNetwork::update_weights_cpu(std::shared_ptr<GradientDescent> opt,
                                       int batch_size) {
    for (Layer* layer : layers) {
        if (layer->n_paras() > 0) {
            vector<SharedStorage> parameters = layer->return_parameters();
            const vector<SharedStorage>& gradients = layer->return_gradients();
            opt->weight_update_cpu(gradients, parameters, batch_size);
            layer->clear_gradients_cpu();
        }
    }
}

void NeuralNetwork::update_weights_gpu(std::shared_ptr<GradientDescent> opt,
                                       int batch_size) {
    for (Layer* layer : layers) {
        if (layer->n_paras() > 0) {
            vector<SharedStorage> parameters = layer->return_parameters();
            const vector<SharedStorage>& gradients = layer->return_gradients();
            opt->weight_update_gpu(gradients, parameters, batch_size);
            layer->clear_gradients_cpu();
        }
    }
}

void NeuralNetwork::random_numbers(vector<int>& samples, std::mt19937& gen) {
    if (!train_args) throw std::runtime_error("Train args is not set");
    std::uniform_int_distribution<> uniform(0,
                                            train_args->x_train().rows() - 1);
    for (size_t i = 0; i < samples.size(); i++) samples[i] = uniform(gen);
}

void NeuralNetwork::get_new_sample(const vector<int>& samples, Matrix& x_train,
                                   Matrix& y_train) {
    if (!train_args) throw std::runtime_error("Train args is not set");
    x_train = train_args->x_train()(samples, all).transpose();
    y_train = train_args->y_train()(samples, all).transpose();
}

void NeuralNetwork::train(const Matrix& features, const Matrix& targets,
                          std::shared_ptr<GradientDescent> sgd, Epochs _epoch,
                          Patience _patience, BatchSize _batch_size) {
    if (layers[0]->output_dimension() != features.cols()) {
        std::string m("N of input features != col in features, in:\n");
        throw std::invalid_argument(m + __PRETTY_FUNCTION__);
    }
    //if (layers.back()->output_dimension() != targets.cols()) {
        //std::string m("N of output layer dim != col in targets, in:\n");
        //throw std::invalid_argument(m + __PRETTY_FUNCTION__);
    //}
    train_args = std::make_unique<trainArgs>(features, targets, _epoch,
                                             _patience, _batch_size);
    train(sgd);
}

void NeuralNetwork::validate(std::chrono::milliseconds diff) {
    int obs = train_args->x_val().rows();
    vector<SharedStorage> vals = allocate_forward(obs);
    SharedStorage SharedTarget =
        std::make_shared<Storage>(train_args->y_val().transpose());
    fill_hiddens(vals, train_args->x_val().transpose());
    forward(vals);
    const SharedStorage& prediction = vals[vals.size() - 1];
    //std::cout << prediction->return_data_const() << std::endl;
    dtype total_loss = loss->loss_cpu(prediction, SharedTarget);
    std::cout << "after iter " << train_args->current_epoch() << "the loss is "
              << total_loss / obs << ", in " << diff.count() << " milliseconds"
              << std::endl;
    train_args->advance_epoch();
    train_args->reset_total_iter();
}

void NeuralNetwork::train(std::shared_ptr<GradientDescent> sgd) {
    std::mt19937 gen;
    gen.seed(0);
    vector<SharedStorage> vals = allocate_forward(train_args->batch_size());
    vector<SharedStorage> grads = allocate_backward(train_args->batch_size());
    Matrix tmp =
        Matrix::Zero(train_args->y_train().cols(), train_args->batch_size());
    SharedStorage SharedTarget = std::make_shared<Storage>(tmp);
    vector<int> samples(train_args->batch_size());
    Matrix x_train, y_train;
    auto begin = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    std::chrono::milliseconds diff;
    while (train_args->current_epoch() < train_args->epochs()) {
        random_numbers(samples, gen);
        get_new_sample(samples, x_train, y_train);
        SharedTarget->update_cpu_data(y_train);
        fill_hiddens(vals, x_train);
        forward(vals);
        //SharedStorage& grad_in = grads[grads.size() - 1];
        loss->grad_loss(grads.back(), vals.back(), SharedTarget, SharedTarget);
        backwards(grads, vals);
        //std::cout << grads[grads.size() - 1]->return_data_const() << std::endl;

        update_weights(sgd, train_args->batch_size());
        //std::cout << "UPDATING\n";
        train_args->advance_total_iter();
        if (train_args->total_iter() > train_args->max_total_iter()) {
            end = std::chrono::system_clock::now();
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                         begin);
            validate(diff);
            begin = std::chrono::system_clock::now();
        }
    }
}
