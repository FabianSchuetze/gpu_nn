#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include "../include/loss/cross_entropy.h"
#include "../include/network.h"

using Eigen::all;
using std::vector;

void NeuralNetwork::backwards(std::vector<SharedStorage>& gradients,
                              const std::vector<SharedStorage>& values) {
    (this->*fun_backward)(gradients, values);
}

void NeuralNetwork::backward_cpu(std::vector<SharedStorage>& gradients,
                                 const std::vector<SharedStorage>& values) {
    int idx = gradients.size() - 1;
    // std::cout << "inside backward cpu\n";
    for (int i = layers.size() - 2; i > 0; i--) {
        // std::cout << layers[i]->name() << std::endl;
        // std::cout << "number " << idx << std::endl;
        const SharedStorage& gradient_in = gradients[idx];
        SharedStorage& gradient_out = gradients[idx - 1];
        const SharedStorage& vals = values[idx - 1];
        layers[i]->backward_cpu(vals, gradient_in, gradient_out);
        // std::cout << "gradient at " << layers[i]->name() << ":\n"
        //<< gradient_out->return_data_const() << std::endl;
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

void NeuralNetwork::update_weights(std::shared_ptr<GradientDescent> opt) {
    (this->*fun_update)(opt);
}

void NeuralNetwork::update_weights_cpu(std::shared_ptr<GradientDescent> opt) {
    for (Layer* layer : layers) {
        if (layer->n_paras() > 0) {
            vector<SharedStorage> parameters = layer->return_parameters();
            const vector<SharedStorage>& gradients = layer->return_gradients();
            opt->weight_update_cpu(gradients, parameters);
            layer->clear_gradients_cpu();
        }
    }
}

void NeuralNetwork::update_weights_gpu(std::shared_ptr<GradientDescent> opt) {
    for (Layer* layer : layers) {
        if (layer->n_paras() > 0) {
            vector<SharedStorage> parameters = layer->return_parameters();
            const vector<SharedStorage>& gradients = layer->return_gradients();
            opt->weight_update_gpu(gradients, parameters);
            layer->clear_gradients_cpu();
        }
    }
}

void NeuralNetwork::random_numbers(vector<int>& samples, std::mt19937& gen,
                                   const Matrix& input) {
    std::uniform_int_distribution<> uniform(0, input.rows() - 1);
    for (size_t i = 0; i < samples.size(); i++) samples[i] = uniform(gen);
}

void NeuralNetwork::get_new_sample(const Matrix& input, const Matrix& targets,
                                   const vector<int>& samples, Matrix& x_train,
                                   Matrix& y_train) {
    x_train = input(samples, all).transpose();
    y_train = targets(samples, all).transpose();
}

void NeuralNetwork::train(const Matrix& features, const Matrix& targets,
                          std::shared_ptr<GradientDescent> sgd) {
    int total_iter(0);
    std::mt19937 gen;
    gen.seed(0);
    vector<SharedStorage> vals = allocate_forward(32);
    vector<SharedStorage> grads = allocate_backward(32);
    double total_loss(0.);
    Matrix x_train, y_train;
    Matrix tmp = Matrix::Zero(features.cols(), 32);
    SharedStorage SharedTarget = std::make_shared<Storage>(tmp);
    vector<int> samples(32);
    // MatrixVec parameters_bkp(parameters);
    auto begin = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    std::chrono::seconds diff;
    while (total_iter < 8 * features.rows()) {
        random_numbers(samples, gen, features);
        get_new_sample(features, targets, samples, x_train, y_train);
        SharedTarget->update_cpu_data(y_train);
        fill_hiddens(vals, x_train);
        SharedStorage& grad_in = grads[grads.size() - 1];
        loss->grad_loss_cpu(grad_in, vals[vals.size() - 1], SharedTarget,
                            SharedTarget);
        backwards(grads, vals);
        update_weights(sgd);
        total_loss += loss->loss_cpu(vals[vals.size() - 1], SharedTarget);
        total_iter += 32;
    }
}
