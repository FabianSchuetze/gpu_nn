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

void NeuralNetwork::random_numbers(vector<int>& samples, std::mt19937& gen) {
    std::uniform_int_distribution<> uniform(0, train_args.x_train().rows() - 1);
    for (size_t i = 0; i < samples.size(); i++) samples[i] = uniform(gen);
}

void NeuralNetwork::get_new_sample(const vector<int>& samples, Matrix& x_train,
                                   Matrix& y_train) {
    x_train = train_args.x_train()(samples, all).transpose();
    y_train = train_args.y_train()(samples, all).transpose();
}

void NeuralNetwork::train(const Matrix& features, const Matrix& targets,
                          std::shared_ptr<GradientDescent> sgd, Epochs _epoch,
                          Patience _patience, BatchSize _batch_size) {
    train_args = trainArgs(features, targets, _epoch, _patience, _batch_size);
    train(sgd);
}

void NeuralNetwork::validate() {
    vector<SharedStorage> vals = allocate_forward(train_args.x_val().rows());
    Matrix tmp = Matrix::Zero(train_args.y_train().cols(), train_args.batch_size());
    SharedStorage SharedTarget = std::make_shared<Storage>(tmp);
    fill_hiddens(vals, train_args.x_val());
    forward(vals);
    const SharedStorage& prediction = vals[vals.size() - 1];
    dtype total_loss = loss->loss_cpu(prediction, SharedTarget);
    std::cout << "after iter " << train_args.current_epoch() << "the loss is "
              << total_loss << std::endl;
    train_args.advance_epoch();
    train_args.reset_total_iter();
}

void NeuralNetwork::train(std::shared_ptr<GradientDescent> sgd) {
    std::mt19937 gen;
    gen.seed(0);
    vector<SharedStorage> vals = allocate_forward(train_args.batch_size());
    vector<SharedStorage> grads = allocate_backward(train_args.batch_size());
    Matrix tmp = Matrix::Zero(train_args.y_train().cols(), train_args.batch_size());
    SharedStorage SharedTarget = std::make_shared<Storage>(tmp);
    vector<int> samples(train_args.batch_size());
    Matrix x_train, y_train;
    auto begin = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    std::chrono::seconds diff;
    while (train_args.current_epoch() < train_args.epochs()) {
        random_numbers(samples, gen);
        get_new_sample(samples, x_train, y_train);
        SharedTarget->update_cpu_data(y_train);
        fill_hiddens(vals, x_train);
        forward(vals);
        SharedStorage& grad_in = grads[grads.size() - 1];
        loss->grad_loss_cpu(grad_in, vals[vals.size() - 1], SharedTarget,
                            SharedTarget);
        backwards(grads, vals);
        update_weights(sgd);
        // total_loss += loss->loss_cpu(vals[vals.size() - 1], SharedTarget);
        train_args.advance_total_iter();
        if (train_args.total_iter() > train_args.max_total_iter()) {
            validate();
        }
    }
}
