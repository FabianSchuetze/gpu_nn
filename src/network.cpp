#include "../include/network.h"
#include <memory>
#include "../include/loss/cross_entropy.h"
#include <iostream>
using std::vector;
//typedef void(Layer::*forward_func)(const SharedStorage&, SharedStorage&);
NeuralNetwork::NeuralNetwork(vector<Layer*> _layers, const std::string& _loss)
    : layers(_layers) {
    create_loss(_loss);
    fun_forward = &NeuralNetwork::forward_gpu;
};

NeuralNetwork::NeuralNetwork(vector<Layer*> _layers, const std::string& _loss,
        const std::string& device)
    : layers(_layers) {
    create_loss(_loss);
    if (device == "GPU")
        fun_forward = &NeuralNetwork::forward_gpu;
    else
        fun_forward = &NeuralNetwork::forward_cpu;
    //forward_func a = &Layer::forward_gpu;
};

void NeuralNetwork::create_loss(const std::string& s) {
    if (s == "Bernoulli")
        ;
    // loss = std::make_shared<Bernoulli>();
    else if (s == "MSE")
        ;
    // loss = std::make_shared<MSE>();
    else if (s == "Categorical_Crossentropy")
        loss = std::make_shared<CrossEntropy>();
    else {
        // string m("Only Bernoulli, MSE and Categorical_Crossentropy, in:\n");
        // throw std::invalid_argument(m + __PRETTY_FUNCTION__);
    }
}

vector<SharedStorage> NeuralNetwork::allocate_shared_storage(int obs) {
    vector<SharedStorage> vals;
    int out_dim(0);
    for (const Layer* layer : layers) {
        if (layer->name() == "Dense") {
            out_dim = layer->output_dimension();
        } else if (layer->name() == "Activation")
            ;
        else if (layer->name() == "BatchNorm") {
            ;
        } else if (layer->name() == "Dropout") {
            ;
        } else if (layer->name() == "Input") {
            out_dim = layer->output_dimension();
        } else {
            std::string m("Cannot figure out name, in:\n");
            throw std::invalid_argument(m + __PRETTY_FUNCTION__);
        }
        vals.push_back(std::make_shared<Storage>(Matrix::Zero(out_dim, obs)));
    }
    return vals;
}

void NeuralNetwork::fill_hiddens(vector<SharedStorage>& values,
                                 const Matrix& features) {
    values[0]->copy_cpu_data(features.transpose());
}

void NeuralNetwork::forward(vector<SharedStorage>& values) {
    (this->*fun_forward)(values);
}

void NeuralNetwork::forward_gpu(vector<SharedStorage>& values) {
    int i = 0;
    for (size_t layer_idx = 1; layer_idx < layers.size(); ++layer_idx) {
        layers[layer_idx]->forward_gpu(values[i], values[i+1]);
        i++;
    }
}

void NeuralNetwork::forward_cpu(vector<SharedStorage>& values) {
    int i = 0;
    for (size_t layer_idx = 1; layer_idx < layers.size(); ++layer_idx) {
        layers[layer_idx]->forward_cpu(values[i], values[i+1]);
        i++;
    }
}

Matrix NeuralNetwork::predict(const Matrix& input) {
    SharedStorage inp = std::make_shared<Storage>(input);
    vector<SharedStorage> vals = allocate_shared_storage(input.rows());
    fill_hiddens(vals, input);
    forward(vals);
    return vals[vals.size() - 1]->return_data().transpose();
}
