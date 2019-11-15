#include "../include/network.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include "../include/loss/cross_entropy.h"
using std::vector;

NeuralNetwork::NeuralNetwork(vector<Layer*> _layers,
                             std::shared_ptr<Loss> _loss)
    : layers(_layers), loss(_loss) {
    // create_loss(_loss);
    fun_forward = &NeuralNetwork::forward_gpu;
    fun_backward = &NeuralNetwork::backward_gpu;
    fun_update = &NeuralNetwork::update_weights_gpu;
};

NeuralNetwork::NeuralNetwork(vector<Layer*> _layers,
                             std::shared_ptr<Loss> _loss,
                             const std::string& device)
    : layers(_layers), loss(_loss) {
    if (device == "GPU") {
        fun_forward = &NeuralNetwork::forward_gpu;
        fun_backward = &NeuralNetwork::backward_gpu;
        fun_update = &NeuralNetwork::update_weights_gpu;
    } else {
        fun_forward = &NeuralNetwork::forward_cpu;
        fun_backward = &NeuralNetwork::backward_cpu;
        fun_update = &NeuralNetwork::update_weights_cpu;
    }
};

void NeuralNetwork::allocate_storage(int obs, int& out_dim,
                                     std::vector<SharedStorage>& inp,
                                     const Layer* layer) {
    if (layer->name() == "Dense") {
        if (layer->input_dimension() != out_dim) {
            std::string m("Dimensions do not fit, in:\n");
            throw std::invalid_argument(m + __PRETTY_FUNCTION__);
        }
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
    inp.push_back(std::make_shared<Storage>(Matrix::Zero(out_dim, obs)));
}

vector<SharedStorage> NeuralNetwork::allocate_forward(int obs) {
    vector<SharedStorage> vals;
    int out_dim(0);
    for (const Layer* layer : layers) {
        allocate_storage(obs, out_dim, vals, layer);
    }
    return vals;
}

vector<SharedStorage> NeuralNetwork::allocate_backward(int obs) {
    vector<SharedStorage> vals;
    int out_dim(0);
    for (size_t i = 0; i < layers.size() - 1; i++) {
        allocate_storage(obs, out_dim, vals, layers[i]);
    }
    return vals;
}

void NeuralNetwork::fill_hiddens(vector<SharedStorage>& values,
                                 const Matrix& features) {
    values[0]->update_cpu_data(features);
}

void NeuralNetwork::forward(vector<SharedStorage>& values) {
    (this->*fun_forward)(values);
}

void NeuralNetwork::forward_gpu(vector<SharedStorage>& values) {
    int i = 0;
    for (size_t layer_idx = 1; layer_idx < layers.size(); ++layer_idx) {
        layers[layer_idx]->forward_gpu(values[i], values[i + 1]);
        i++;
    }
}

void NeuralNetwork::forward_cpu(vector<SharedStorage>& values) {
    int i = 0;
    for (size_t layer_idx = 1; layer_idx < layers.size(); ++layer_idx) {
        layers[layer_idx]->forward_cpu(values[i], values[i + 1]);
        i++;
    }
}

Matrix NeuralNetwork::predict(const Matrix& input) {
    //std::cout <<input << std::endl;
    SharedStorage inp = std::make_shared<Storage>(input);
    vector<SharedStorage> vals = allocate_forward(input.rows());
    fill_hiddens(vals, input.transpose());
    forward(vals);
    return vals[vals.size() - 1]->return_data().transpose();
}
