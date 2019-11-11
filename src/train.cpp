#include <iostream>
#include <memory>
#include "../include/loss/cross_entropy.h"
#include "../include/network.h"

using std::vector;

void NeuralNetwork::backwards(std::vector<SharedStorage>& gradients,
                              const std::vector<SharedStorage>& values) {
    (this->*fun_backward)(gradients, values);
}

void NeuralNetwork::backward_cpu(std::vector<SharedStorage>& gradients,
                                 const std::vector<SharedStorage>& values) {
    int idx = gradients.size() - 1;
    //std::cout << "inside backward cpu\n";
    for (int i = layers.size() - 2; i > 0; i--) {
        //std::cout << layers[i]->name() << std::endl;
        //std::cout << "number " << idx << std::endl;
        const SharedStorage& gradient_in = gradients[idx] ;
        SharedStorage& gradient_out = gradients[idx - 1];
        const SharedStorage& vals = values[idx - 1];
        layers[i]->backward_cpu(vals, gradient_in, gradient_out);
        //std::cout << "gradient at " << layers[i]->name() << ":\n"
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
