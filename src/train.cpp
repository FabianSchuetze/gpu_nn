#include <iostream>
#include <memory>
#include "../include/loss/cross_entropy.h"
#include "../include/network.h"

using std::vector;

void NeuralNetwork::backward_cpu(std::vector<SharedStorage>& gradients,
                                 const std::vector<SharedStorage>& values) {
    int tmp = 0;
    int idx = gradients.size() - 1;
    std::cout << "inside backward cpu\n";
    for (int i = layers.size() - 2; i > 0; i--) {
        std::cout << layers[i]->name() << std::endl;
        std::cout << "number " << idx << std::endl;
        const SharedStorage& gradient_in = gradients[idx];
        SharedStorage& gradient_out = gradients[idx - 1];
        const SharedStorage& vals = values[idx - 2];
        layers[i]->backward_cpu(tmp, vals, gradient_in, gradient_out);
        std::cout << "gradient at " << layers[i]->name() << ":\n"
                  << gradient_out->return_data_const() << std::endl;
        idx--;
    }
}
void NeuralNetwork::backwards(std::vector<SharedStorage>& gradients,
                                 const std::vector<SharedStorage>& values) {
    (this->*fun_backward)(gradients, values);
}
void NeuralNetwork::backward_gpu(vector<SharedStorage>& gradients,
                              const vector<SharedStorage>& values) {
    int tmp = 0;
    int idx = gradients.size() - 1;
    std::cout << "inside backward gpu\n";
    for (int i = layers.size() - 2; i > 0; i--) {
        std::cout << layers[i]->name() << std::endl;
        std::cout << "number " << idx << std::endl;
        const SharedStorage& gradient_in = gradients[idx];
        SharedStorage& gradient_out = gradients[idx - 1];
        const SharedStorage& vals = values[idx - 2];
        layers[i]->backward_gpu(tmp, vals, gradient_in, gradient_out);
        std::cout << "gradient at " << layers[i]->name() << ":\n"
                  << gradient_out->return_data_const() << std::endl;
        idx--;
    }
}

