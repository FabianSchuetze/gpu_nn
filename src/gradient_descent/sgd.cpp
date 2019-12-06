#include "../../include/gradient_descent/sgd.h"
#include <iostream>
#include "../../include/common.h"
#include "../../include/layer/layer.h"
#include "../../include/math.h"
#include "../../include/storage.h"

StochasticGradientDescent::StochasticGradientDescent(dtype _learning_rate)
    : GradientDescent(_learning_rate, "SGD"){};

StochasticGradientDescent::~StochasticGradientDescent() { ; };

void StochasticGradientDescent::weight_update_cpu(
    const VecSharedStorage& gradients, VecSharedStorage& parameters,
    int batch_size, VecSharedStorage&) {
    dtype effective_learing_rate = learing_rate / batch_size;
    for (size_t i = 0; i < parameters.size(); ++i) {
        Matrix new_weight =
            parameters[i]->return_data_const() -
            effective_learing_rate * gradients[i]->return_data_const();
        parameters[i]->update_cpu_data(new_weight);
    }
}

void StochasticGradientDescent::weight_update_gpu(
    const VecSharedStorage& gradients, VecSharedStorage& parameters,
    int batch_size, VecSharedStorage&) {
    dtype effective_learing_rate = learing_rate / batch_size;
    for (size_t i = 0; i < parameters.size(); ++i) {
        SharedStorage& para = parameters[i];
        const SharedStorage& grad = gradients[i];
        dtype alpha = -1 * effective_learing_rate;
        my_Matrix_addition_inplace(grad, para, alpha);
    }
}
