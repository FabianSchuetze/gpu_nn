#include "../../include/gradient_descent/sgd.h"
#include <iostream>
#include "../../include/common.h"
#include "../../include/layer/layer.h"
#include "../../include/math.h"
#include "../../include/storage.h"

StochasticGradientDescent::StochasticGradientDescent(
    LearningRate _learning_rate, WeightDecay _weight_decay)
    : GradientDescent(_learning_rate, "SGD", _weight_decay){};

StochasticGradientDescent::~StochasticGradientDescent() { ; };

void StochasticGradientDescent::weight_update_cpu(
    const VecSharedStorage& gradients, VecSharedStorage& parameters,
    int batch_size, VecSharedStorage&) {
    // dtype effective_learing_rate = learing_rate.get() / batch_size;
    for (size_t i = 0; i < parameters.size(); ++i) {
        const Matrix& curr = parameters[i]->return_data_const();
        Matrix new_weight = curr -
                            learing_rate.get() / batch_size *
                                gradients[i]->return_data_const() -
                            weight_decay.get() * learing_rate.get() * curr;
        parameters[i]->update_cpu_data(new_weight);
    }
}

void StochasticGradientDescent::weight_update_gpu(
    const VecSharedStorage& gradients, VecSharedStorage& parameters,
    int batch_size, VecSharedStorage&) {
    // dtype effective_learing_rate = learing_rate.get() / batch_size;
    for (size_t i = 0; i < parameters.size(); ++i) {
        SharedStorage& para = parameters[i];
        const SharedStorage& grad = gradients[i];
        dtype alpha_B = -1 * learing_rate.get() / batch_size;
        dtype alpha_A = -1 * learing_rate.get() * weight_decay.get();
        my_Matrix_addition(para, grad, para, alpha_A, alpha_B);
    }
}
