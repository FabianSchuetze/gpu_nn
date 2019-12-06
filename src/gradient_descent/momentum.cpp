#include "../../include/gradient_descent/momentum.hpp"
#include <iostream>
#include "../../include/common.h"
#include "../../include/cuda_math.h"
#include "../../include/math.h"
#include "../../include/neural_network.h"

Momentum::Momentum(dtype _learning_rate, dtype momentum)
    : GradientDescent(_learning_rate, "Momentum"),
      _momentum(momentum) {};

Momentum::~Momentum() { ; };

void Momentum::initialize_gradients(const VecSharedStorage& gradients,
                                    VecSharedStorage& helper) {
    for (SharedStorage storage : gradients) {
        Matrix init = Matrix::Zero(storage->get_rows(), storage->get_cols());
        SharedStorage tmp = std::make_shared<Storage>(init);
        helper.push_back(tmp);
    }
}

void Momentum::weight_update_cpu(const VecSharedStorage& curr,
                                 VecSharedStorage& parameters, int batch_size,
                                 VecSharedStorage& helper) {
    //if (helper.size() == 0) {
        //initialize_gradients(curr, helper);
    //}
    //if (helper.size() != curr.size()) {
        //throw std::runtime_error("Size doesnt fit");
    //}
    dtype effective_learing_rate = learing_rate / batch_size;
    for (size_t i = 0; i < parameters.size(); ++i) {
        Matrix tmp = _momentum * helper[i]->return_data_const() -
                     effective_learing_rate * curr[i]->return_data_const();
        helper[i]->update_cpu_data(tmp);
        Matrix new_weight =
            parameters[i]->return_data_const() + helper[i]->return_data_const();
        parameters[i]->update_cpu_data(new_weight);
    }
}

void Momentum::weight_update_gpu(const VecSharedStorage& curr,
                                 VecSharedStorage& parameters, int batch_size,
                                 VecSharedStorage& helper) {
    //if (helper.size() == 0) {
        //initialize_gradients(curr, helper);
    //}
    //if (helper.size() != curr.size()) {
        //throw std::runtime_error("Size doesnt fit");
    //}
    dtype effective_learing_rate = learing_rate / batch_size;
    for (size_t i = 0; i < parameters.size(); ++i) {
        SharedStorage& para = parameters[i];
        dtype alpha_A = _momentum;
        dtype alpha_B = -1 * effective_learing_rate;
        my_Matrix_addition(helper[i], curr[i], helper[i], alpha_A, alpha_B);
        dtype alpha = 1;
        // const SharedStorage& grad = gradients[i];
        my_Matrix_addition_inplace(helper[i], para, alpha);
    }
}
