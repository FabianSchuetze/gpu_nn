#include "../../include/gradient_descent/momentum.hpp"
#include <iostream>
#include "../../include/common.h"
#include "../../include/cuda_math.h"
#include "../../include/math.h"
#include "../../include/neural_network.h"

Momentum::Momentum(LearningRate _learning_rate, MomentumRate _momentum,
                   WeightDecay _weight_decay)
    : GradientDescent(_learning_rate, "Momentum", _weight_decay),
      momentum(_momentum){};

Momentum::~Momentum() { ; };

//void Momentum::initialize_gradients(const VecSharedStorage& gradients,
                                    //VecSharedStorage& helper) {
    //for (SharedStorage storage : gradients) {
        //Matrix init = Matrix::Zero(storage->get_rows(), storage->get_cols());
        ////SharedStorage tmp = std::make_shared<Storage>(init);
        //helper.push_back(std::make_shared<Storage>(init));
    //}
//}

void Momentum::weight_update_cpu(const VecSharedStorage& curr,
                                 VecSharedStorage& parameters, int batch_size,
                                 VecSharedStorage& helper) {
    dtype effective_learing_rate = learing_rate.get() / batch_size;
    dtype lower = -1 * learing_rate.get() * weight_decay.get();
    for (size_t i = 0; i < parameters.size(); ++i) {
        Matrix tmp = momentum.get() * helper[i]->return_data_const() -
                     effective_learing_rate * curr[i]->return_data_const()
                     + lower * curr[i]->return_data_const();
        helper[i]->update_cpu_data(tmp);
        Matrix new_weight =
            parameters[i]->return_data_const() + helper[i]->return_data_const();
        parameters[i]->update_cpu_data(new_weight);
    }
}

void Momentum::weight_update_gpu(const VecSharedStorage& curr,
                                 VecSharedStorage& parameters, int batch_size,
                                 VecSharedStorage& helper) {
    //dtype effective_learing_rate = learing_rate / batch_size;
    dtype alpha_A = momentum.get();
    dtype alpha_B = -1 * learing_rate.get() / batch_size;
    dtype lower = -1 * learing_rate.get() * weight_decay.get();
    dtype alpha = 1;
    for (size_t i = 0; i < parameters.size(); ++i) {
        SharedStorage& para = parameters[i];
        my_Matrix_addition(helper[i], curr[i], helper[i], alpha_A, alpha_B);
        my_Matrix_addition_inplace(para, helper[i], lower);
        // const SharedStorage& grad = gradients[i];
        my_Matrix_addition_inplace(helper[i], para, alpha);
    }
}
