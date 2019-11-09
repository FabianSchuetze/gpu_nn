#include "../../include/gradient_descent/sgd.h"
#include "../../include/layer/layer.h"
#include "../../include/math.h"

StochasticGradientDescent::StochasticGradientDescent(int _learning_rate)
    : GradientDescent(_learning_rate){};

void StochasticGradientDescent::weight_update_cpu(Layer* layer) {
    for (size_t i = 0; i < layer->return_parameters().size(); ++i) {
        SharedStorage& para = layer->return_parameters()[i];
        SharedStorage& grad = layer->return_parameters()[i];
        Matrix new_weight = para->return_data_const() -
                            learing_rate * grad->return_data_const();
        para->update_cpu_data(new_weight);
        layer->clear_gradients_cpu();
    }
}

void StochasticGradientDescent::weight_update_gpu(Layer* layer) {
    for (size_t i = 0; i < layer->return_parameters().size(); ++i) {
        SharedStorage& para = layer->return_parameters()[i];
        const SharedStorage& grad = layer->return_parameters()[i];
        dtype alpha = -1 * learing_rate;
        my_Matrix_addition_inplace(grad, para, alpha);
        layer->clear_gradients_gpu();
    }
}
