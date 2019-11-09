#pragma once
#ifndef stochastic_gradient_descent_h
#define stochastic_gradient_descent_h
#include <memory>
#include <vector>
#include "gradient_descent.h"
class StochasticGradientDescent : public GradientDescent {
   public:
    StochasticGradientDescent(int);
    ~StochasticGradientDescent() = default;
    void weight_update_cpu(Layer* layer) override;
    void weight_update_gpu(Layer* layer) override;
};
#endif
