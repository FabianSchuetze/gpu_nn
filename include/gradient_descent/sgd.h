#pragma once
#ifndef stochastic_gradient_descent_h
#define stochastic_gradient_descent_h
#include <memory>
#include <vector>
#include "gradient_descent.h"
class StochasticGradientDescent : public GradientDescent {
   public:
    StochasticGradientDescent(dtype);
    virtual ~StochasticGradientDescent();
    void weight_update_cpu(const std::vector<SharedStorage>&
        gradient, std::vector<SharedStorage>& parameters) override;
    void weight_update_gpu(const std::vector<SharedStorage>&
        gradient, std::vector<SharedStorage>& parameters) override;
};
#endif
