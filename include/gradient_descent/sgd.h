#pragma once
#ifndef stochastic_gradient_descent_h
#define stochastic_gradient_descent_h
#include <memory>
#include <vector>
#include "gradient_descent.h"
class StochasticGradientDescent : public GradientDescent {
   public:
    StochasticGradientDescent(LearningRate, WeightDecay = WeightDecay(0.));
    virtual ~StochasticGradientDescent();
    void weight_update_cpu(const VecSharedStorage&, VecSharedStorage&,
                           int, VecSharedStorage&) override;
    void weight_update_gpu(const VecSharedStorage&, VecSharedStorage&,
                           int, VecSharedStorage&) override;
};
#endif
