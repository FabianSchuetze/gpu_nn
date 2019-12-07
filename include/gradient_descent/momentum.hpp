#pragma once
#ifndef momentum_h
#define momentum_h
#include <memory>
#include <vector>
#include "gradient_descent.h"
class Momentum : public GradientDescent {
   public:
    Momentum(LearningRate, MomentumRate, WeightDecay = WeightDecay(0.));
    virtual ~Momentum();
    void weight_update_cpu(const VecSharedStorage&, VecSharedStorage&,
                           int, VecSharedStorage&) override;
    void weight_update_gpu(const VecSharedStorage&, VecSharedStorage&,
                           int, VecSharedStorage&) override;
   private:
    void initialize_gradients(const VecSharedStorage&, VecSharedStorage&);
    MomentumRate momentum;
};
#endif
