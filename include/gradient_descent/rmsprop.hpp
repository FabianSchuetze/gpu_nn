#pragma once
#ifndef rmsprop_hpp
#define rmsprop_hpp
#include <memory>
#include <vector>
#include "gradient_descent.h"
class RMSProp : public GradientDescent {
   public:
    RMSProp(LearningRate, DecayRate, WeightDecay = WeightDecay(0.),
            LearingRateDecay = LearingRateDecay(0, 1.));
    virtual ~RMSProp();
    void weight_update_cpu(const VecSharedStorage&, VecSharedStorage&,
                           int, VecSharedStorage&) override;
    void weight_update_gpu(const VecSharedStorage&, VecSharedStorage&,
                           int, VecSharedStorage&) override;
   private:
    DecayRate decay;
};
#endif
