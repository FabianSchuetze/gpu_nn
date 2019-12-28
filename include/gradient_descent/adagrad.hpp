#pragma once
#ifndef adagrad_hpp
#define adagrad_hpp
#include <memory>
#include <vector>
#include "gradient_descent.h"
class AdaGrad : public GradientDescent {
   public:
    AdaGrad(LearningRate, WeightDecay = WeightDecay(0.));
    virtual ~AdaGrad();
    void weight_update_cpu(const VecSharedStorage&, VecSharedStorage&,
                           int, VecSharedStorage&) override;
    void weight_update_gpu(const VecSharedStorage&, VecSharedStorage&,
                           int, VecSharedStorage&) override;
   private:
    //void initialize_gradients(const VecSharedStorage&, VecSharedStorage&);
};
#endif
