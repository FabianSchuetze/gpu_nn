#pragma once
#ifndef gradient_descent_h
#define gradient_descent_h
#include <memory>
#include <vector>
#include "../../include/layer/layer.h"
#include "../common.h"
#include "../storage.h"
class GradientDescent {
   protected:
    typedef std::shared_ptr<Storage> SharedStorage;
    typedef std::vector<SharedStorage> VecSharedStorage;
    LearningRate learing_rate;
    WeightDecay weight_decay;
    LearingRateDecay lr_decay;
    std::string _name;

   public:
    GradientDescent(LearningRate, WeightDecay = WeightDecay(0.),
                             LearingRateDecay = LearingRateDecay(0, 1.0));
    GradientDescent(LearningRate, const std::string&,
                    WeightDecay = WeightDecay(0.),
                    LearingRateDecay = LearingRateDecay(0, 1.0));
    virtual ~GradientDescent(){};
    void learning_rate_decay(int);
    virtual void weight_update_cpu(const VecSharedStorage&, VecSharedStorage&,
                                   int, VecSharedStorage&);
    virtual void weight_update_gpu(const VecSharedStorage&, VecSharedStorage&,
                                   int, VecSharedStorage&);
    virtual const std::string& name() { return _name; }
};
#endif
