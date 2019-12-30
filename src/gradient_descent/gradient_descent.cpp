#include "../../include/gradient_descent/gradient_descent.h"
#include <iostream>

GradientDescent::GradientDescent(LearningRate _learning_rate,
                                 WeightDecay _weight_decay,
                                 LearingRateDecay _lr_decay)
    : learing_rate(_learning_rate),
      weight_decay(_weight_decay),
      lr_decay(_lr_decay),
      _name("GradientDescent"){};

GradientDescent::GradientDescent(LearningRate _learning_rate,
                                 const std::string& name,
                                 WeightDecay _weight_decay,
                                 LearingRateDecay _lr_decay)
    : learing_rate(_learning_rate),
      weight_decay(_weight_decay),
      lr_decay(_lr_decay),
      _name(name){};

void GradientDescent::weight_update_cpu(const VecSharedStorage&,
                                        VecSharedStorage&, int,
                                        VecSharedStorage&) {
    ;
};
void GradientDescent::weight_update_gpu(const VecSharedStorage&,
                                        VecSharedStorage&, int,
                                        VecSharedStorage&) {
    ;
};

void GradientDescent::learning_rate_decay(int curr_epoch) {
    if ((curr_epoch >= lr_decay.first()) and (lr_decay.second() < 1) ) {
        dtype decay = lr_decay.second();
        learing_rate.get() *= decay;
        std::cout << "the learning rate has been reduced to: "
                  << learing_rate.get() << std::endl;
    }
}
