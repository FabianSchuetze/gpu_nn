#include "../../include/gradient_descent/gradient_descent.h"

GradientDescent::GradientDescent(LearningRate _learning_rate,
                                 WeightDecay _weight_decay)
    : learing_rate(_learning_rate),
      weight_decay(_weight_decay),
      _name("GradientDescent"){};

GradientDescent::GradientDescent(LearningRate _learning_rate,
                                 const std::string& name,
                                 WeightDecay _weight_decay)
    : learing_rate(_learning_rate), weight_decay(_weight_decay), _name(name){};

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
