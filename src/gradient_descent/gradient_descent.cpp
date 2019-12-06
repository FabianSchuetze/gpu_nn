#include "../../include/gradient_descent/gradient_descent.h"

GradientDescent::GradientDescent(dtype _learning_rate)
    : learing_rate(_learning_rate), _name("GradientDescent") {};

GradientDescent::GradientDescent(dtype _learning_rate, std::string name)
    : learing_rate(_learning_rate), _name(name) {};

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
