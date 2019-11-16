#include "../../include/gradient_descent/gradient_descent.h"

GradientDescent::GradientDescent(dtype _learning_rate)
    : learing_rate(_learning_rate){};
void GradientDescent::weight_update_cpu(const VecSharedStorage&,
                                        VecSharedStorage&, int) {
    ;
};
void GradientDescent::weight_update_gpu(const VecSharedStorage&,
                                        VecSharedStorage&, int) {
    ;
};
