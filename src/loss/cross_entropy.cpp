#include "../../include/loss/cross_entropy.h"
#include <iomanip>
#include <iostream>
#include <string>
#include "../../include/common.h"
#include "../../include/math.h"
dtype CrossEntropy::loss_cpu(const SharedStorage& prediction,
                             const SharedStorage& actual) {
    const Matrix& pred = prediction->return_data_const();
    const Matrix& act = actual->return_data_const();
    if ((pred.rows() != act.rows()) or (pred.cols() != act.cols())) {
        std::string m("prediction must have the same shape a target, in:\n");
        throw std::runtime_error(m + __PRETTY_FUNCTION__);
    }
    dtype tot(0.);
    for (int i = 0; i < pred.cols(); i++) {
        tot += loss(Vector(pred.col(i)), Vector(act.col(i)));
    }
    return tot;
}

dtype CrossEntropy::loss_gpu(const SharedStorage& prediction,
                             const SharedStorage& actual) {
    dtype loss(0);
    my_cross_entropy_loss(loss, prediction, actual);
    return loss;
}

dtype CrossEntropy::loss(const Vector& pred, const Vector& actual) {
    dtype loss(0);
    dtype diff = pred.sum() - 1;
    const static dtype epsilon = 1e-5;
    if ((diff > epsilon) or (-diff > epsilon)) {
        std::string m("predictions don't sum to one, they are ");
        std::string m2{std::to_string(pred.sum()) + " in:\n"};
        throw std::runtime_error(m + m2 + __PRETTY_FUNCTION__);
    }
    for (int i = 0; i < pred.rows(); i++) {
        if (actual(i) == 1.) {
            loss = -1 * log(pred(i));
            break;
        }
    }
    return loss;
}
void CrossEntropy::grad_loss_cpu(SharedStorage& gradient,
                                 const SharedStorage& prediction,
                                 const SharedStorage& target,
                                 const SharedStorage&) {
    gradient->return_data() =
        prediction->return_data_const() - target->return_data_const();
}

void CrossEntropy::grad_loss_gpu(SharedStorage& gradient,
                                 const SharedStorage& prediction,
                                 const SharedStorage& target,
                                 const SharedStorage&) {
    my_cross_entropy_gradient(gradient, prediction, target);
}
