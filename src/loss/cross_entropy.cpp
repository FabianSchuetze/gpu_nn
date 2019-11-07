#include "../../include/loss/cross_entropy.h"
#include "../../include/common.h"
#include "../../include/math.h"

dtype CrossEntropy::loss_cpu(const SharedStorage& prediction,
                             const SharedStorage& actual) {
    const Matrix& pred = prediction->return_data_const();
    const Matrix& act = actual->return_data_const();
    dtype tot = 0;
    for (int i = 0; i < pred.cols(); i++) tot += loss(pred.col(i), act.col(i));
    return tot;
}

dtype CrossEntropy::loss_gpu(const SharedStorage& prediction,
                             const SharedStorage& actual) {
    //int obs = prediction->get_cols();
    dtype loss(0);
    my_cross_entropy_loss(loss, prediction, actual);
    return loss;
}

dtype CrossEntropy::loss(const Vector& pred, const Vector& actual) {
    dtype loss(0);
    if (pred.sum() != 1) {
        std::string m("predictions don't sum to one, in:\n");
        throw std::runtime_error(m + __PRETTY_FUNCTION__);
    }
    for (int i = 0; i < pred.rows(); i++) {
        if (actual(i) == 1.) {
            loss = -1 * log(pred(i));
            break;
        }
    }
    return loss;
}
void CrossEntropy::grad_loss_cpu(SharedStorage&, const SharedStorage&,
                       const SharedStorage&, const SharedStorage&) {};
void CrossEntropy::grad_loss_gpu(SharedStorage&, const SharedStorage&,
                       const SharedStorage&, const SharedStorage&) {};
