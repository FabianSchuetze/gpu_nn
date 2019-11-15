#include "../../include/loss/loss.h"
Loss::Loss(const std::string& s) : device(s) {
    if (device == "GPU") {
        fun_loss = &Loss::loss_gpu;
        fun_grad  = &Loss::grad_loss_gpu;
    } else {
        fun_loss = &Loss::loss_cpu;
        fun_grad  = &Loss::grad_loss_cpu;
    }
}
Loss::Loss() : device("GPU") { 
    fun_loss = &Loss::loss_gpu; 
    fun_grad = &Loss::grad_loss_gpu;
}

dtype Loss::loss_cpu(const SharedStorage&, const SharedStorage&) { return 0; }
dtype Loss::loss_gpu(const SharedStorage&, const SharedStorage&) { return 0; }
void Loss::grad_loss_cpu(SharedStorage&, const SharedStorage&,
                         const SharedStorage&, const SharedStorage&) {
    ;
};
void Loss::grad_loss_gpu(SharedStorage&, const SharedStorage&,
                         const SharedStorage&, const SharedStorage&) {
    ;
};

dtype Loss::loss(const SharedStorage& prediction, const SharedStorage& actual) {
    return (this->*fun_loss)(prediction, actual);
}

void Loss::grad_loss(SharedStorage& gradient,
                                 const SharedStorage& prediction,
                                 const SharedStorage& target,
                                 const SharedStorage& target2) {
    (this->*fun_grad)(gradient, prediction, target, target2);
}
