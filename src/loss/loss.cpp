#include "../../include/loss/loss.h"
dtype Loss::loss_cpu(const SharedStorage&, const SharedStorage&) {return 0;}
dtype Loss::loss_gpu(const SharedStorage&, const SharedStorage&) {return 0;}
void Loss::grad_loss_cpu(SharedStorage&, const SharedStorage&,
                               const SharedStorage&, const SharedStorage&) {;};
void Loss::grad_loss_gpu(SharedStorage&, const SharedStorage&,
                               const SharedStorage&, const SharedStorage&) {;};
