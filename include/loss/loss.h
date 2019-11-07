#ifndef loss_h
#define loss_h
#include <memory>
#include <vector>
#include "../storage.h"
#include "../common.h"
typedef std::shared_ptr<Storage> SharedStorage;
class Loss {
   public:
    Loss() = default;
    ~Loss() = default;
    virtual dtype loss_cpu(const SharedStorage&, const SharedStorage&) = 0;
    virtual dtype loss_gpu(const SharedStorage&, const SharedStorage&) = 0;
    virtual void grad_loss_cpu(SharedStorage&, const SharedStorage&,
                               const SharedStorage&, const SharedStorage&) = 0;
    virtual void grad_loss_gpu(SharedStorage&, const SharedStorage&,
                               const SharedStorage&, const SharedStorage&)= 0;
};
#endif
