#ifndef loss_h
#define loss_h
#include <memory>
#include <vector>
#include "../storage.h"
#include "../common.h"
typedef std::shared_ptr<Storage> SharedStorage;
class Loss {
   public:
    Loss();
    Loss(const std::string&);
    virtual ~Loss() = default;
    virtual dtype loss_cpu(const SharedStorage&, const SharedStorage&);
    virtual dtype loss_gpu(const SharedStorage&, const SharedStorage&);
    virtual void grad_loss_cpu(SharedStorage&, const SharedStorage&,
                               const SharedStorage&, const SharedStorage&);
    virtual void grad_loss_gpu(SharedStorage&, const SharedStorage&,
                               const SharedStorage&, const SharedStorage&);
    virtual dtype loss(const SharedStorage&, const SharedStorage&);
    virtual void grad_loss(SharedStorage&, const SharedStorage&, 
            const SharedStorage&, const SharedStorage&);
   protected:
    std::string device;
   private:
    typedef dtype (Loss::*loss_func)(const SharedStorage&, const SharedStorage&);
    typedef void (Loss::*grad_func)(SharedStorage&, const SharedStorage&,
                                    const SharedStorage&, const SharedStorage&);
    Loss::loss_func fun_loss;
    Loss::grad_func fun_grad;
};
#endif
