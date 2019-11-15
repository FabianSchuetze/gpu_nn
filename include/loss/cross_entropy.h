#ifndef cross_entropy_h
#define cross_entropy_h
#include <memory>
#include <vector>
#include "../storage.h"
#include "loss.h"
typedef std::shared_ptr<Storage> SharedStorage;
class CrossEntropy : public Loss {
   public:
    CrossEntropy();
    CrossEntropy(const std::string&);
    virtual ~CrossEntropy() = default;
    // THE LOSSES ARE ONLY AVAILABLE ON THE HOST -> IS THAT INTENDET?
    dtype loss_cpu(const SharedStorage&, const SharedStorage&) override;
    dtype loss_gpu(const SharedStorage&, const SharedStorage&) override;
    void grad_loss_cpu(SharedStorage&, const SharedStorage&,
                       const SharedStorage&, const SharedStorage&) override;
    // THR GPU IS RELATIVELY SLOW HERE!!!
    void grad_loss_gpu(SharedStorage&, const SharedStorage&,
                       const SharedStorage&, const SharedStorage&) override;
    virtual dtype loss(const SharedStorage&, const SharedStorage&) override;
    virtual void grad_loss(SharedStorage&, const SharedStorage&,
                           const SharedStorage&, const SharedStorage&) override;

   private:
    dtype vec_loss(const Vector&, const Vector&);
    typedef dtype (CrossEntropy::*loss_func)(const SharedStorage&,
                                             const SharedStorage&);
    typedef void (CrossEntropy::*grad_func)(SharedStorage&,
                                            const SharedStorage&,
                                            const SharedStorage&,
                                            const SharedStorage&);
    CrossEntropy::loss_func fun_loss;
    CrossEntropy::grad_func fun_grad;
};
#endif
