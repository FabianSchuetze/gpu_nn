#ifndef cross_entropy_h
#define cross_entropy_h
#include <memory>
#include <vector>
#include "../storage.h"
#include "loss.h"
typedef std::shared_ptr<Storage> SharedStorage;
class CrossEntropy : public Loss {
   public:
    CrossEntropy() = default;
    virtual ~CrossEntropy() = default;
    // THE LOSSES ARE ONLY AVAILABLE ON THE HOST -> IS THAT INTENDET?
    dtype loss_cpu(const SharedStorage&, const SharedStorage&) override;
    dtype loss_gpu(const SharedStorage&, const SharedStorage&) override;
    void grad_loss_cpu(SharedStorage&, const SharedStorage&,
                       const SharedStorage&, const SharedStorage&) override;
    // THR GPU IS RELATIVELY SLOW HERE!!!
    void grad_loss_gpu(SharedStorage&, const SharedStorage&,
                       const SharedStorage&, const SharedStorage&) override;

   private:
    dtype loss(const Vector&, const Vector&);
};
#endif
