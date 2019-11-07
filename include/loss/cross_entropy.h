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
    ~CrossEntropy() = default;
    dtype loss_cpu(const SharedStorage&, const SharedStorage&) override;
    dtype loss_gpu(const SharedStorage&, const SharedStorage&) override;
    void grad_loss_cpu(SharedStorage&, const SharedStorage&,
                       const SharedStorage&, const SharedStorage&) override;
    void grad_loss_gpu(SharedStorage&, const SharedStorage&,
                       const SharedStorage&, const SharedStorage&) override;

   private:
    dtype loss(const Vector&, const Vector&);
};
#endif
