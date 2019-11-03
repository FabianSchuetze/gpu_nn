#ifndef layer_h
#define layer_h
#include <memory>
#include "../storage.h"
#include <vector>
class Layer {
   public:
    Layer() = default;
    //Layer(int, int);
    virtual void forward_gpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&) = 0;
    virtual void forward_cpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&)= 0;
    virtual std::vector<std::shared_ptr<Storage>> return_parameters() = 0;

   private:
    std::vector<std::shared_ptr<Storage>> parameters;
    std::vector<std::shared_ptr<Storage>> gradients;
};
#endif
