#ifndef layer_h
#define layer_h
#include <memory>
#include <vector>
#include "../storage.h"
class Layer {
   public:
    Layer() = default;
    ~Layer() = default;
    // Layer(int, int);
    virtual void forward_gpu(const std::shared_ptr<Storage>&,
                             std::shared_ptr<Storage>&) = 0;
    virtual void forward_cpu(const std::shared_ptr<Storage>&,
                             std::shared_ptr<Storage>&) = 0;
    virtual void backward_gpu(int&, const std::shared_ptr<Storage>&,
                              std::vector<std::shared_ptr<Storage>>&) = 0;
    virtual void backward_cpu(int&, const std::shared_ptr<Storage>&,
                              std::vector<std::shared_ptr<Storage>>&) = 0;
    virtual std::vector<std::shared_ptr<Storage>> return_parameters() = 0;
    virtual std::vector<std::shared_ptr<Storage>> return_gradients() = 0;

   private:
    std::vector<std::shared_ptr<Storage>> parameters;
    std::vector<std::shared_ptr<Storage>> gradients;
};
#endif
