#ifndef relu_h
#define relu_h
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "../math.h"
#include "../storage.h"
#include "cublas_v2.h"
#include "layer.h"
class Relu : public Layer {
   public:
    Relu(cublasHandle_t&);
    void forward_gpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&) override;
    void forward_cpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&) override;
    void backward_gpu(int&, const std::shared_ptr<Storage>&,
                      std::vector<std::shared_ptr<Storage>>&) override;
    void backward_cpu(int&, const std::shared_ptr<Storage>&,
                      std::vector<std::shared_ptr<Storage>>&) override;
    std::vector<std::shared_ptr<Storage>> return_parameters() override {
        return parameters;
    };
    std::vector<std::shared_ptr<Storage>> return_gradients() override {
        return gradients;
    }

   private:
    std::vector<std::shared_ptr<Storage>> parameters;
    std::vector<std::shared_ptr<Storage>> gradients;
    cublasHandle_t _handle;
};
#endif
