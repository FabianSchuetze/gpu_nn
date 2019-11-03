#ifndef dense_h
#define dense_h
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "../math.h"
#include "../storage.h"
#include "cublas_v2.h"
#include "layer.h"
class Dense : public Layer {
   public:
    Dense(int, int, cublasHandle_t&);
    void forward_gpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&) override;
    void forward_cpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&) override;
    void backward_gpu(int, const std::vector<std::shared_ptr<Storage>>&,
                      std::vector<std::shared_ptr<Storage>>&) override;
    void backward_cpu(int, const std::vector<std::shared_ptr<Storage>>&,
                      std::vector<std::shared_ptr<Storage>>&) override;
    std::vector<std::shared_ptr<Storage>> return_parameters() override {
        return parameters;
    };
    std::vector<std::shared_ptr<Storage>> return_gradients() override {
        return gradients;
    }

   private:
    void initialize_weight(int, int);
    void initialize_bias(int, int);
    void initialize_grad(int, int);
    std::vector<std::shared_ptr<Storage>> parameters;
    std::vector<std::shared_ptr<Storage>> gradients;
    cublasHandle_t _handle;
};
#endif
