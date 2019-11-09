#ifndef softmax_h
#define softmax_h
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "../math.h"
#include "../storage.h"
#include "cublas_v2.h"
#include "layer.h"
class Softmax : public Layer {
    typedef std::shared_ptr<Storage> SharedStorage;

   public:
    Softmax(cublasHandle_t&);
    virtual ~Softmax() = default;
    int input_dimension() override { return 0; };
    int output_dimension() override { return 0; };
    int input_dimension() const override { return 0; };
    int output_dimension() const override { return 0; };
    void forward_gpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&) override;
    void forward_cpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&) override;
    void backward_gpu(int&, const std::shared_ptr<Storage>&,
                      std::vector<std::shared_ptr<Storage>>&) override;
    void backward_cpu(int&, const std::shared_ptr<Storage>&,
                      std::vector<std::shared_ptr<Storage>>&) override;
    std::vector<SharedStorage> return_parameters() override {
        return parameters;
    };
    std::vector<SharedStorage> return_gradients() override {
        return gradients;
    };
    std::vector<SharedStorage> return_parameters() const override {
        return parameters;
    };
    std::vector<SharedStorage> return_gradients() const override {
        return gradients;
    };

   private:
    cublasHandle_t _handle;
};
#endif
