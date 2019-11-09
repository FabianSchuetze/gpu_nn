#ifndef relu_h
#define relu_h
//#include <cuda_runtime.h>
#include <memory>
#include "cublas_v2.h"
#include "layer.h"
class Relu : public Layer {
    typedef std::shared_ptr<Storage> SharedStorage;

   public:
    Relu(cublasHandle_t&);
    int input_dimension() override { return 0; };
    int output_dimension() override { return 0; };
    int input_dimension() const override { return 0; };
    int output_dimension() const override { return 0; };
    void forward_gpu(const SharedStorage&, SharedStorage&) override;
    void forward_cpu(const SharedStorage&, SharedStorage&) override;
    void backward_gpu(int&, const SharedStorage&,
                      std::vector<SharedStorage>&) override;
    void backward_cpu(int&, const SharedStorage&,
                      std::vector<SharedStorage>&) override;
    std::vector<SharedStorage> return_parameters() override {
        return parameters;
    };
    std::vector<SharedStorage> return_parameters() const override {
        return parameters;
    };
    std::vector<SharedStorage> return_gradients() override {
        return gradients;
    };
    std::vector<SharedStorage> return_gradients() const override {
        return gradients;
    };

   private:
    cublasHandle_t _handle;
};
#endif
