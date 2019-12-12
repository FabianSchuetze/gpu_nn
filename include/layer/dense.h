#pragma once
#include <memory>
#ifndef dense_h
#define dense_h
#include "../initalization/init.hpp"
#include "cublas_v2.h"
#include "layer.h"
class Dense : public Layer {
   public:
    Dense(Features, Features, Init*);
    Dense(Features, const std::shared_ptr<Layer>&, Init*);
    virtual ~Dense() { CHECK_CUBLAS(cublasDestroy(_handle)); };
    void forward_gpu(const SharedStorage&, SharedStorage&,
                     const std::string&) override;
    void forward_cpu(const SharedStorage&, SharedStorage&,
                     const std::string&) override;
    void backward_gpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    void backward_cpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    VecSharedStorage return_parameters() override { return parameters; };
    VecSharedStorage return_gradients() override { return gradients; }
    VecSharedStorage return_parameters() const override { return parameters; };
    VecSharedStorage return_gradients() const override { return gradients; }

   private:
    void initialize_weight(int, int, Init*);
    void initialize_bias(int, int);
    void initialize_grad(int, int);
    void initialize_output_dimension() override;
    void initialize_input_dimension(const std::shared_ptr<Layer>&);
    std::vector<SharedStorage> assistance_parameters;
    cublasHandle_t _handle;
    Features _out;
    Features _in;
};
#endif
