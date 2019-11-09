#pragma once
#ifndef dense_h
#define dense_h
#include "cublas_v2.h"
#include "layer.h"
class Dense : public Layer {
   public:
    Dense(int, int, cublasHandle_t&);
    int input_dimension() override { return _input_dimension; };
    int input_dimension() const override { return _input_dimension; };
    int output_dimension() override { return _output_dimension; };
    int output_dimension() const override { return _output_dimension; };
    void forward_gpu(const SharedStorage&, SharedStorage&) override;
    void forward_cpu(const SharedStorage&, SharedStorage&) override;
    void backward_gpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    void backward_cpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    VecSharedStorage return_parameters() override { return parameters; };
    VecSharedStorage return_gradients() override { return gradients; }
    VecSharedStorage return_parameters() const override { return parameters; };
    VecSharedStorage return_gradients() const override { return gradients; }

   private:
    void initialize_weight(int, int);
    void initialize_bias(int, int);
    void initialize_grad(int, int);
    std::vector<SharedStorage> parameters;
    std::vector<SharedStorage> gradients;
    cublasHandle_t _handle;
    int _input_dimension;
    int _output_dimension;
};
#endif
