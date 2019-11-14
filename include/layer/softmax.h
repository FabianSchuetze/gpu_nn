#pragma once
#ifndef softmax_h
#define softmax_h
#include "cublas_v2.h"
#include "layer.h"
class Softmax : public Layer {
    typedef std::shared_ptr<Storage> SharedStorage;
    typedef std::vector<std::shared_ptr<Storage>> VecSharedStorage;

   public:
    Softmax();
    virtual ~Softmax() {CHECK_CUBLAS(cublasDestroy(_handle));};
    int input_dimension() override { return 0; };
    int output_dimension() override { return 0; };
    int input_dimension() const override { return 0; };
    int output_dimension() const override { return 0; };
    void forward_gpu(const SharedStorage&, SharedStorage&) override;
    void forward_cpu(const SharedStorage&, SharedStorage&) override;
    void backward_gpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    void backward_cpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    VecSharedStorage return_parameters() override { return parameters; };
    VecSharedStorage return_gradients() override { return gradients; };
    VecSharedStorage return_parameters() const override { return parameters; };
    VecSharedStorage return_gradients() const override { return gradients; };
    void clear_gradients_cpu() override;
    void clear_gradients_gpu() override;

   private:
    cublasHandle_t _handle;
};
#endif
