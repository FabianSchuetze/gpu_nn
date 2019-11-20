#pragma once
#ifndef dropout_h
#define dropout_h
//#include "cublas_v2.h"
#include "layer.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
class Dropout : public Layer {
    typedef std::shared_ptr<Storage> SharedStorage;
    typedef std::vector<std::shared_ptr<Storage>> VecSharedStorage;

   public:
    Dropout(int, int, dtype);
    virtual ~Dropout();
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
    curandState* d_state;
    std::vector<SharedStorage> assistance_parameters;

    void initialize_probability(const dtype&);
    void initialize_random(int, int);
};
#endif
