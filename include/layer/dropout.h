#pragma once
#ifndef dropout_h
#define dropout_h
//#include "cublas_v2.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "layer.h"
#include <random>
class Dropout : public Layer {
    typedef std::shared_ptr<Storage> SharedStorage;
    typedef std::vector<std::shared_ptr<Storage>> VecSharedStorage;

   public:
    explicit Dropout(dtype);
    Dropout(dtype, const std::shared_ptr<Layer>&);
    virtual ~Dropout();
    void forward_gpu(const SharedStorage&, SharedStorage&,
                     const std::string&) override;
    void forward_cpu(const SharedStorage&, SharedStorage&,
                     const std::string&) override;
    void backward_gpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    void backward_cpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    std::vector<int> output_dimension() override;
   private:
    curandGenerator_t gen_device;
    std::mt19937 gen_host;
    SharedStorage masking;
    dtype probability;
    std::uniform_real_distribution<float> dis;

    void initialize_random();
    void initialize_masking();
    void check_masking(const SharedStorage&);
    void check_backward();
};
#endif
