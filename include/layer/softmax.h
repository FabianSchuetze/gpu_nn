#pragma once
#include <algorithm>
#include <memory>
#ifndef softmax_h
#define softmax_h
#include "cublas_v2.h"
#include "layer.h"
class Softmax : public Layer {
    typedef std::shared_ptr<Storage> SharedStorage;
    typedef std::vector<std::shared_ptr<Storage>> VecSharedStorage;

   public:
    Softmax();
    explicit Softmax(const std::shared_ptr<Layer>&);
    virtual ~Softmax() { CHECK_CUBLAS(cublasDestroy(_handle)); };
    void forward_gpu(const SharedStorage&, SharedStorage&,
                     const std::string&) override;
    void forward_cpu(const SharedStorage&, SharedStorage&,
                     const std::string&) override;
    void backward_gpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    void backward_cpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;

   private:
    cublasHandle_t _handle;
    SharedStorage max;
    SharedStorage ones;
    void resize_maybe(int, int);
    void create_storage();
};
#endif
