#ifndef relu_h
#define relu_h
//#include <cuda_runtime.h>
#include <memory>
#include "cublas_v2.h"
#include "layer.h"
class Relu : public Layer {
    // typedef std::shared_ptr<Storage> SharedStorage;

   public:
    Relu();
    explicit Relu(const std::shared_ptr<Layer>&);
    virtual ~Relu() { CHECK_CUBLAS(cublasDestroy(_handle)); }
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
};
#endif
