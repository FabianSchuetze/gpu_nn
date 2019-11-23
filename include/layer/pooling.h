#pragma once
#ifndef pooling_h
#define pooling_h
#include "layer.h"
class Pooling : public Layer {
   public:
    Pooling(int);
    virtual ~Pooling() = default;
    int output_dimension() override { return _output_dimension; };
    int output_dimension() const override { return _output_dimension; };
    void forward_gpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&, const std::string&) override;
    void forward_cpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&, const std::string&) override;
    void backward_gpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    void backward_cpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;

   private:
    int _output_dimension;
};
#endif
