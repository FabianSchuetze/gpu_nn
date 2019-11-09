#pragma once
#ifndef input_h
#define input_h
#include "layer.h"
class Input : public Layer {
   public:
    Input(int);
    virtual ~Input() = default;
    int output_dimension() override { return _output_dimension; };
    int input_dimension() override { return 0; };
    int output_dimension() const override { return _output_dimension; };
    int input_dimension() const override { return 0; };
    void forward_gpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&) override;
    void forward_cpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&) override;
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
    int _output_dimension;
};
#endif
