#pragma once
#ifndef input_h
#define input_h
#include "layer.h"
class Input : public Layer {
   public:
    Input(int);
    virtual ~Input() = default;
    int output_dimension() override { return _output_dimension; };
    //int input_dimension() override { return 0; };
    int output_dimension() const override { return _output_dimension; };
    //int input_dimension() const override { return 0; };
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
