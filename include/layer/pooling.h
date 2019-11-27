#pragma once
#ifndef pooling_h
#define pooling_h
#include "layer.h"
class Pooling : public Layer {
   public:
    Pooling(int);
    Pooling(Window, Stride, ImageShape, Channels);
    virtual ~Pooling() = default;
    //int output_dimension() override { return _output_dimension; };
    //int output_dimension() const override { return _output_dimension; };
    void forward_gpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&, const std::string&) override;
    void forward_cpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&, const std::string&) override;
    void backward_gpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    void backward_cpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;

   private:
    SharedStorage mask;
    Window _window;
    Stride _stride;
    ImageShape _inp;
    Channels _channels;
    int batch_size;

    void check_masking(const SharedStorage&);
    void initialize_masking();
    void inline check_input_size(const SharedStorage&);
};
#endif
