#pragma once
#ifndef im2col_layer_h
#define im2col_layer_h
#include "layer.h"
class Im2ColLayer : public Layer {

   public:
    Im2ColLayer(FilterShape, Pad, Stride, ImageShape, Channels);
    virtual ~Im2ColLayer() { };
    int output_dimension() override;
    int output_dimension() const override;
    virtual int n_cols() override;
    virtual int n_cols() const override;
    void forward_gpu(const SharedStorage&, SharedStorage&,
                     const std::string&) override;
    void forward_cpu(const SharedStorage&, SharedStorage&,
                     const std::string&) override;
    void backward_gpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    void backward_cpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
   private:
    FilterShape _kernel;
    Pad _pad;
    Stride _stride;
    ImageShape _inp, _out;
    Channels _channels;

    void output_shape();
    void check_size(const SharedStorage&, const SharedStorage&);
    void advance_pointers_forward(const float*&, float*&, int);
    void advance_pointers_backward(const float*&, float*&);


};
#endif
