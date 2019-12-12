#pragma once
#include <memory>
#ifndef im2col_layer_h
#define im2col_layer_h
#include "layer.h"
#include "convolution.h"
class Im2ColLayer : public Layer {

   public:
    explicit Im2ColLayer(const std::shared_ptr<Convolution>& convolution_layer);
    virtual ~Im2ColLayer() { };
    //std::vector<int> output_dimension() const override;
    void forward_gpu(const SharedStorage&, SharedStorage&,
                     const std::string&) override;
    void forward_cpu(const SharedStorage&, SharedStorage&,
                     const std::string&) override;
    void backward_gpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    void backward_cpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    int input_dimension() override;
    int input_dimension() const override;
   private:
    FilterShape _kernel;
    Pad _pad;
    Stride _stride;
    ImageShape _inp, _out;
    Channels _channels;
    //std::shared_ptr<Convolution> next;

    void initialize_output_dimension() override;
    void check_size(const SharedStorage&, const SharedStorage&);
    void advance_pointers_forward(const float*&, float*&, int);
    void advance_pointers_backward(const float*&, float*&);


};
#endif
