#pragma once
#ifndef convolution_h
#define convolution_h
//#include "cublas_v2.h"
#include "../common.h"
#include "layer.h"
#include "cublas_v2.h"
class Convolution : public Layer {
   public:
    int output_dimension() override;
    int output_dimension() const override;
    Convolution(FilterShape, Pad, Stride, Filters, ImageShape, Channels);
    virtual ~Convolution();
    void forward_gpu(const SharedStorage&, SharedStorage&,
                     const std::string&) override;
    void forward_cpu(const SharedStorage&, SharedStorage&,
                     const std::string&) override;
    void backward_gpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    void backward_cpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;

    //SharedStorage Column;  // contains the converted image

   private:
    FilterShape _kernel;
    Pad _pad;
    Stride _stride;
    Filters _filters;
    ImageShape _inp, _out;
    Channels _channels;
    cublasHandle_t _handle;

    void initialize_weight();
    void initialize_grad();
    void initialize_kernel();
    void check_size(const SharedStorage&);
    void check_size_backwards(const SharedStorage&, const SharedStorage&);
    int n_batches(const SharedStorage& in);
    void output_shape();
    void advance_pointers_forward(const float*&, float*&);
    void advance_pointers_backward(const float*&, const float*&, float*&);
    void backwards_weight_grad_para(int&, int&, int&);
    void backwards_out_grad_para(int&, int&, int&);
};
#endif
