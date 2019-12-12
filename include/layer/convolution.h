#pragma once
#include <memory>
#ifndef convolution_h
#define convolution_h
//#include "cublas_v2.h"
#include "../common.h"
#include "../initalization/init.hpp"
#include "cublas_v2.h"
#include "layer.h"
class Convolution : public Layer {
    friend class Im2ColLayer;
    friend class Pooling;
    friend class NeuralNetwork;
   public:
    Convolution(FilterShape, Pad, Stride, Filters, ImageShape, Channels, Init*);
    Convolution(FilterShape, Pad, Stride, Filters,
                const std::shared_ptr<Layer>&, Init*);
    virtual ~Convolution();
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
    Filters _filters;
    ImageShape _inp, _out;
    Channels _channels;
    cublasHandle_t _handle;
    std::vector<SharedStorage> assistance_parameters;

    void initialize_weight(Init*);
    void initialize_grad();
    void initialize_bias();
    void initialize_kernel();
    void check_size(const SharedStorage&);
    void check_size_backwards(const SharedStorage&, const SharedStorage&);
    int n_batches(const SharedStorage& in);
    void initialize_input_dimension(const std::shared_ptr<Layer>&);
    //void initialize_output_dimension(const std::shared_ptr<Layer>&) override;
    void initialize_output_dimension() override;
    void advance_pointers_forward(const float*&, float*&);
    void advance_pointers_backward(const float*&, const float*&, float*&);
    void backwards_weight_grad_para(int&, int&, int&);
    void backwards_out_grad_para(int&, int&, int&);
    void resize_assistance(const SharedStorage&);
    void initialize_previous(Layer*);
    void reset_previous(const std::shared_ptr<Layer>&);
};
#endif
