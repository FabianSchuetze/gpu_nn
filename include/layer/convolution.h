#pragma once
#ifndef convolution_h
#define convolution_h
//#include "cublas_v2.h"
#include <cudnn.h>
#include "../common.h"
#include "layer.h"
#include "../workspace_manager.hpp"
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

    SharedStorage Column;  // contains the converted image
    void im2col(const SharedStorage&);

   private:
    FilterShape _filter_shape;
    Pad _pad;
    Stride _stride;
    Filters _filters;
    ImageShape _inp, _out;
    Channels _channels;
    void initialize_weight();
    void initialize_grad();
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_des, output_des;
    cudnnFilterDescriptor_t kernel_des;
    cudnnConvolutionDescriptor_t convolution_des;
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    cudnnConvolutionBwdFilterAlgo_t convolution_bwd_algorithm;
    cudnnConvolutionBwdDataAlgo_t convolution_bwd_data_algo;
    WorkspaceManager ffw, bwd, bwd_data;
    int batch_size;
    SharedStorage col;

    void initialize_cudnn_handles();
    void resize(int);
    void resize_gpu(int);
    void resize_cpu(int);
    void initialize_tensors(int);
    void allocate_memory();
    void initialize_algorithm();
    void initialize_kernel();
    void calculate_output_size();
};
#endif
