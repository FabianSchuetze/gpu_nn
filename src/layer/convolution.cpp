#include "../../include/layer/convolution.h"
#include <iostream>

Convolution::Convolution(FilterShape filtershape, Pad pad, Stride stride,
                         Filters filters, ImageShape imageshape,
                         Channels channels)
    : _filter_shape(filtershape),
      _pad(pad),
      _stride(stride),
      _filters(filters),
      _image_shape(imageshape),
      _channels(channels),
      workspace_bytes(0),
      d_workspace(NULL),
      batch_size(0) {
    initialize_cudnn_handles();
    initialize_weight();
    initialize_grad();
}

Convolution::~Convolution() {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_descriptor));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
    CHECK_CUDNN(cudnnDestroy(cudnn));
}

void Convolution::initialize_cudnn_handles() {
    CHECK_CUDNN(cudnnCreate(&cudnn));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
}

void Convolution::resize(int new_batch_size) {
    if (batch_size != new_batch_size) {
        cudaFree(d_workspace);
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(
            input_descriptor,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/new_batch_size,
            /*channels=*/_channels.get(),
            /*image_height=*/_image_shape.get().first,
            /*image_width=*/_image_shape.get().second));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(
            output_descriptor,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/new_batch_size,
            /*channels=*/_channels.get(),
            /*image_height=*/_image_shape.get().first,
            /*image_width=*/_image_shape.get().second));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(
            kernel_descriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*out_channels=*/_filters.get(),
            /*in_channels=*/_channels.get(),
            /*kernel_height=*/_filter_shape.get().first,
            /*kernel_width=*/_filter_shape.get().second));
        allocate_memory();
    }
}

void Convolution::allocate_memory() {
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
        output_descriptor, convolution_algorithm, &workspace_bytes));
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
              << std::endl;
    cudaMalloc(&d_workspace, workspace_bytes);
}

void Convolution::initialize_data_structures() {
    CHECK_CUDNN(
        cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                        /*pad_height=*/_pad.get(),
                                        /*pad_width=*/_pad.get(),
                                        /*vertical_stride=*/_stride.get(),
                                        /*horizontal_stride=*/_stride.get(),
                                        /*dilation_height=*/0,
                                        /*dilation_width=*/0,
                                        /*mode=*/CUDNN_CROSS_CORRELATION,
                                        /*computeType=*/CUDNN_DATA_FLOAT));
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(
        cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
        output_descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        /*memoryLimitInBytes=*/0, &convolution_algorithm));
}

void Convolution::initialize_grad() {
    int rows = _filters.get();
    int cols =
        _channels.get() *_filter_shape.get().first * _filter_shape.get().second;
    Matrix tmp = Matrix(rows, cols).setZero();
    gradients.push_back(std::make_shared<Storage>(tmp));
}

void Convolution::initialize_weight() {
    srand((unsigned int)time(0));
    int rows = _filters.get();
    int cols =
        _channels.get() *_filter_shape.get().first * _filter_shape.get().second;
    Matrix tmp = Matrix(rows, cols).setZero();
    dtype glorot_scale = std::sqrt(6.) / std::sqrt(rows + cols);
    tmp *= glorot_scale;
    parameters.push_back(std::make_shared<Storage>(tmp));
}

void Convolution::forward_gpu(const SharedStorage& in, SharedStorage& out,
                              const std::string&) {
    const float alpha = 1, beta = 0;
    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn, &alpha, input_descriptor, in->gpu_pointer_const(),
        kernel_descriptor, parameters[0]->gpu_pointer(), convolution_descriptor,
        convolution_algorithm, d_workspace, workspace_bytes, &beta,
        output_descriptor, out->gpu_pointer()));
}

void Convolution::forward_cpu(const SharedStorage& in, SharedStorage& out,
                              const std::string&) {}
void Convolution::backward_gpu(const SharedStorage&, const SharedStorage&,
                               SharedStorage&) {
    ;
};
void Convolution::backward_cpu(const SharedStorage&, const SharedStorage&,
                               SharedStorage&) {
    ;
};

