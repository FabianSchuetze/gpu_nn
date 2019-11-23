#include "../../include/layer/convolution.h"
#include <iostream>
#include "../../include/math.h"

void print_Matrix_to_stdout2(const Matrix& val, std::string loc) {
    int rows(val.rows()), cols(val.cols());
    std::ofstream myfile(loc);
    myfile << "dimensions: rows, cols: " << rows << ", " << cols << std::endl;
    myfile << std::fixed;
    myfile << std::setprecision(2);
    for (int row = 0; row < rows; ++row) {
        myfile << val(row, 0);
        for (int col = 1; col < cols; ++col) {
            myfile << ", " << val(row, col);
        }
        myfile << std::endl;
    }
}
Convolution::Convolution(FilterShape filtershape, Pad pad, Stride stride,
                         Filters filters, ImageShape imageshape,
                         Channels channels)
    : _filter_shape(filtershape),
      _pad(pad),
      _stride(stride),
      _filters(filters),
      _inp(imageshape),
      _out(0, 0),
      _channels(channels),
      workspace_bytes(0),
      d_workspace(NULL),
      batch_size(0) {
    calculate_output_size();
    initialize_weight();
    initialize_grad();
    initialize_cudnn_handles();
    initialize_kernel();
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
}

void Convolution::calculate_output_size() {
    int h_num = (_inp.get().first - _filter_shape.get().first + 2 * _pad.get());
    int w_num =
        (_inp.get().second - _filter_shape.get().second + 2 * _pad.get());
    if ((h_num % _stride.get()) or (w_num % _stride.get())) {
        std::stringstream ss;
        ss << "Output size is not an integer, in:\n"
           << __PRETTY_FUNCTION__ << "\ncalled from " << __FILE__ << " at "
           << __LINE__;
        throw std::invalid_argument(ss.str());
    }
    int height = h_num / _stride.get() + 1;
    int width = w_num / _stride.get() + 1;
    _out = ImageShape(height, width);
}

void Convolution::resize_gpu(int new_batch_size) {
    cudaFree(d_workspace);
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                           /*format=*/CUDNN_TENSOR_NHWC,
                                           /*dataType=*/CUDNN_DATA_FLOAT,
                                           /*batch_size=*/new_batch_size,
                                           /*channels=*/_channels.get(),
                                           /*image_height=*/_inp.get().first,
                                           /*image_width=*/_inp.get().second));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                           /*format=*/CUDNN_TENSOR_NHWC,
                                           /*dataType=*/CUDNN_DATA_FLOAT,
                                           /*batch_size=*/new_batch_size,
                                           /*channels=*/_filters.get(),
                                           /*image_height=*/_out.get().first,
                                           /*image_width=*/_out.get().second));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        kernel_descriptor,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*out_channels=*/_filters.get(),
        /*in_channels=*/_channels.get(),
        /*kernel_height=*/_filter_shape.get().first,
        /*kernel_width=*/_filter_shape.get().second));
    initialize_algorithm();
    allocate_memory();
}

void Convolution::allocate_memory() {
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
        output_descriptor, convolution_algorithm, &workspace_bytes));
    cudaMalloc(&d_workspace, workspace_bytes);
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
              << std::endl;
}

void Convolution::initialize_kernel() {
    int pad = _pad.get();
    int stride = _stride.get();
    const int dilation = 1;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(
        cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                        /*pad_height=*/pad,
                                        /*pad_width=*/pad,
                                        /*vertical_stride=*/stride,
                                        /*horizontal_stride=*/stride,
                                        /*dilation_height=*/dilation,
                                        /*dilation_width=*/dilation,
                                        /*mode=*/CUDNN_CROSS_CORRELATION,
                                        /*computeType=*/CUDNN_DATA_FLOAT));
}

void Convolution::initialize_algorithm() {
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(
        cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
        output_descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        /*memoryLimitInBytes=*/0, &convolution_algorithm));
}

void Convolution::initialize_grad() {
    int rows = _filters.get();
    int cols = _channels.get() * _filter_shape.get().first *
               _filter_shape.get().second;
    Matrix tmp = Matrix(rows, cols).setZero();
    gradients.push_back(std::make_shared<Storage>(tmp));
}

void Convolution::initialize_weight() {
    srand((unsigned int)time(0));
    int rows = _filters.get();
    int cols = _channels.get() * _filter_shape.get().first *
               _filter_shape.get().second;
    Matrix tmp = Matrix::Random(rows, cols);
    dtype glorot_scale = std::sqrt(6.) / std::sqrt(rows + cols);
    tmp *= glorot_scale;
    parameters.push_back(std::make_shared<Storage>(tmp));
}

void Convolution::resize(int new_batch_size) {
    if (batch_size != new_batch_size) {
        resize_gpu(new_batch_size);
        resize_cpu(new_batch_size);
    }
}
void Convolution::forward_gpu(const SharedStorage& in, SharedStorage& out,
                              const std::string&) {
    resize_gpu(in->get_cols());
    const float alpha = 1.0f, beta = 0.f;
    checkCUDNN(cudnnConvolutionForward(
        cudnn, &alpha, input_descriptor, in->gpu_pointer_const(),
        kernel_descriptor, parameters[0]->gpu_pointer_const(),
        convolution_descriptor,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        // convolution_algorithm,
        d_workspace, workspace_bytes, &beta, output_descriptor,
        out->gpu_pointer()));
}

void Convolution::im2col(const SharedStorage& image) {
    dtype* colp = col->cpu_pointer();
    dtype const* imagep = image->cpu_pointer_const();
    for (int t = 0; t < image->get_cols(); ++t) {
        ::im2col(imagep, _channels.get(), _inp.get().first, _inp.get().second,
                 _filter_shape.get().first, _filter_shape.get().second,
                 _pad.get(), _pad.get(), _pad.get(), _pad.get(), _stride.get(),
                 _stride.get(), colp);
        imagep += _inp.get().first * _inp.get().second * _channels.get();
        colp += _filter_shape.get().first * _filter_shape.get().second *
                _channels.get() * _out.get().first * _out.get().second;
    }
}

void Convolution::resize_cpu(int new_batch_size) {
    int rows = _filter_shape.get().first * _filter_shape.get().second *
               _channels.get();
    int cols = _out.get().first * _out.get().second * new_batch_size;
    Matrix tmp = Matrix::Zero(rows, cols);
    col = std::make_shared<Storage>(tmp);
}

void Convolution::forward_cpu(const SharedStorage& in, SharedStorage& out,
                              const std::string&) {
    resize_cpu(in->get_cols());
    im2col(in);
    print_Matrix_to_stdout2(col->return_data_const(),
                            "/home/fabian/Documents/work/gpu_nn/debug/col.txt");
    print_Matrix_to_stdout2(
        parameters[0]->return_data_const(),
        "/home/fabian/Documents/work/gpu_nn/debug/weight.txt");
    out->return_data() =
        parameters[0]->return_data_const() * col->return_data_const();
    // int out_rows =
    // out->return_data() = Eigen::Map<Matrix>(
    // out->return_data() = tmp;
    print_Matrix_to_stdout2(out->return_data_const(),
                            "/home/fabian/Documents/work/gpu_nn/debug/out.txt");
}

void Convolution::backward_gpu(const SharedStorage&, const SharedStorage&,
                               SharedStorage&) {
    ;
};

void Convolution::backward_cpu(const SharedStorage&, const SharedStorage&,
                               SharedStorage&) {
    ;
};
