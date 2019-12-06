#include "../include/math.h"
#include <iostream>
#include <memory>
#include "../include/common.h"
#include "../include/cuda_math.h"
void my_Dgemm(cublasHandle_t handle, cublasOperation_t transA,
              cublasOperation_t transB, const SharedStorage& A,
              const SharedStorage& B, SharedStorage& C, dtype alpha,
              dtype beta) {
    int M(0), N(0), K(0), LDA(0), LDB(0), LDC(0);
    if (transA == CUBLAS_OP_N) {
        M = A->get_rows();
        K = A->get_cols();
        LDA = M;
    } else if (transA == CUBLAS_OP_T) {
        M = A->get_cols();
        K = A->get_rows();
        LDA = K;
    } else {
        std::cout << "connot find contion\n";
    }
    if (transB == CUBLAS_OP_N) {
        N = B->get_cols();
        LDB = K;
    } else if (transB == CUBLAS_OP_T) {
        N = B->get_rows();
        LDB = N;
    } else {
        std::cout << "connot find contion2\n";
    }
    LDC = M;
    // int N = B->get_cols();
    // int K = A->get_cols();
    const dtype* d_A = A->gpu_pointer_const();
    const dtype* d_B = B->gpu_pointer_const();
    dtype* d_C = C->gpu_pointer();
    my_cuda_Dgemm(handle, transA, transB, M, N, K, &alpha, d_A, LDA, d_B, LDB,
                  &beta, d_C, LDC);
    // cudaDeviceSyncronize();
}

void my_Dgemv(cublasHandle_t handle, cublasOperation_t transA,
              const SharedStorage& A, const SharedStorage& B, SharedStorage& C,
              dtype alpha, dtype beta) {
    int M = A->get_rows();
    int N = A->get_cols();
    const dtype* d_A = A->gpu_pointer_const();
    const dtype* d_B = B->gpu_pointer_const();
    dtype* d_C = C->gpu_pointer();
    my_cuda_Dgemv(handle, transA, M, N, &alpha, d_A, d_B, &beta, d_C);
    // cudaDeviceSyncronize();
}

void my_add_vec_to_mat_colwise(SharedStorage& A, const SharedStorage& B,
                               dtype alpha) {
    int rows = A->get_rows();
    int cols = A->get_cols();
    dtype* d_A = A->gpu_pointer();
    const dtype* d_B = B->gpu_pointer_const();
    add_vec_to_mat_colwise(rows, cols, d_A, d_B, alpha);
    // cudaDeviceSyncronize();
}

void my_add_vec_to_mat_colwise(const SharedStorage& in, const SharedStorage& B,
                               SharedStorage& out, dtype alpha) {
    int rows = in->get_rows();
    int cols = in->get_cols();
    const dtype* d_A = in->gpu_pointer_const();
    const dtype* d_B = B->gpu_pointer_const();
    dtype* d_C = out->gpu_pointer();
    add_vec_to_mat_colwise(rows, cols, d_A, d_B, d_C, alpha);
    // cudaDeviceSyncronize();
}

void my_Exponential(SharedStorage& in) {
    int rows = in->get_rows();
    int cols = in->get_cols();
    dtype* d_A = in->gpu_pointer();
    exponential(rows, cols, d_A);
}

void my_Divide_colwise(SharedStorage& in, const SharedStorage& vec) {
    int rows = in->get_rows();
    int cols = in->get_cols();
    dtype* d_A = in->gpu_pointer();
    const dtype* d_B = vec->gpu_pointer_const();
    divide_colwise(rows, cols, d_A, d_B);
}

void my_relu(SharedStorage& in, const SharedStorage& vec) {
    int rows = in->get_rows();
    int cols = in->get_cols();
    dtype* d_a = in->gpu_pointer();
    const dtype* d_b = vec->gpu_pointer_const();
    relu(rows, cols, d_a, d_b);
}

void my_relu_backwards(const SharedStorage& values,
                       const SharedStorage& grad_in, SharedStorage& grad_out) {
    int rows = values->get_rows();
    int cols = values->get_cols();
    const dtype* d_A = values->gpu_pointer_const();
    const dtype* d_B = grad_in->gpu_pointer_const();
    dtype* d_C = grad_out->gpu_pointer();
    relu_backwards(rows, cols, d_A, d_B, d_C);
}

void my_cross_entropy_loss(dtype& loss, const SharedStorage& prediction,
                           const SharedStorage& actual) {
    int cols = prediction->get_cols();
    int rows = prediction->get_rows();
    SharedStorage all_losses = std::make_shared<Storage>(Matrix::Zero(cols, 1));
    const dtype* d_A = prediction->gpu_pointer_const();
    const dtype* d_B = actual->gpu_pointer_const();
    dtype* d_C = all_losses->gpu_pointer();
    all_cross_entropy_losses(rows, cols, d_A, d_B, d_C);
    loss = all_losses->return_data_const().sum();
}

void my_cross_entropy_gradient(SharedStorage& gradient,
                               const SharedStorage& prediction,
                               const SharedStorage target) {
    int cols = prediction->get_cols();
    int rows = prediction->get_rows();
    const dtype* d_A = prediction->gpu_pointer_const();
    const dtype* d_B = target->gpu_pointer_const();
    dtype* d_C = gradient->gpu_pointer();
    cross_entropy_gradient(rows, cols, d_A, d_B, d_C);
}

// IN PRICINPILE THATS A DUPLICATE FROM ABOVE!!!! FIND COMMON MATH FUNCTION!!!
void my_Matrix_addition_inplace(const SharedStorage& gradient,
                                SharedStorage& parameters, dtype alpha) {
    int cols = parameters->get_cols();
    int rows = parameters->get_rows();
    const dtype* d_A = gradient->gpu_pointer_const();
    dtype* d_B = parameters->gpu_pointer();
    matrix_addition_inplace(rows, cols, d_A, d_B, alpha);
}

void my_Matrix_addition(const SharedStorage& A, const SharedStorage& B,
                        SharedStorage& C, dtype alpha_A, dtype alpha_B) {
    int cols = C->get_cols();
    int rows = C->get_rows();
    const dtype* d_A = A->gpu_pointer_const();
    const dtype* d_B = B->gpu_pointer_const();
    dtype* d_C = C->gpu_pointer();
    matrix_addition(rows, cols, d_A, d_B, d_C, alpha_A, alpha_B);
}

void my_mult_elementwise(const SharedStorage& A, const SharedStorage& B,
                         SharedStorage& C) {
    int rows = A->get_rows();
    int cols = A->get_cols();
    const dtype* d_A = A->gpu_pointer_const();
    const dtype* d_B = B->gpu_pointer_const();
    dtype* d_C = C->gpu_pointer();
    multiply_elementwise(rows, cols, d_A, d_B, d_C);
}

void my_cuda_masking(dtype probability, SharedStorage& mask) {
    int rows = mask->get_rows();
    int cols = mask->get_cols();
    dtype* d_A = mask->gpu_pointer();
    cuda_masking(rows, cols, probability, d_A);
}

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void im2col_cpu(const float* data_im, int channels, int rows, int cols,
                int kernel_h, const int kernel_w, int pad, int stride,
                float* data_col) {
    const int out_height = (rows + 2 * pad - kernel_h) / stride + 1;
    const int out_width = (rows + 2 * pad - kernel_h) / stride + 1;
    const int channel_size = rows * cols;
    for (int c = channels; c--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad + kernel_row;
                for (int output_rows = out_height; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, rows)) {
                        for (int output_cols = out_width; output_cols;
                             output_cols--) {
                            *(data_col++) = 0;
                        }
                    } else {
                        int input_col = -pad + kernel_col;
                        for (int output_col = out_width; output_col;
                             output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, cols)) {
                                int l = input_row * cols + input_col;
                                // std::cout << "linear : " << l << std::endl;
                                *(data_col++) = data_im[l];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col += stride;
                        }
                    }
                    input_row += stride;
                }
            }
        }
    }
}

void col2im_cpu(const dtype* data_col, int channels, int rows, int cols,
                int kernel_h, int kernel_w, int pad, int stride,
                dtype* data_im) {
    const int out_height = (rows + 2 * pad - kernel_h) / stride + 1;
    const int out_width = (rows + 2 * pad - kernel_h) / stride + 1;
    const int channel_size = rows * cols;
    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad + kernel_row;
                for (int output_rows = out_height; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, rows)) {
                        data_col += out_width;
                    } else {
                        int input_col = -pad + kernel_col;
                        for (int output_col = out_width; output_col;
                             output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, cols)) {
                                data_im[input_row * cols + input_col] +=
                                    *data_col;
                            }
                            data_col++;
                            input_col += stride;
                        }
                    }
                    input_row += stride;
                }
            }
        }
    }
}
void pooling_cpu(const float* src, int window, int stride, int rows, int cols,
                 int channels, int n_batches, float* dest, float* mask) {
    if (((rows - window) % stride) or ((cols - window) % stride)) {
        throw std::invalid_argument("Doesnt match");
    }
    int out_height = (rows - window) / stride + 1;
    int out_width = (cols - window) / stride + 1;
    for (int n = 0; n < n_batches; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int ph = 0; ph < out_height; ++ph) {
                for (int pw = 0; pw < out_width; ++pw) {
                    int hstart = ph * stride;
                    int wstart = pw * stride;
                    // int hend = std::min(hstart + window, rows);
                    // int wend = std::min(wstart + window, cols);
                    int hend = hstart + window;
                    int wend = wstart + window;
                    // hstart = std::max(hstart, 0);
                    // wstart = std::max(wstart, 0);
                    const int pool_index = ph * out_width + pw;
                    for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                            const int index = h * cols + w;
                            if (src[index] > dest[pool_index]) {
                                dest[pool_index] = src[index];
                                mask[pool_index] = index;
                            }
                        }
                    }
                }
            }
            src += (rows * cols);
            dest += (out_height * out_width);
            mask += (out_height * out_width);
        }
    }
}

void pooling_backward_cpu(const float* src, const float* mask, int window,
                          int stride, int rows, int cols, int channels,
                          int n_batches, float* dest) {
    if (((rows - window) % stride) or ((cols - window) % stride)) {
        throw std::invalid_argument("Doesnt match");
    }
    int out_height = (rows - window) / stride + 1;
    int out_width = (cols - window) / stride + 1;
    for (int n = 0; n < n_batches; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int ph = 0; ph < out_height; ++ph) {
                for (int pw = 0; pw < out_width; ++pw) {
                    int index = ph * out_width + pw;
                    int dest_idx = mask[index];
                    dest[dest_idx] += src[index];
                }
            }
            dest += (rows * cols);
            src += out_height * out_width;
            mask += out_height * out_width;
        }
    }
}

