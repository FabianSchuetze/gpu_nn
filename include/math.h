#pragma once
#ifndef math_h
#define math_h
#include <curand.h>
#include <curand_kernel.h>
#include <memory>
#include "common.h"
#include "cublas_v2.h"
#include "storage.h"

typedef std::shared_ptr<Storage> SharedStorage;
void my_Dgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t,
              const SharedStorage&, const SharedStorage&, SharedStorage&, dtype,
              dtype);
void my_Dgemv(cublasHandle_t, cublasOperation_t, const SharedStorage&,
              const SharedStorage&, SharedStorage&, dtype, dtype);
void my_add_vec_to_mat_colwise(SharedStorage&, const SharedStorage&, dtype);
void my_add_vec_to_mat_colwise(const SharedStorage&, const SharedStorage&,
                               SharedStorage&, dtype);
void my_mult_elementwise(const SharedStorage&, const SharedStorage&,
                         SharedStorage&);
void my_Exponential(SharedStorage&);
void my_Divide_colwise(SharedStorage&, const SharedStorage&);
void my_relu(SharedStorage&, const SharedStorage&);
void my_relu_backwards(const SharedStorage&, const SharedStorage&,
                       SharedStorage&);
void my_cross_entropy_loss(dtype&, const SharedStorage&, const SharedStorage&);
void my_cross_entropy_gradient(SharedStorage&, const SharedStorage&,
                               const SharedStorage);
void my_Matrix_addition_inplace(const SharedStorage&, SharedStorage&, dtype);
void my_Matrix_addition(const SharedStorage&, const SharedStorage&,
                        SharedStorage&, dtype, dtype);
void my_cuda_masking(dtype, SharedStorage&);
void pooling_cpu(const float* src, int window, int stride, int rows, int cols,
                 int channels, int out_height, int out_width, int n_batches,
                 float* dest, float* mask);
void pooling_backward_cpu(const float* src, const float* mask, int window,
                          int stride, int rows, int cols, int channels,
                          int out_height, int out_width,
                          int n_batches, float* dest);
void im2col_cpu(const float* data_im, int channels, int rows, int cols,
                int kernel_h, const int kernel_w, int pad, int stride,
                float* data_col);
void col2im_cpu(const dtype* data_col, int channels, int rows, int cols,
                int kernel_h, int kernel_w, int pad, int stride, dtype*);
#endif
