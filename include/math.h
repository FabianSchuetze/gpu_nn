#pragma once
#ifndef math_h
#define math_h
#include <curand_kernel.h>
#include <curand.h>
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
void my_cuda_masking(dtype, SharedStorage&);
void im2col(const dtype* input_data, const int depth, const int height,
            const int width, const int filter_h, const int filter_w,
            const int pad_t, const int pad_l, const int pad_b, const int pad_r,
            const int stride_h, const int stride_w, dtype* col_data);
void im2col_cpu(const dtype* input_data, const int depth, const int height,
            const int width, const int filter_h, const int filter_w,
            const int pad_t, const int pad_l, const int pad_b, const int pad_r,
            const int stride_h, const int stride_w, dtype* col_data);
void pooling_cpu(const float* src, int window, int stride, int rows, int cols,
                 int channels, int n_batches, float* dest, float* mask);
#endif
