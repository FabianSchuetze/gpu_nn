#ifndef cuda_math_h
#define cuda_math_h
#include "cublas_v2.h"
void my_cuda_Dgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int M,
                   int N, int K, double* alpha, const double*& d_A, int,
                   const double*& d_B, int, double* beta, double*& d_C, int);
void my_cuda_Dgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int M,
                   int N, int K, float* alpha, const float*& d_A, int,
                   const float*& d_B, int, float* beta, float*& d_C, int);
void my_cuda_Dgemv(cublasHandle_t, cublasOperation_t, int M, int N,
                   double* alpha, const double*& d_A, const double*& d_B,
                   double* beta, double*& d_C);
void my_cuda_Dgemv(cublasHandle_t, cublasOperation_t, int M, int N,
                   float* alpha, const float*& d_A, const float*& d_B,
                   float* beta, float*& d_C);
void add_vec_to_mat_colwise(int, int, double*, const double*, double);
void add_vec_to_mat_colwise(int, int, float*, const float*, float);
void add_vec_to_mat_colwise(int, int, const double*, const double*, double*,
                            double);
void add_vec_to_mat_colwise(int, int, const float*, const float*, float*,
                            float);
void exponential(int, int, double*);
void exponential(int, int, float*);
void divide_colwise(int, int, double*, const double*);
void divide_colwise(int, int, float*, const float*);
void relu(int, int, double*, const double*);
void relu(int, int, float*, const float*);
void relu_backwards(int, int, const double*, const double*, double*);
void relu_backwards(int, int, const float*, const float*, float*);
void all_cross_entropy_losses(int, int, const double*, const double*, double*);
void all_cross_entropy_losses(int, int, const float*, const float*, float*);
void sum_cross_entropy_losses(int, float*, const float*);
void sum_cross_entropy_losses(int, double*, const double*);
#endif
