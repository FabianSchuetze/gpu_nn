#ifndef cuda_math_h
#define cuda_math_h
#include "cublas_v2.h"
void my_cuda_Dgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int M,
                   int N, int K, double* alpha, const double*& d_A, int,
                   const double*& d_B, int, double* beta, double*& d_C, int);
void my_cuda_Dgemv(cublasHandle_t, cublasOperation_t, int M, int N,
                   double* alpha, const double*& d_A, const double*& d_B,
                   double* beta, double*& d_C);
void add_vec_to_mat_colwise(int, int, double*, const double*);
#endif
