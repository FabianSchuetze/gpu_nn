#include "../include/common.h"
#include "../include/cuda_math.h"

void my_cuda_Dgemm(cublasHandle_t handle, cublasOperation_t transA,
                   cublasOperation_t transB, int M, int N, int K, double* alpha,
                   const double*& d_A, const double*& d_B, double* beta,
                   double*& d_C) {
    // M defines the number of rows in Matrix A and C
    // N Defines the number of columns of the Matrix B and C
    // K defiens the number of columns of the Matrhx A and rows of Matix B
    cublasDgemm(handle, transA, transB, M, N, K, alpha, d_A, M, d_B, K, beta,
                d_C, M);
}

void my_cuda_Dgemv(cublasHandle_t handle, cublasOperation_t transA, int M,
                   int N, double* alpha, const double*& d_A, const double*& d_B,
                   double* beta, double*& d_C) {
    // M defines the number of rows in Matrix A and C
    // N Defines the number of columns of the Matrix B and C
    cublasDgemv(handle, transA, M, N, alpha, d_A, M, d_B, 1, beta, d_C, 1);
}
