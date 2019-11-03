#include "../include/math.h"
#include <iostream>
#include <memory>
#include "../include/cuda_math.h"
void my_Dgemm(cublasHandle_t handle, cublasOperation_t transA,
              cublasOperation_t transB, const SharedStorage& A,
              const SharedStorage& B, SharedStorage& C, double alpha,
              double beta) {
    int M = A->get_rows();
    int N = B->get_cols();
    int K = A->get_cols();
    const double* d_A = A->gpu_pointer_const();
    const double* d_B = B->gpu_pointer_const();
    double* d_C = C->gpu_pointer();
    my_cuda_Dgemm(handle, transA, transB, M, N, K, &alpha, d_A, d_B, &beta,
                  d_C);
}

void my_Dgemv(cublasHandle_t handle, cublasOperation_t transA,
              const SharedStorage& A, const SharedStorage& B, SharedStorage& C,
              double alpha, double beta) {
    int M = A->get_rows();
    int N = A->get_cols();
    const double* d_A = A->gpu_pointer_const();
    const double* d_B = B->gpu_pointer_const();
    double* d_C = C->gpu_pointer();
    my_cuda_Dgemv(handle, transA, M, N, &alpha, d_A, d_B, &beta, d_C);
}
