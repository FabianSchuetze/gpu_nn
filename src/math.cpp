#include "../include/cuda_math.h"
#include "../include/math.h"
#include <memory>
#include <iostream>
void multonGPU(cublasHandle_t handle, const SharedStorage& A, const SharedStorage& B,
               SharedStorage& C, double alpha, double beta) {
    int M = A->get_rows();
    int N = B->get_cols();
    int K = A->get_cols();
    const double* d_A = A->gpu_pointer_const();
    const double* d_B = B->gpu_pointer_const();
    double* d_C = C->gpu_pointer();
    cuda_mathmult(handle, M, N, K, &alpha, d_A, d_B, &beta, d_C);
}
