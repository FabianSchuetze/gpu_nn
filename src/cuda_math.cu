#include "../include/common.h"
#include "../include/cuda_math.h"

void cuda_mathmult(cublasHandle_t handle, int M, int N, int K, double* alpha,
                    const double*& d_A, const double*& d_B, double* beta,
                   double*& d_C) {
    // M defines the number of rows in Matrix A and C
    // N Defines the number of columns of the Matrix B and C
    // K defiens the number of columns of the Matrhx A and rows of Matix B
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, alpha, d_A, M, d_B,
                K, beta, d_C, M);
}
