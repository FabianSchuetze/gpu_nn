#ifndef cuda_math_h
#define cuda_math_h
#include "cublas_v2.h"
void cuda_mathmult(cublasHandle_t handle, int M, int N, int K, double* alpha,
                   double*& d_A, double*& d_B, double* beta, double*& d_C);
#endif
