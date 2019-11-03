#include "../include/math.h"
#include <iostream>
#include <memory>
#include "../include/cuda_math.h"
void my_Dgemm(cublasHandle_t handle, cublasOperation_t transA,
              cublasOperation_t transB, const SharedStorage& A,
              const SharedStorage& B, SharedStorage& C, double alpha,
              double beta) {
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
    //int N = B->get_cols();
    //int K = A->get_cols();
    const double* d_A = A->gpu_pointer_const();
    const double* d_B = B->gpu_pointer_const();
    double* d_C = C->gpu_pointer();
    my_cuda_Dgemm(handle, transA, transB, M, N, K, &alpha, d_A, LDA, d_B, LDB,
            &beta, d_C, LDC);
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

void my_add_vec_to_mat_colwise(SharedStorage&A, const SharedStorage& B) {
    int rows = A->get_rows();
    int cols = A->get_cols();
    double* d_A = A->gpu_pointer();
    const double* d_B = B->gpu_pointer_const();
    add_vec_to_mat_colwise(rows, cols, d_A, d_B);

}
