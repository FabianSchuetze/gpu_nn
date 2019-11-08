#include "../include/math.h"
#include <iostream>
#include <memory>
#include "../include/cuda_math.h"
#include "../include/common.h"
void my_Dgemm(cublasHandle_t handle, cublasOperation_t transA,
              cublasOperation_t transB, const SharedStorage& A,
              const SharedStorage& B, SharedStorage& C, dtype alpha,
              dtype beta) {
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
    // int N = B->get_cols();
    // int K = A->get_cols();
    const dtype* d_A = A->gpu_pointer_const();
    const dtype* d_B = B->gpu_pointer_const();
    dtype* d_C = C->gpu_pointer();
    my_cuda_Dgemm(handle, transA, transB, M, N, K, &alpha, d_A, LDA, d_B, LDB,
                  &beta, d_C, LDC);
}


void my_Dgemv(cublasHandle_t handle, cublasOperation_t transA,
              const SharedStorage& A, const SharedStorage& B, SharedStorage& C,
              dtype alpha, dtype beta) {
    int M = A->get_rows();
    int N = A->get_cols();
    const dtype* d_A = A->gpu_pointer_const();
    const dtype* d_B = B->gpu_pointer_const();
    dtype* d_C = C->gpu_pointer();
    my_cuda_Dgemv(handle, transA, M, N, &alpha, d_A, d_B, &beta, d_C);
}

void my_add_vec_to_mat_colwise(SharedStorage& A, const SharedStorage& B,
                               dtype alpha) {
    int rows = A->get_rows();
    int cols = A->get_cols();
    dtype* d_A = A->gpu_pointer();
    const dtype* d_B = B->gpu_pointer_const();
    add_vec_to_mat_colwise(rows, cols, d_A, d_B, alpha);
    // cudaDeviceSyncronize();
}

void my_add_vec_to_mat_colwise(const SharedStorage& in, const SharedStorage& B,
                               SharedStorage& out, dtype alpha) {
    int rows = in->get_rows();
    int cols = in->get_cols();
    const dtype* d_A = in->gpu_pointer_const();
    const dtype* d_B = B->gpu_pointer_const();
    dtype* d_C = out->gpu_pointer();
    add_vec_to_mat_colwise(rows, cols, d_A, d_B, d_C, alpha);
    // cudaDeviceSyncronize();
}

void my_Exponential(SharedStorage& in) {
    int rows = in->get_rows();
    int cols = in->get_cols();
    dtype* d_A = in->gpu_pointer();
    exponential(rows, cols, d_A);
}

void my_Divide_colwise(SharedStorage& in, const SharedStorage& vec) {
    int rows = in->get_rows();
    int cols = in->get_cols();
    dtype* d_A = in->gpu_pointer();
    const dtype* d_B = vec->gpu_pointer_const();
    divide_colwise(rows, cols, d_A, d_B);
}

void my_relu(SharedStorage& in, const SharedStorage& vec) {
    int rows = in->get_rows();
    int cols = in->get_cols();
    dtype* d_a = in->gpu_pointer();
    const dtype* d_b = vec->gpu_pointer_const();
    relu(rows, cols, d_a, d_b);
}

void my_relu_backwards(const SharedStorage& values,
                       const SharedStorage& grad_in, SharedStorage& grad_out) {
    int rows = values->get_rows();
    int cols = values->get_cols();
    const dtype* d_A = values->gpu_pointer_const();
    const dtype* d_B = grad_in->gpu_pointer_const();
    dtype* d_C = grad_out->gpu_pointer();
    relu_backwards(rows, cols, d_A, d_B, d_C);
}

void my_cross_entropy_loss(dtype& loss, const SharedStorage& prediction,
                             const SharedStorage& actual) {
    int cols = prediction->get_cols();
    int rows = prediction->get_rows();
    SharedStorage all_losses = std::make_shared<Storage>(Matrix::Zero(cols, 1));
    const dtype* d_A = prediction->gpu_pointer_const();
    const dtype* d_B = actual->gpu_pointer_const();
    dtype* d_C = all_losses->gpu_pointer();
    all_cross_entropy_losses(rows, cols, d_A, d_B, d_C);
    loss = all_losses->return_data_const().sum();
}
