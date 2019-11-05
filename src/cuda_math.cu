#include "../include/common.h"
#include "../include/cuda_math.h"

void my_cuda_Dgemm(cublasHandle_t handle, cublasOperation_t transA,
                   cublasOperation_t transB, int M, int N, int K, double* alpha,
                   const double*& d_A, int LDA, const double*& d_B, int LDB,
                   double* beta, double*& d_C, int LDC) {
    // M defines the number of rows in Matrix A and C
    // N Defines the number of columns of the Matrix B and C
    // K defiens the number of columns of the Matrhx A and rows of Matix B
    cublasDgemm(handle, transA, transB, M, N, K, alpha, d_A, LDA, d_B, LDB,
                beta, d_C, LDC);
}

void my_cuda_Dgemv(cublasHandle_t handle, cublasOperation_t transA, int M,
                   int N, double* alpha, const double*& d_A, const double*& d_B,
                   double* beta, double*& d_C) {
    // M defines the number of rows in Matrix A and C
    // N Defines the number of columns of the Matrix B and C
    cublasDgemv(handle, transA, M, N, alpha, d_A, M, d_B, 1, beta, d_C, 1);
    // WHAT ABOUT SYNRONIZING THE DEVICE?
}

__global__ void add_vec_to_mat_colwise_cu(int rows, int cols, double* matrix,
                                          const double* vector, double alpha) {
    // get the current element index for the thread
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        matrix[idx] += alpha * vector[idx % rows];
    }
}

__global__ void add_vec_to_mat_colwise_cu(int rows, int cols, const double* in,
                                          const double* vector, double* out,
                                          double alpha) {
    // get the current element index for the thread
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        out[idx] = in[idx] + alpha * vector[idx % rows];
    }
}

__global__ void cuda_exponential(int rows, int cols, double* in) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        in[idx] = exp(in[idx]);
    }
}

__global__ void cuda_divide_colwise(int rows, int cols, double* in,
                                    const double* vec) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        in[idx] /= vec[idx / rows];
    }
}

void add_vec_to_mat_colwise(int rows, int cols, double* matrix,
                            const double* vector, double alpha) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    add_vec_to_mat_colwise_cu<<<grid, block>>>(rows, cols, matrix, vector,
                                               alpha);
    // cudaDeviceSynronize();
}

void add_vec_to_mat_colwise(int rows, int cols, const double* in,
                            const double* vector, double* out, double alpha) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    add_vec_to_mat_colwise_cu<<<grid, block>>>(rows, cols, in, vector, out,
                                               alpha);
    // cudaDeviceSynronize();
}

void exponential(int rows, int cols, double* in) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_exponential<<<grid, block>>>(rows, cols, in);
}

void divide_colwise(int rows, int cols, double* in, const double* vec) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_divide_colwise<<<grid, block>>>(rows, cols, in, vec);
}
