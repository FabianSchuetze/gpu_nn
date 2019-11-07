#include "../include/common.h"
#include "../include/cuda_math.h"

void my_cuda_Dgemm(cublasHandle_t handle, cublasOperation_t transA,
                   cublasOperation_t transB, int M, int N, int K, double* alpha,
                   const double*& d_A, int LDA, const double*& d_B, int LDB,
                   double* beta, double*& d_C, int LDC) {
    // M defines the number of rows in Matrix A and C
    // N Defines the number of columns of the Matrix B and C
    // K defiens the number of columns of the Matrhx A and rows of Matix B
    CHECK_CUBLAS(cublasDgemm(handle, transA, transB, M, N, K, alpha, d_A, LDA,
                             d_B, LDB, beta, d_C, LDC));
}
void my_cuda_Dgemm(cublasHandle_t handle, cublasOperation_t transA,
                   cublasOperation_t transB, int M, int N, int K, float* alpha,
                   const float*& d_A, int LDA, const float*& d_B, int LDB,
                   float* beta, float*& d_C, int LDC) {
    // M defines the number of rows in Matrix A and C
    // N Defines the number of columns of the Matrix B and C
    // K defiens the number of columns of the Matrhx A and rows of Matix B
    CHECK_CUBLAS(cublasSgemm(handle, transA, transB, M, N, K, alpha, d_A, LDA,
                             d_B, LDB, beta, d_C, LDC));
}

void my_cuda_Dgemv(cublasHandle_t handle, cublasOperation_t transA, int M,
                   int N, double* alpha, const double*& d_A, const double*& d_B,
                   double* beta, double*& d_C) {
    // M defines the number of rows in Matrix A and C
    // N Defines the number of columns of the Matrix B and C
    CHECK_CUBLAS(
        cublasDgemv(handle, transA, M, N, alpha, d_A, M, d_B, 1, beta, d_C, 1));
    // cudaDeviceSynchronize();

    // WHAT ABOUT SYNRONIZING THE DEVICE?
}

void my_cuda_Dgemv(cublasHandle_t handle, cublasOperation_t transA, int M,
                   int N, float* alpha, const float*& d_A, const float*& d_B,
                   float* beta, float*& d_C) {
    // M defines the number of rows in Matrix A and C
    // N Defines the number of columns of the Matrix B and C
    CHECK_CUBLAS(
        cublasSgemv(handle, transA, M, N, alpha, d_A, M, d_B, 1, beta, d_C, 1));
    // cudaDeviceSynchronize();

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

__global__ void add_vec_to_mat_colwise_cu(int rows, int cols, float* matrix,
                                          const float* vector, float alpha) {
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
        out[idx] = in[idx] + alpha * vector[idx / rows];
    }
}

__global__ void add_vec_to_mat_colwise_cu(int rows, int cols, const float* in,
                                          const float* vector, float* out,
                                          float alpha) {
    // get the current element index for the thread
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        out[idx] = in[idx] + alpha * vector[idx / rows];
    }
}
__global__ void cuda_exponential(int rows, int cols, double* in) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        in[idx] = exp(in[idx]);
    }
}

__global__ void cuda_exponential(int rows, int cols, float* in) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        in[idx] = exp(in[idx]);
    }
}

__global__ void cuda_divide_colwise(int rows, int cols, float* in,
                                    const float* vec) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        in[idx] /= vec[idx / rows];
    }
}

__global__ void cuda_divide_colwise(int rows, int cols, double* in,
                                    const double* vec) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        in[idx] /= vec[idx / rows];
    }
}

__global__ void cuda_relu(int rows, int cols, double* out, const double* in) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        out[idx] = (in[idx] > 0) ? in[idx] : 0.;
    }
}

__global__ void cuda_relu(int rows, int cols, float* out, const float* in) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        out[idx] = (in[idx] > 0) ? in[idx] : 0.;
    }
}

__global__ void cuda_relu_backwards(int rows, int cols, const double* values,
                                    const double* grad_in, double* grad_out) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        grad_out[idx] = (values[idx] > 0) ? grad_in[idx] : 0.;
    }
}

__global__ void cuda_relu_backwards(int rows, int cols, const float* values,
                                    const float* grad_in, float* grad_out) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        grad_out[idx] = (values[idx] > 0) ? grad_in[idx] : 0.;
    }
}

__global__ void cuda_all_cross_entropy_losses(int rows, int cols,
                                              const double* prediction,
                                              const double* actual,
                                              double* losses) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int linear = row + col * rows;
    if ((row < rows) && (col < cols)) {
        if (actual[linear] == 1) {
            losses[linear / rows] = -1 * log(prediction[linear]);
        }
    }
}

__global__ void cuda_all_cross_entropy_losses(int rows, int cols,
                                              const float* prediction,
                                              const float* actual,
                                              float* losses) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int linear = row + col * rows;
    if ((row < rows) && (col < cols)) {
        if (actual[linear] == 1) {
            losses[linear / rows] = -1 * log(prediction[linear]);
        }
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

void add_vec_to_mat_colwise(int rows, int cols, float* matrix,
                            const float* vector, float alpha) {
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

void add_vec_to_mat_colwise(int rows, int cols, const float* in,
                            const float* vector, float* out, float alpha) {
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

void exponential(int rows, int cols, float* in) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_exponential<<<grid, block>>>(rows, cols, in);
}

void divide_colwise(int rows, int cols, double* in, const double* vec) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_divide_colwise<<<grid, block>>>(rows, cols, in, vec);
}

void divide_colwise(int rows, int cols, float* in, const float* vec) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_divide_colwise<<<grid, block>>>(rows, cols, in, vec);
}

void relu(int rows, int cols, double* out, const double* in) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_relu<<<grid, block>>>(rows, cols, out, in);
}
void relu(int rows, int cols, float* out, const float* in) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_relu<<<grid, block>>>(rows, cols, out, in);
}

void relu_backwards(int rows, int cols, const double* values,
                    const double* grad_in, double* grad_out) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_relu_backwards<<<grid, block>>>(rows, cols, values, grad_in, grad_out);
    cudaDeviceSynchronize();
}
void relu_backwards(int rows, int cols, const float* values,
                    const float* grad_in, float* grad_out) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_relu_backwards<<<grid, block>>>(rows, cols, values, grad_in, grad_out);
    cudaDeviceSynchronize();
}

void all_cross_entropy_losses(int rows, int cols, const double* prediction,
                              const double* actual, double* losses) {
    dim3 block(16, 16);
    dim3 grid(ceil(cols / 16), ceil(rows / 16));
    cuda_all_cross_entropy_losses<<<grid, block>>>(rows, cols, prediction,
                                                   actual, losses);
}

void all_cross_entropy_losses(int rows, int cols, const float* prediction,
                              const float* actual, float* losses) {
    dim3 block(16, 16);
    dim3 grid(ceil(cols / 16), ceil(rows / 16));
    cuda_all_cross_entropy_losses<<<grid, block>>>(rows, cols, prediction,
                                                   actual, losses);
}
