#include <curand.h>
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
    // MY_CHECK(cudaDeviceSynchronize());
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
    // MY_CHECK(cudaDeviceSynchronize());
}

void my_cuda_Dgemv(cublasHandle_t handle, cublasOperation_t transA, int M,
                   int N, double* alpha, const double*& d_A, const double*& d_B,
                   double* beta, double*& d_C) {
    // M defines the number of rows in Matrix A and C
    // N Defines the number of columns of the Matrix B and C
    CHECK_CUBLAS(
        cublasDgemv(handle, transA, M, N, alpha, d_A, M, d_B, 1, beta, d_C, 1));
    // MY_CHECK(cudaDeviceSynchronize());

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
    // MY_CHECK(cudaDeviceSynchronize());

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

__global__ void cuda_cross_entropy_gradient(int rows, int cols,
                                            const float* prediction,
                                            const float* actual,
                                            float* gradient) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int linear = row + col * rows;
    if ((row < rows) && (col < cols)) {
        gradient[linear] = prediction[linear] - actual[linear];
    }
}

__global__ void cuda_cross_entropy_gradient(int rows, int cols,
                                            const double* prediction,
                                            const double* actual,
                                            double* gradient) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int linear = row + col * rows;
    if ((row < rows) && (col < cols)) {
        gradient[linear] = prediction[linear] - actual[linear];
    }
}

// I NEED TO THINK HOW I DO SUCH A SUM BETTER - Reduction!!!
__global__ void cuda_sum_cross_entropy_losses(int obs, float* loss,
                                              const float* all_losses) {
    unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear == 0) {
        for (int i = 0; i < obs; ++i) loss[0] += all_losses[i];
    }
}

__global__ void cuda_sum_cross_entropy_losses(int obs, double* loss,
                                              const double* all_losses) {
    unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear == 0) {
        for (int i = 0; i < obs; ++i) loss[0] += all_losses[i];
    }
}

__global__ void cuda_matrix_addition_inplace(int rows, int cols,
                                             const float* d_A, float* d_B,
                                             const float alpha) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int linear = row + col * rows;
    if ((row < rows) && (col < cols)) {
        // printf("row %d, col %d, linear %d, old_val %.3f, gradient %.3f\n",
        // row, col, linear, d_B[linear], d_A[linear]);
        d_B[linear] += alpha * d_A[linear];
    }
}

__global__ void cuda_matrix_addition_inplace(int rows, int cols,
                                             const double* d_A, double* d_B,
                                             const double alpha) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int linear = row + col * rows;
    if ((row < rows) && (col < cols)) {
        d_B[linear] += alpha * d_A[linear];
    }
}

__global__ void multiply_ele(int rows, int cols, const float* d_A,
                             const float* d_B, float* d_C) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int linear = row + col * rows;
    if ((row < rows) && (col < cols)) {
        d_C[linear] = d_A[linear] * d_B[linear];
    }
}

__global__ void multiply_ele(int rows, int cols, const double* d_A,
                             const double* d_B, double* d_C) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int linear = row + col * rows;
    if ((row < rows) && (col < cols)) {
        d_C[linear] = d_A[linear] * d_B[linear];
    }
}

__global__ void masking(int rows, int cols, const float prob, float* d_A) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int linear = row + col * rows;
    if ((row < rows) && (col < cols)) {
        d_A[linear] = (d_A[linear] < prob) ? 1 / prob : 0.;
    }
}

__global__ void masking(int rows, int cols, const double prob, double* d_A) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int linear = row + col * rows;
    if ((row < rows) && (col < cols)) {
        d_A[linear] = (d_A[linear] < prob) ? 1 / prob : 0.;
    }
}

__global__ void MaxPoolBackward(const float* top_diff, const float* mask,
                                int window, int stride, const int rows,
                                int cols, int channels, int batches,
                                float* bottom_diff) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x);
    int col = (blockIdx.y * blockDim.y + threadIdx.y);
    int c = (blockIdx.z * blockDim.z + threadIdx.z);
    int out_height = (rows - window) / stride + 1;
    int out_width = (cols - window) / stride + 1;
    if (row < rows && col < cols && c < channels) {
        const int phstart = (row < window) ? 0 : (row - window) / stride + 1;
        const int phend = min((row) / stride + 1, out_height);
        const int pwstart = (cols < window) ? 0 : (col - window) / stride + 1;
        const int pwend = min((col) / stride + 1, out_width);
        const int idx = c * rows * cols + row * cols + col;
        for (int n = 0; n < batches; ++n) {
            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                    int li = out_width * (c * out_height + ph) + pw;
                    if (mask[li] == row * cols + col) {
                        bottom_diff[idx] += top_diff[li];
                    }
                }
            }
            mask += out_width * out_height * channels;
            top_diff += out_width * out_height * channels;
            bottom_diff += channels * rows * cols;
        }
    }
}

__global__ void CudaPooling(float const* inp, int window, int stride, int rows,
                            int cols, int channels, int batches, float* out,
                            float* mask) {
    int ph = (blockIdx.x * blockDim.x + threadIdx.x);
    int pw = (blockIdx.y * blockDim.y + threadIdx.y);
    int c = (blockIdx.z * blockDim.z + threadIdx.z);
    int out_height = (rows - window) / stride + 1;
    int out_width = (cols - window) / stride + 1;
    if (ph < out_height && pw < out_width && c < channels) {
        for (int n = 0; n < batches; n++) {
            for (int i = 0; i < window; i++) {
                for (int j = 0; j < window; j++) {
                    int curRow = ph * stride + i;
                    int curCol = pw * stride + j;
                    int li = c * cols * rows + curRow * cols + curCol;
                    int lo = c * out_width * out_height + ph * out_width + pw;
                    if (inp[li] > out[lo]) {
                        out[lo] = inp[li];
                        mask[lo] = li % (cols * rows);
                    }
                }
            }
            inp += rows * cols * channels;
            out += out_width * out_height * channels;
            mask += out_width * out_height * channels;
        }
    }
}

void add_vec_to_mat_colwise(int rows, int cols, double* matrix,
                            const double* vector, double alpha) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    add_vec_to_mat_colwise_cu<<<grid, block>>>(rows, cols, matrix, vector,
                                               alpha);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
    // cudaDeviceSynronize();
}

void add_vec_to_mat_colwise(int rows, int cols, float* matrix,
                            const float* vector, float alpha) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    add_vec_to_mat_colwise_cu<<<grid, block>>>(rows, cols, matrix, vector,
                                               alpha);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
    // cudaDeviceSynronize();
}

void add_vec_to_mat_colwise(int rows, int cols, const double* in,
                            const double* vector, double* out, double alpha) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    add_vec_to_mat_colwise_cu<<<grid, block>>>(rows, cols, in, vector, out,
                                               alpha);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
    // cudaDeviceSynronize();
}

void add_vec_to_mat_colwise(int rows, int cols, const float* in,
                            const float* vector, float* out, float alpha) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    add_vec_to_mat_colwise_cu<<<grid, block>>>(rows, cols, in, vector, out,
                                               alpha);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
    // cudaDeviceSynronize();
}

void exponential(int rows, int cols, double* in) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_exponential<<<grid, block>>>(rows, cols, in);
    MY_CHECK(cudaPeekAtLastError());
    MY_CHECK(cudaDeviceSynchronize());
}

void exponential(int rows, int cols, float* in) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_exponential<<<grid, block>>>(rows, cols, in);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}

void divide_colwise(int rows, int cols, double* in, const double* vec) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_divide_colwise<<<grid, block>>>(rows, cols, in, vec);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}

void divide_colwise(int rows, int cols, float* in, const float* vec) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_divide_colwise<<<grid, block>>>(rows, cols, in, vec);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}

void relu(int rows, int cols, double* out, const double* in) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_relu<<<grid, block>>>(rows, cols, out, in);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}
void relu(int rows, int cols, float* out, const float* in) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_relu<<<grid, block>>>(rows, cols, out, in);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}

void relu_backwards(int rows, int cols, const double* values,
                    const double* grad_in, double* grad_out) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_relu_backwards<<<grid, block>>>(rows, cols, values, grad_in, grad_out);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}
void relu_backwards(int rows, int cols, const float* values,
                    const float* grad_in, float* grad_out) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    cuda_relu_backwards<<<grid, block>>>(rows, cols, values, grad_in, grad_out);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}

void all_cross_entropy_losses(int rows, int cols, const double* prediction,
                              const double* actual, double* losses) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    cuda_all_cross_entropy_losses<<<grid, block>>>(rows, cols, prediction,
                                                   actual, losses);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}

void all_cross_entropy_losses(int rows, int cols, const float* prediction,
                              const float* actual, float* losses) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    cuda_all_cross_entropy_losses<<<grid, block>>>(rows, cols, prediction,
                                                   actual, losses);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}

void sum_cross_entropy_losses(int obs, float* loss, const float* all_losses) {
    dim3 block(256);
    dim3 grid((obs + block.x - 1) / block.x);
    cuda_sum_cross_entropy_losses<<<grid, block>>>(obs, loss, all_losses);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}

void sum_cross_entropy_losses(int obs, double* loss, const double* all_losses) {
    dim3 block(256);
    dim3 grid((obs + block.x - 1) / block.x);
    cuda_sum_cross_entropy_losses<<<grid, block>>>(obs, loss, all_losses);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}
void cross_entropy_gradient(int rows, int cols, const double* prediction,
                            const double* target, double* gradient) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    cuda_cross_entropy_gradient<<<grid, block>>>(rows, cols, prediction, target,
                                                 gradient);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}
void cross_entropy_gradient(int rows, int cols, const float* prediction,
                            const float* target, float* gradient) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    cuda_cross_entropy_gradient<<<grid, block>>>(rows, cols, prediction, target,
                                                 gradient);
    // cudaDeviceSynchronize();
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}

void matrix_addition_inplace(int rows, int cols, const float* gradient,
                             float* parameters, const float alpha) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    cuda_matrix_addition_inplace<<<grid, block>>>(rows, cols, gradient,
                                                  parameters, alpha);
    // cudaDeviceSynchronize();
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}

void matrix_addition_inplace(int rows, int cols, const double* gradient,
                             double* parameters, const double alpha) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    cuda_matrix_addition_inplace<<<grid, block>>>(rows, cols, gradient,
                                                  parameters, alpha);
    // cudaDeviceSynchronize();
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}

void multiply_elementwise(int rows, int cols, const float* d_A,
                          const float* d_B, float* d_C) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    multiply_ele<<<grid, block>>>(rows, cols, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    MY_CHECK(cudaPeekAtLastError());
}

void multiply_elementwise(int rows, int cols, const double* d_A,
                          const double* d_B, double* d_C) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    multiply_ele<<<grid, block>>>(rows, cols, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    MY_CHECK(cudaPeekAtLastError());
}

void cuda_masking(int rows, int cols, const float prob, float* d_A) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    masking<<<grid, block>>>(rows, cols, prob, d_A);
    cudaDeviceSynchronize();
    MY_CHECK(cudaPeekAtLastError());
}

void cuda_masking(int rows, int cols, const double prob, double* d_A) {
    dim3 block(16, 16);
    dim3 grid((rows + block.y - 1) / block.y, (cols + block.x - 1) / block.x);
    masking<<<grid, block>>>(rows, cols, prob, d_A);
    cudaDeviceSynchronize();
    MY_CHECK(cudaPeekAtLastError());
}

void pooling_gpu(const float* bottom_data, const int window,
                 const int stride, int rows, int cols, const int channels,
                 const int batches, float* top_data, float* mask) {
    if (((rows - window) % stride) or ((cols - window) % stride)) {
        throw std::invalid_argument("Doesnt match");
    }
    dim3 block(16, 16, 4);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y,
              (channels + block.z -1) / block.z);
    CudaPooling<<<grid, block>>>(bottom_data, window, stride, rows, cols,
                                 channels, batches, top_data, mask);
    //MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}

void pooling_backward_gpu(const float* bottom_data, const float* mask,
                          const int window, const int stride, int rows,
                          int cols, const int channels, const int batches,
                          float* dest) {
    if (((rows - window) % stride) or ((cols - window) % stride)) {
        throw std::invalid_argument("Doesnt match");
    }
    dim3 block(16, 16, 4);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y,
              (channels + block.z -1) / block.z);
    MaxPoolBackward<<<grid, block>>>(bottom_data, mask, window, stride, rows,
                                     cols, channels, batches, dest);
    //MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}
