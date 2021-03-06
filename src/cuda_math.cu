#include <curand.h>
#include <float.h>
#include <sys/time.h>
#include "../include/common.h"
#include "../include/cuda_math.h"
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void my_cuda_Dgemm(cublasHandle_t handle, cublasOperation_t transA,
                   cublasOperation_t transB, int M, int N, int K, double* alpha,
                   const double*& d_A, int LDA, const double*& d_B, int LDB,
                   double* beta, double*& d_C, int LDC) {
    //  C = α op ( A ) op ( B ) + β C
    // M defines the number of rows in Matrix op(A) and C
    // N Defines the number of columns of the Matrix op(B) and C
    // K defiens the number of columns of the Matrhx op(A) and rows of Matix B
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
}

void my_cuda_Dgemv(cublasHandle_t handle, cublasOperation_t transA, int M,
                   int N, float* alpha, const float*& d_A, const float*& d_B,
                   float* beta, float*& d_C) {
    // M defines the number of rows in Matrix A and C
    // N Defines the number of columns of the Matrix B and C
    CHECK_CUBLAS(
        cublasSgemv(handle, transA, M, N, alpha, d_A, M, d_B, 1, beta, d_C, 1));
    // MY_CHECK(cudaDeviceSynchronize());
}

__global__ void add_vec_to_mat_colwise_cu(int rows, int cols, dtype* matrix,
                                          const dtype* vector, dtype alpha) {
    // get the current element index for the thread
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        matrix[idx] += alpha * vector[idx % rows];
    }
}

__global__ void add_vec_to_mat_colwise_cu(int rows, int cols, const dtype* in,
                                          const dtype* vector, dtype* out,
                                          dtype alpha) {
    // get the current element index for the thread
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        out[idx] = in[idx] + alpha * vector[idx / rows];
    }
}

//__global__ void cuda_exponential(int rows, int cols, double* in) {
    //unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if (idx < rows * cols) {
        //in[idx] = exp(in[idx]);
    //}
//}

__global__ void cuda_exponential(int rows, int cols, dtype* in) {
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
    if ((row < rows) && (col < cols)) {
        unsigned int linear = row + col * rows;
        d_B[linear] += alpha * d_A[linear];
    }
}

__global__ void cuda_matrix_addition(int rows, int cols, const float* d_A,
                                     const float* d_B, float* d_C,
                                     const float alpha_A, const float alpha_B) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < rows) && (col < cols)) {
        unsigned int linear = row + col * rows;
        float tmp = alpha_B * d_B[linear] + alpha_A * d_A[linear];
        d_C[linear] = tmp;
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

__global__ void multiply_ele(int rows, int cols, const dtype* d_A,
                             const dtype* d_B, dtype* d_C) {
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

__global__ void MaxPoolBackward(const int nthreads, const dtype* const top_diff,
                                const float* const mask, const int num,
                                const int channels, const int height,
                                const int width, const int pooled_height,
                                const int pooled_width, const int window,
                                const int stride, dtype* const bottom_diff) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nthreads) {
        // CUDA_KERNEL_LOOP(index, nthreads) {
        // find out the local index
        // find out the local offset
        const int h = index % height;
        const int w = (index / height) % width;
        const int c = (index / height / width) % channels;
        const int n = index / height / width / channels;
        const int phstart = (h < window) ? 0 : (h - window) / stride + 1;
        const int phend = min(h / stride + 1, pooled_height);
        const int pwstart = (w < window) ? 0 : (w - window) / stride + 1;
        const int pwend = min(w / stride + 1, pooled_width);
        dtype gradient = 0;
        const int offset = (n * channels + c) * pooled_height * pooled_width;
        const dtype* const top_diff_slice = top_diff + offset;
        const float* const mask_slice = mask + offset;
        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                if (mask_slice[pw * pooled_height + ph] == w * height + h) {
                    gradient += top_diff_slice[pw * pooled_height + ph];
                }
            }
        }
        bottom_diff[index] = gradient;
    }
}

__global__ void MaxPoolForward(int nthreads, const dtype* bottom_data, int num,
                               int channels, int height, int width,
                               int out_height, int out_width, int window,
                               int stride, dtype* top_data, dtype* mask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nthreads) {
        const int ph = index % out_height;
        const int pw = (index / out_height) % out_width;
        const int c = (index / out_height / out_width) % channels;
        const int n = index / out_height / out_width / channels;
        int hstart = ph * stride;
        int wstart = pw * stride;
        const int hend = min(hstart + window, height);
        const int wend = min(wstart + window, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        dtype maxval = -FLT_MAX;
        int maxidx = -1;
        const dtype* bottom_slice =
            bottom_data + (n * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                if (bottom_slice[w * height + h] > maxval) {
                    maxidx = w * height + h;
                    maxval = bottom_slice[maxidx];
                }
            }
        }
        top_data[index] = maxval;
        mask[index] = maxidx;
    }
}

__global__ void im2col_gpu_kernel(int numThreads, const dtype* data_im,
                                  const int height, const int width,
                                  const int kernel_h, const int kernel_w,
                                  const int pad, const int stride,
                                  const int out_height, const int out_width,
                                  dtype* data_col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numThreads) {
        const int h_index = index / out_width;
        const int h_col = h_index % out_height;
        const int w_col = index % out_width;
        const int c_im = h_index / out_height;
        const int c_col = c_im * kernel_h * kernel_w;
        const int h_offset = h_col * stride - pad;
        const int w_offset = w_col * stride - pad;
        dtype* data_col_ptr = data_col;
        data_col_ptr += (c_col * out_height + h_col) * out_width + w_col;
        const dtype* data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h_im = h_offset + i;
                int w_im = w_offset + j;
                *data_col_ptr =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                        ? data_im_ptr[i * width + j]
                        : 0;
                data_col_ptr += out_height * out_width;
            }
        }
    }
}

__global__ void col2im_gpu_kernel(int numThreads, const dtype* data_col,
                                  int height, int width, int channels,
                                  int kernel_h, int kernel_w, const int pad,
                                  const int stride, const int height_col,
                                  const int width_col, dtype* data_im) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numThreads) {
        dtype val = 0;
        int w_im = index % width + pad;
        int h_im = (index / width) % height + pad;
        int c_im = index / (width * height);
        // compute the start and end of the output
        int w_col_start =
            (w_im < kernel_w) ? 0 : (w_im - kernel_w) / stride + 1;
        int w_col_end = min(w_im / stride + 1, width_col);
        int h_col_start =
            (h_im < kernel_h) ? 0 : (h_im - kernel_h) / stride + 1;
        int h_col_end = min(h_im / stride + 1, height_col);
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                int h_k = (h_im - h_col * stride);
                int w_k = (w_im - w_col * stride);
                // if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                // h_k /= dilation_h;
                // w_k /= dilation_w;
                int data_col_index =
                    (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col +
                     h_col) *
                        width_col +
                    w_col;
                val += data_col[data_col_index];
            }
        }
        data_im[index] = val;
    }
}

__global__ void colwise_max(const dtype* in, int rows, int cols, dtype* out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < cols) {
        dtype maxval = -FLT_MAX;
        for (int i = 0; i < rows; ++i) {
            int src_pos = index * rows + i;
            if (in[src_pos] > maxval) {
                maxval = in[src_pos];
            }
        }
        out[index] = maxval;
    }
}

__global__ void sigmoid(const dtype* in, int rows, int cols, dtype* out) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < rows) && (col < cols)) {
        unsigned int linear = row + col * rows;
        dtype c = in[linear];
        out[linear] =
            (c > 0) ? 1 / (1 + expf(-1 * c)) : expf(c) / (1 + expf(c));
    }
}

__global__ void tanh(const dtype* in, int rows, int cols, dtype* out) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < rows) && (col < cols)) {
        unsigned int linear = row + col * rows;
        out[linear] = tanhf(in[linear]);
    }
}

__global__ void next_lstm_cell(const dtype* funcs, int rows, dtype* out) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        dtype first = funcs[row + rows] * out[row];
        dtype second = funcs[row] * funcs[row + 3 * rows];
        out[row + rows] = first + second;
    }
}

__global__ void next_lstm_state(const dtype* funcs, const dtype* cell, int rows,
                                dtype* out) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        out[row + rows] = funcs[row + 2 * rows] * tanhf(cell[row + rows]);
    }
}

__global__ void cuda_compute_deriv_cell(int rows, int cols, const dtype* cell,
                                        dtype* deriv_cell) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < rows) && (col < cols)) {
        unsigned int linear = row + col * rows;
        deriv_cell[linear] = 1.0 - tanhf(cell[linear]) * tanhf(cell[linear]);
    }
}

__global__ void cuda_new_cell_state(int rows, const dtype* dcum_c,
                                    const dtype* dh, const dtype* o,
                                    const dtype* sigma_c, dtype* dc) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        dc[row] = dcum_c[row] + dh[row] * o[row] * sigma_c[row];
    }
}

__global__ void cuda_internal_deriv(int rows, const dtype* dh, const dtype* dc,
                                    const dtype* cell, const dtype* funcs,
                                    dtype* d_tmp) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        d_tmp[2 * rows + row] = dh[row] * tanhf(cell[row + rows]);
        d_tmp[row] = dc[row] * funcs[3 * rows + row];
        d_tmp[row + rows] = dc[row] * cell[row];
        d_tmp[3 * rows + row] = dc[row] * funcs[row];
    }
}

__global__ void cuda_copy_data(int rows, int cols, const dtype* in,
                               dtype* out) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < rows) && (col < cols)) {
        unsigned int linear = row + col * rows;
        out[linear] = in[linear];
    }
}

__global__ void cuda_sigmoid_deriv(int rows, int max_rows, int cols,
                                   const dtype* in, dtype* out) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < rows) && (col < cols)) {
        unsigned int linear = row + col * max_rows;
        out[linear] = in[linear] * (1 - in[linear]);
    }
}

__global__ void cuda_tanh_deriv(int rows, int max_rows, int cols,
                                const dtype* in, dtype* out) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < rows) && (col < cols)) {
        unsigned int linear = 3 * rows + row + col * max_rows;
        out[linear] = (1 - in[linear] * in[linear]);
    }
}

__global__ void cuda_clip_gradients(int rows, int cols, dtype max,
                                    dtype* grad) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < rows) && (col < cols)) {
        unsigned int linear = row + col * rows;
        dtype curr = grad[linear];
        if (curr < -1 * max) {
            grad[linear] = -1 * max;
        } else if (curr > max) {
            grad[linear] = max;
        }
    }
}

void add_vec_to_mat_colwise(int rows, int cols, dtype* matrix,
                            const dtype* vector, dtype alpha) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    add_vec_to_mat_colwise_cu<<<grid, block>>>(rows, cols, matrix, vector,
                                               alpha);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
    // cudaDeviceSynronize();
}

void add_vec_to_mat_colwise(int rows, int cols, const dtype* in,
                            const dtype* vector, dtype* out, dtype alpha) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    add_vec_to_mat_colwise_cu<<<grid, block>>>(rows, cols, in, vector, out,
                                               alpha);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}


void exponential(int rows, int cols, dtype* in) {
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

void matrix_addition(int rows, int cols, const dtype* A, const dtype* B,
                     dtype* C, const dtype alpha_A, const dtype alpha_B) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    cuda_matrix_addition<<<grid, block>>>(rows, cols, A, B, C, alpha_A,
                                          alpha_B);
    MY_CHECK(cudaPeekAtLastError());
    // MY_CHECK(cudaDeviceSynchronize());
}

void matrix_addition_inplace(int rows, int cols, const float* gradient,
                             float* parameters, const float alpha) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    cuda_matrix_addition_inplace<<<grid, block>>>(rows, cols, gradient,
                                                  parameters, alpha);
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

void multiply_elementwise(int rows, int cols, const dtype* d_A,
                          const dtype* d_B, dtype* d_C) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    multiply_ele<<<grid, block>>>(rows, cols, d_A, d_B, d_C);
    // cudaDeviceSynchronize();
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

void pooling_gpu(const float* bottom_data, int window, int stride, int rows,
                 int cols, int channels, int out_height, int out_width,
                 int batches, float* top_data, float* mask) {
    dim3 block(512);
    int eles = out_height * out_width * channels * batches;
    dim3 grid((eles + block.x - 1) / block.x);
    MaxPoolForward<<<grid, block>>>(eles, bottom_data, batches, channels, rows,
                                    cols, out_height, out_width, window, stride,
                                    top_data, mask);
    MY_CHECK(cudaPeekAtLastError());
}

void pooling_backward_gpu(const float* src, const float* mask, int window,
                          int stride, int rows, int cols, int channels,
                          int out_height, int out_width, int batches,
                          float* dest) {
    dim3 block(512);
    int eles = rows * cols * channels * batches;
    dim3 grid((eles + block.x - 1) / block.x);
    MaxPoolBackward<<<grid, block>>>(eles, src, mask, batches, channels, rows,
                                     cols, out_height, out_width, window,
                                     stride, dest);
    MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}

void im2col_gpu(const float* data_im, int channels, int height, const int width,
                int kernel_h, const int kernel_w, int pad, int stride,
                float* data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int out_height = (height + 2 * pad - kernel_h) / stride + 1;
    int out_width = (width + 2 * pad - kernel_w) / stride + 1;
    int numThreads = channels * out_height * out_width;
    dim3 block(512);
    dim3 grid((numThreads + block.x - 1) / block.x);
    im2col_gpu_kernel<<<grid, block>>>(numThreads, data_im, height, width,
                                       kernel_h, kernel_w, pad, stride,
                                       out_height, out_width, data_col);
    // MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}

void col2im_gpu(const dtype* data_col, int channels, int height, int width,
                int kernel_h, int kernel_w, int pad, int stride,
                dtype* data_im) {
    int out_height = (height + 2 * pad - kernel_h) / stride + 1;
    int out_width = (width + 2 * pad - kernel_w) / stride + 1;
    int numThreads = channels * height * width;
    dim3 block(512);
    dim3 grid((numThreads + block.x - 1) / block.x);
    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    // NOLINT_NEXT_LINE(whitespace/operators)
    col2im_gpu_kernel<<<grid, block>>>(numThreads, data_col, height, width,
                                       channels, kernel_h, kernel_w, pad,
                                       stride, out_height, out_width, data_im);
    MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}

void cuda_colwise_max(const dtype* input, int rows, int cols, dtype* out) {
    dim3 block(512);
    dim3 grid((cols + block.x - 1) / block.x);
    colwise_max<<<grid, block>>>(input, rows, cols, out);
    MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}

void cuda_sigmoid(int rows, int cols, const dtype* d_A, dtype* d_B) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    sigmoid<<<grid, block>>>(d_A, rows, cols, d_B);
    // cudaDeviceSynchronize();
    MY_CHECK(cudaPeekAtLastError());
}

void cuda_tanh(int rows, int cols, const dtype* d_A, dtype* d_B) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    tanh<<<grid, block>>>(d_A, rows, cols, d_B);
    // cudaDeviceSynchronize();
    MY_CHECK(cudaPeekAtLastError());
}

void next_lstm_cell(int rows, const dtype* d_A, dtype* d_B) {
    dim3 block(512);
    dim3 grid((rows + block.x - 1) / block.x);
    next_lstm_cell<<<grid, block>>>(d_A, rows, d_B);
    // MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}

void next_lstm_state(int rows, const dtype* d_A, const dtype* d_B, dtype* d_C) {
    dim3 block(512);
    dim3 grid((rows + block.x - 1) / block.x);
    next_lstm_state<<<grid, block>>>(d_A, d_B, rows, d_C);
    // MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}

void compute_deriv_cell(int rows, int cols, const dtype* cell,
                        dtype* deriv_cell) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    cuda_compute_deriv_cell<<<grid, block>>>(rows, cols, cell, deriv_cell);
    // MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}

void new_cell_state(int rows, const dtype* dcum_c, const dtype* dh,
                    const dtype* o, const dtype* sigma_c, dtype* dc) {
    dim3 block(512);
    dim3 grid((rows + block.x - 1) / block.x);
    cuda_new_cell_state<<<grid, block>>>(rows, dcum_c, dh, o, sigma_c, dc);
    // MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}

void internal_deriv(int rows, const dtype* dh, const dtype* dc,
                    const dtype* cell, const dtype* funcs, dtype* d_tmp) {
    dim3 block(512);
    dim3 grid((rows + block.x - 1) / block.x);
    cuda_internal_deriv<<<grid, block>>>(rows, dh, dc, cell, funcs, d_tmp);
    // MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}

void copy_data(int rows, int cols, const dtype* in, dtype* out) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    cuda_copy_data<<<grid, block>>>(rows, cols, in, out);
    // MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}

void sigmoid_deriv(int rows, int max_rows, int cols, const dtype* in,
                   dtype* out) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    cuda_sigmoid_deriv<<<grid, block>>>(rows, max_rows, cols, in, out);
    // MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}

void tanh_deriv(int rows, int max_rows, int cols, const dtype* in, dtype* out) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    cuda_tanh_deriv<<<grid, block>>>(rows, max_rows, cols, in, out);
    // MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}

void clip_gradients_gpu(int rows, int cols, dtype max, dtype* grad) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    cuda_clip_gradients<<<grid, block>>>(rows, cols, max, grad);
    // MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
}
