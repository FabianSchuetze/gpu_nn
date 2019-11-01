#include "../include/storage.h"
#include <cuda_runtime.h>
#include "../include/common.h"

Storage::Storage(const Eigen::MatrixXd& data)
    : _data(data), _cpu_pointer(_data.data()), _gpu_pointer() {
    initialize_gpu_memory();
};

void Storage::initialize_gpu_memory() {
    unsigned int nBytes = _data.rows() * _data.cols() * sizeof(double);
    CHECK(cudaMalloc((void**)&_gpu_pointer, nBytes));
    CHECK(
        cudaMemcpy(_gpu_pointer, _data.data(), nBytes, cudaMemcpyHostToDevice));
}

void Storage::copy_to_cpu() {
    unsigned int nBytes = _data.rows() * _data.cols() * sizeof(double);
    CHECK(cudaMemcpy(_cpu_pointer, _gpu_pointer, nBytes, cudaMemcpyDeviceToHost));
}

