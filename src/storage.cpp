#include "../include/storage.h"
#include <cuda_runtime.h>
#include "../include/common.h"
#include <iostream>

Storage::Storage(const Eigen::MatrixXd& data)
    : _data(data),
      _cpu_pointer(_data.data()),
      _gpu_pointer(),
      recent_head("SYNC") {
    initialize_gpu_memory();
};

Storage::~Storage() {
    //delete _cpu_pointer; // I don't know how to delete this pointer properly
    cudaFree(_gpu_pointer);
}

void Storage::initialize_gpu_memory() {
    unsigned int nBytes = _data.rows() * _data.cols() * sizeof(double);
    CHECK(cudaMalloc((void**)&_gpu_pointer, nBytes));
    CHECK(
        cudaMemcpy(_gpu_pointer, _cpu_pointer, nBytes, cudaMemcpyHostToDevice));
}

void Storage::sync_to_cpu() {
    if (recent_head == "GPU") {
        std::cout << "copying to CPU\n"; 
        unsigned int nBytes = _data.rows() * _data.cols() * sizeof(double);
        CHECK(cudaMemcpy(_cpu_pointer, _gpu_pointer, nBytes,
                         cudaMemcpyDeviceToHost));
        recent_head = "SYNC";
    }
}

void Storage::sync_to_gpu() {
    if (recent_head == "CPU") {
        unsigned int nBytes = _data.rows() * _data.cols() * sizeof(double);
        CHECK(cudaMemcpy(_gpu_pointer, _data.data(), nBytes,
                         cudaMemcpyHostToDevice));
        recent_head = "SYNC";
    }
}

const double* Storage::cpu_pointer_const() {
    sync_to_cpu();
    const double* cp = const_cast<const double*>(_cpu_pointer);
    return cp;
}
const double* Storage::gpu_pointer_const() {
    sync_to_gpu();
    const double* cp = const_cast<const double*>(_gpu_pointer);
    return cp;
    //return (const double*)_cpu_pointer;
}

double* Storage::cpu_pointer() {
    sync_to_cpu();
    recent_head = "CPU";
    return _cpu_pointer;
}

double* Storage::gpu_pointer() {
    sync_to_gpu();
    recent_head = "GPU";
    return _gpu_pointer;
}

const Eigen::MatrixXd Storage::return_data_const() {
    sync_to_cpu();
    return _data;
}

Eigen::MatrixXd Storage::return_data() {
    sync_to_cpu();
    recent_head = "CPU";
    return _data;
}
