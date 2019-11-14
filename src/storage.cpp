#include "../include/storage.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include "../include/common.h"

Storage::Storage()
    : _data(), _cpu_pointer(NULL), _gpu_pointer(NULL), recent_head("SYNC"){};

Storage::Storage(const Matrix& data)
    : _data(data),
      _cpu_pointer(_data.data()),
      _gpu_pointer(),
      recent_head("SYNC") {
    initialize_gpu_memory();
};

Storage::~Storage() {
    // delete _cpu_pointer; // I don't know how to delete this pointer properly
    cudaFree(_gpu_pointer);
}

void Storage::initialize_gpu_memory() {
    unsigned int nBytes = _data.rows() * _data.cols() * sizeof(dtype);
    MY_CHECK(cudaMalloc((void**)&_gpu_pointer, nBytes));
    MY_CHECK(
        cudaMemcpy(_gpu_pointer, _cpu_pointer, nBytes, cudaMemcpyHostToDevice));
}

void Storage::update_cpu_data(Matrix new_data) {
    std::cout << "inside the update data\n";
    if (_cpu_pointer == NULL) {
        std::string m("The new data size does not match the old, in:\n");
        throw std::invalid_argument(m + __PRETTY_FUNCTION__);
    }
    _data = new_data;
    _cpu_pointer = _data.data();
    recent_head = "CPU";
}

void Storage::update_gpu_data(dtype new_data) {
    unsigned int nBytes = _data.rows() * _data.cols() * sizeof(dtype);
    MY_CHECK(cudaMemset(_gpu_pointer, new_data, nBytes));
    recent_head = "GPU";
}

void Storage::sync_to_cpu() {
    if (recent_head == "GPU") {
        std::cout << "copying to CPU\n";
        unsigned int nBytes = _data.rows() * _data.cols() * sizeof(dtype);
        MY_CHECK(cudaMemcpy(_cpu_pointer, _gpu_pointer, nBytes,
                            cudaMemcpyDeviceToHost));
        recent_head = "SYNC";
    }
}

void Storage::sync_to_gpu() {
    if (recent_head == "CPU") {
        std::cout << "syncing to GPU\n";
        unsigned int nBytes = _data.rows() * _data.cols() * sizeof(dtype);
        MY_CHECK(cudaMemcpy(_gpu_pointer, _data.data(), nBytes,
                            cudaMemcpyHostToDevice));
        recent_head = "SYNC";
    }
}

const dtype* Storage::cpu_pointer_const() {
    sync_to_cpu();
    const dtype* cp = const_cast<const dtype*>(_cpu_pointer);
    return cp;
}

const dtype* Storage::gpu_pointer_const() {
    sync_to_gpu();
    const dtype* cp = const_cast<const dtype*>(_gpu_pointer);
    return cp;
}

dtype* Storage::cpu_pointer() {
    sync_to_cpu();
    recent_head = "CPU";
    return _cpu_pointer;
}

dtype* Storage::gpu_pointer() {
    sync_to_gpu();
    recent_head = "GPU";
    return _gpu_pointer;
}

const Matrix& Storage::return_data_const() {
    sync_to_cpu();
    return _data;
}

Matrix& Storage::return_data() {
    sync_to_cpu();
    recent_head = "CPU";
    return _data;
}
