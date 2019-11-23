#include "../include/storage.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include "../include/common.h"

Storage::Storage()
    : _data(), _cpu_pointer(NULL), _gpu_pointer(NULL), recent_head("UNINIT"){};

Storage::Storage(const Matrix& data)
    : _data(data),
      _cpu_pointer(_data.data()),
      _gpu_pointer(),
      recent_head("SYNC") {
    initialize_gpu_memory();
};

bool Storage::is_set() { return recent_head != "UNINIT"; }

Storage::~Storage() {
    // delete _cpu_pointer; // I don't know how to delete this pointer properly
    cudaFree(_gpu_pointer);
    delete _cpu_pointer;
}

void Storage::initialize_gpu_memory() {
    unsigned int nBytes = _data.rows() * _data.cols() * sizeof(dtype);
    MY_CHECK(cudaMalloc((void**)&_gpu_pointer, nBytes));
    MY_CHECK(
        cudaMemcpy(_gpu_pointer, _cpu_pointer, nBytes, cudaMemcpyHostToDevice));
    MY_CHECK(cudaDeviceSynchronize());
}

void Storage::update_cpu_data(Matrix new_data) {
    if (recent_head == "UNINIT") {
        std::string m("No data set yet, in:\n");
        throw std::invalid_argument(m + __PRETTY_FUNCTION__);
    }
    if (new_data.rows() != get_rows() or new_data.cols() != get_cols()) {
        std::string m("The new data size does not match the old, in:\n");
        throw std::invalid_argument(m + __PRETTY_FUNCTION__);
    }
    _data = new_data;
    _cpu_pointer = _data.data();
    recent_head = "CPU";
}

void Storage::update_gpu_data(const dtype new_data) {
    unsigned int nBytes = _data.rows() * _data.cols() * sizeof(dtype);
    MY_CHECK(cudaMemset(_gpu_pointer, new_data, nBytes));
    MY_CHECK(cudaDeviceSynchronize());
    recent_head = "GPU";
}

void Storage::update_gpu_data(const dtype* src) {
    unsigned int nBytes = _data.rows() * _data.cols() * sizeof(dtype);
    MY_CHECK(cudaMemcpy(_gpu_pointer, src, nBytes, cudaMemcpyDeviceToDevice));
    MY_CHECK(cudaDeviceSynchronize());
    recent_head = "GPU";
}

void Storage::sync_to_cpu() {
    if (recent_head == "GPU") {
        // std::cout << "copying to CPU\n";
        unsigned int nBytes = _data.rows() * _data.cols() * sizeof(dtype);
        MY_CHECK(cudaMemcpy(_cpu_pointer, _gpu_pointer, nBytes,
                            cudaMemcpyDeviceToHost));
        MY_CHECK(cudaDeviceSynchronize());
        recent_head = "SYNC";
    }
}

void Storage::sync_to_gpu() {
    if (recent_head == "CPU") {
        // std::cout << "syncing to GPU\n";
        unsigned int nBytes = _data.rows() * _data.cols() * sizeof(dtype);
        MY_CHECK(cudaMemcpy(_gpu_pointer, _data.data(), nBytes,
                            cudaMemcpyHostToDevice));
        MY_CHECK(cudaDeviceSynchronize());
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

bool same_size(const std::shared_ptr<Storage>& a,
               const std::shared_ptr<Storage>& b) {
    return ((a->get_rows() == b->get_rows()) and
            (a->get_cols() == b->get_cols()));
}
