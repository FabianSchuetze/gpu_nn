#include "../include/storage.h"
#include "../include/math.h"
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include "../include/common.h"

int main() {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(5,5);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(5,5);
    Eigen::MatrixXd C = Eigen::MatrixXd::Random(5,5);
    std::shared_ptr<Storage> storageA = std::make_shared<Storage>(A);
    std::shared_ptr<Storage> storageB = std::make_shared<Storage>(B);
    std::shared_ptr<Storage> storageC = std::make_shared<Storage>(C);
    std::cout << "beore copying\n";
    std::cout << storageC->return_data() << std::endl;
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    multonGPU(handle, storageA, storageB, storageC);
    std::cout << "beore copying\n";
    std::cout << storageC->return_data() << std::endl;
    storageC->copy_to_cpu();
    std::cout << "after copying\n";
    std::cout << storageC->return_data() << std::endl;
}
