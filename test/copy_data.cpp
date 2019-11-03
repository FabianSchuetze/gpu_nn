#include "../include/storage.h"
#include "../include/math.h"
#include <eigen-git-mirror/Eigen/src/Core/util/ForwardDeclarations.h>
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include "../include/common.h"
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
int main() {
    //int dev = 0;
    //cudaDeviceProp deviceProp;
    //CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    //printf("Using Device %d: %s\n", dev, deviceProp.name);
    //CHECK(cudaSetDevice(dev));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(1000,1000);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(1000,1000);
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(1000,1000);
    Eigen::MatrixXd D = Eigen::MatrixXd::Random(1000,1000);
    Eigen::MatrixXd E = Eigen::MatrixXd::Random(1000,1000);
    std::shared_ptr<Storage> storageA = std::make_shared<Storage>(A);
    std::shared_ptr<Storage> storageB = std::make_shared<Storage>(B);
    std::shared_ptr<Storage> storageC = std::make_shared<Storage>(C);
    std::shared_ptr<Storage> storageD = std::make_shared<Storage>(D);
    std::shared_ptr<Storage> storageE = std::make_shared<Storage>(E);
    //multonGPU(handle, storageD, storageE, storageC, 1, 1);
    double iStart = cpuSecond();
    //multonGPU(handle, storageA, storageB, storageC, 1, 1);
    //multonGPU(handle, storageD, storageE, storageC, 1, 1);
    double Elaps = cpuSecond() - iStart;
    std::cout << Elaps << std::endl;
    iStart = cpuSecond();
    C += A * B;
    C += D * E;
    Elaps = cpuSecond() - iStart;
    std::cout << Elaps << std::endl;

}
