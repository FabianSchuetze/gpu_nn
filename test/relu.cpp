#include <eigen-git-mirror/Eigen/Core>
#include <memory>
#define CATCH_CONFIG_MAIN
#include <sys/time.h>
#include <iostream>
#include "../include/common.h"
#include "../include/layer/relu.h"
#include "../include/storage.h"
#include "../third_party/catch/catch.hpp"

using std::make_shared;
using std::shared_ptr;
using std::vector;

typedef std::shared_ptr<Storage> SharedStorage;
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
TEST_CASE("relu forward gpu", "[gpu]") {
    srand((unsigned int)time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* inp1;
    Relu s1(handle);
    inp1 = &s1;
    Matrix in = Matrix::Random(6, 5);
    Matrix out = Matrix::Zero(6, 5);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    inp1->forward_gpu(storage_in, storage_out);
    Vector sum = storage_out->return_data_const().colwise().sum();
    REQUIRE(storage_out->return_data_const().maxCoeff() ==
            Approx(storage_in->return_data_const().maxCoeff()));
    REQUIRE(storage_out->return_data_const().sum() >
            storage_in->return_data_const().sum());
}

TEST_CASE("relu backward equivlaence", "[backward equivalence]") {
    srand((unsigned int)time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* inp1;
    Relu s1(handle);
    inp1 = &s1;
    Matrix gradient_in = Matrix::Random(1024, 32);
    Matrix gradient_out_cpu = Matrix::Zero(1024, 32);
    Matrix gradient_out_gpu = Matrix::Zero(1024, 32);
    Matrix values = Matrix::Random(1024, 32);
    SharedStorage shared_grad_in = make_shared<Storage>(gradient_in);
    SharedStorage shared_grad_out_cpu = make_shared<Storage>(gradient_out_cpu);
    SharedStorage shared_grad_out_gpu = make_shared<Storage>(gradient_out_gpu);
    SharedStorage shared_values = make_shared<Storage>(values);
    double cpuStart = cpuSecond();
    inp1->backward_cpu(shared_values, shared_grad_in,
            shared_grad_out_cpu);
    double cpuEnd = cpuSecond() - cpuStart;
    double gpuStart = cpuSecond();
    inp1->backward_gpu(shared_values, shared_grad_in,
            shared_grad_out_gpu);
    double gpuEnd = cpuSecond() - gpuStart;
    Matrix  diff = shared_grad_out_cpu->return_data_const() -
        shared_grad_out_gpu->return_data_const();
    dtype max = diff.cwiseAbs().maxCoeff();
    std::cout << max << std::endl;
    std::cout << "The CPU took " << cpuEnd << " and hte GPU took " << gpuEnd
              << std::endl;
    REQUIRE(cpuEnd > gpuEnd);
    REQUIRE(max < 1e-6);
}

TEST_CASE("relu forward cpu", "[cpu]") {
    srand((unsigned int)time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* inp1;
    Relu s1(handle);
    inp1 = &s1;
    Matrix in = Matrix::Random(6, 5);
    Matrix out = Matrix::Zero(6, 5);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    inp1->forward_cpu(storage_in, storage_out);
    Vector sum = storage_out->return_data_const().colwise().sum();
    REQUIRE(storage_out->return_data_const().maxCoeff() ==
            Approx(storage_in->return_data_const().maxCoeff()));
    REQUIRE(storage_out->return_data_const().sum() >
            storage_in->return_data_const().sum());
}

TEST_CASE("relu forward equivalence", "[forward equivalence]") {
    srand((unsigned int)time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* inp1;
    Relu s1(handle);
    inp1 = &s1;
    Matrix in = Matrix::Random(1024, 32);
    Matrix out_cpu = Matrix::Zero(1024, 32);
    Matrix out_gpu = Matrix::Zero(1024, 32);
    shared_ptr<Storage> storage_in = make_shared<Storage>(in);
    shared_ptr<Storage> storage_out_cpu = make_shared<Storage>(out_cpu);
    shared_ptr<Storage> storage_out_gpu = make_shared<Storage>(out_gpu);
    double cpuStart = cpuSecond();
    inp1->forward_cpu(storage_in, storage_out_cpu);
    double cpuEnd = cpuSecond() - cpuStart;
    double gpuStart = cpuSecond();
    inp1->forward_gpu(storage_in, storage_out_gpu);
    double gpuEnd = cpuSecond() - gpuStart;
    Matrix diff = storage_out_cpu->return_data_const() -
                           storage_out_gpu->return_data_const();
    dtype max = diff.cwiseAbs().maxCoeff();
    std::cout << max << std::endl;
    std::cout << "The CPU took " << cpuEnd << " and hte GPU took " << gpuEnd
              << std::endl;
    REQUIRE(cpuEnd > gpuEnd);
    REQUIRE(max < 1e-6);
}
