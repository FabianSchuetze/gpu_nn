#define CATCH_CONFIG_MAIN
#include "../third_party/catch/catch.hpp"
#include <cuda_runtime.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <sys/time.h>
#include <iostream>
#include <memory>
#include "../include/neural_network.h"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

TEST_CASE("NeuralNetwork forward gpu", "[forward gpu]") {
    srand((unsigned int) time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* l1;
    Layer* l2;
    Layer* l3;
    int input_dimension = 5;
    int obs = 3;
    Input i1(input_dimension);
    Dense d1(6, input_dimension, handle);
    Softmax  s1(handle);
    l1 = &i1;
    l2 = &d1;
    l3 = &s1;
    std::vector<Layer*> vec = {l1, l2, l3};
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    NeuralNetwork n1(vec, loss, "GPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    Matrix out = n1.predict(in);
    Vector sums = out.rowwise().sum();
    REQUIRE(sums(0) == Approx(1.));
}

TEST_CASE("NeuralNetwork2 forward cpu", "[forward2 cpu]") {
    srand((unsigned int) time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* l1;
    Layer* l2;
    Layer* l3;
    int input_dimension = 5;
    int obs = 3;
    Input i1(input_dimension);
    Dense d1(6, input_dimension, handle);
    Softmax  s1(handle);
    l1 = &i1;
    l2 = &d1;
    l3 = &s1;
    std::vector<Layer*> vec = {l1, l2, l3};
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    NeuralNetwork n1(vec, loss, "CPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    Matrix out = n1.predict(in);
    Vector sums = out.rowwise().sum();
    REQUIRE(sums(0) == Approx(1.));
}

TEST_CASE("NeuralNetwork equivalence", "[equivalence]") {
    srand((unsigned int) time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* l1;
    Layer* l2;
    Layer* l3;
    int input_dimension = 1024;
    int obs = 32;
    Input i1(input_dimension);
    Dense d1(100, input_dimension, handle);
    Softmax  s1(handle);
    l1 = &i1;
    l2 = &d1;
    l3 = &s1;
    std::vector<Layer*> vec = {l1, l2, l3};
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    NeuralNetwork n_cpu(vec, loss, "CPU");
    NeuralNetwork n_gpu(vec, loss, "GPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    double cpuStart = cpuSecond();
    Matrix out_cpu = n_cpu.predict(in);
    double cpuEnd = cpuSecond() - cpuStart;
    double gpuStart = cpuSecond();
    Matrix out_gpu = n_gpu.predict(in);
    double gpuEnd = cpuSecond() - gpuStart;
    std::cout << "The CPU took " << cpuEnd << " and hte GPU took " <<
        gpuEnd << std::endl;
    Matrix diff = out_cpu - out_gpu;
    dtype maxDiff = diff.array().abs().maxCoeff();
    REQUIRE(cpuEnd > gpuEnd);
    REQUIRE(maxDiff < 1e-6);
}
