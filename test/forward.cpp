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
    NeuralNetwork n1(vec, "Categorical_Crossentropy");
    Matrix in = Matrix::Random(obs, input_dimension);
    std::cout << "the input matrix is\n" << in << std::endl;
    Matrix out = n1.predict(in);
    std::cout << out << std::endl;
}
