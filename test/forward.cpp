#define CATCH_CONFIG_MAIN
#include "../third_party/catch/catch.hpp"
#include <cuda_runtime.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <sys/time.h>
#include <iostream>
#include <memory>
#include "../include/neural_network.h"

typedef std::shared_ptr<Layer> s_Layer;
using std::make_shared;
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

TEST_CASE("NeuralNetwork forward gpu", "[forward gpu]") {
    //int main() {
    srand((unsigned int) time(0));
    int input_dimension = 5;
    int obs = 3;
    Init* init = new Glorot();
    s_Layer l1 =  make_shared<Input>(Features(input_dimension));
    s_Layer l2 = make_shared<Dense>(Features(6), l1, init);
    s_Layer l3 = make_shared<Softmax>(l2);
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    NeuralNetwork n1(l3, loss, "GPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    Matrix out = n1.predict(in);
    Vector sums = out.rowwise().sum();
    REQUIRE(sums(0) == Approx(1.));
}

TEST_CASE("NeuralNetwork2 forward cpu", "[forward2 cpu]") {
    srand((unsigned int) time(0));
    int input_dimension = 5;
    int obs = 3;
    Init* init = new Glorot();
    s_Layer l1 =  make_shared<Input>(Features(input_dimension));
    s_Layer l2 = make_shared<Dense>(Features(6), l1, init);
    s_Layer l3 = make_shared<Softmax>(l2);
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    NeuralNetwork n1(l3, loss, "CPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    Matrix out = n1.predict(in);
    Vector sums = out.rowwise().sum();
    REQUIRE(sums(0) == Approx(1.));
}

TEST_CASE("NeuralNetwork equivalence", "[equivalence]") {
    srand((unsigned int) time(0));
    int input_dimension = 1024;
    int obs = 32;
    Init* init = new Glorot();
    s_Layer l1 =  make_shared<Input>(Features(input_dimension));
    s_Layer l2 = make_shared<Dense>(Features(6), l1, init);
    s_Layer l3 = make_shared<Softmax>(l2);
    s_Layer l1_gpu =  make_shared<Input>(Features(input_dimension));
    s_Layer l2_gpu = make_shared<Dense>(Features(6), l1_gpu, init);
    s_Layer l3_gpu = make_shared<Softmax>(l2_gpu);
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    NeuralNetwork n_cpu(l3, loss, "CPU");
    NeuralNetwork n_gpu(l3_gpu, loss, "GPU");
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
