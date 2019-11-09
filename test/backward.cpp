#define CATCH_CONFIG_MAIN
#include "../third_party/catch/catch.hpp"
#include <cuda_runtime.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <sys/time.h>
#include <iostream>
#include <memory>
#include "../include/neural_network.h"

using std::vector;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

TEST_CASE("NeuralNetwork backward gpu", "[backward gpu]") {
    srand((unsigned int) time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* l1;
    Layer* l2;
    Layer* l3;
    Layer* l4;
    Layer* l5;
    int input_dimension = 5;
    int obs = 3;
    Input i1(input_dimension);
    Dense d1(10, input_dimension, handle);
    Relu relu1(handle);
    Dense d2(2, 10, handle);
    Matrix target = Matrix::Zero(obs, 2);
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for (int i = 0; i < obs; i++) {
        double abc = distribution(generator);
        if (abc > 0.5) {
            target(i, 0) = 0;
            target(i, 1) = 1;
        }
        else {
            target(i, 0) = 1;
            target(i, 1) = 0;
        }
    }
    SharedStorage SharedTarget = std::make_shared<Storage>(target.transpose());
    Softmax  s1(handle);
    l1 = &i1;
    l2 = &d1;
    l3 = &relu1;
    l4 = &d2;
    l5 = &s1;
    std::vector<Layer*> vec = {l1, l2, l3, l4, l5};
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    NeuralNetwork n1(vec, loss, "GPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    //the forward part
    SharedStorage inp = std::make_shared<Storage>(in);
    vector<SharedStorage> vals = n1.allocate_forward(in.rows());
    vector<SharedStorage> gradients = n1.allocate_backward(in.rows());
    n1.fill_hiddens(vals, in);
    n1.forward(vals);
    SharedStorage& grad_in = gradients[gradients.size() -1];
    loss->grad_loss_gpu(grad_in, vals[vals.size() -1], SharedTarget, SharedTarget);
    n1.backwards(gradients, vals);
}

TEST_CASE("NeuralNetwork backward cpu", "[backward cpu]") {
    srand((unsigned int) time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* l1;
    Layer* l2;
    Layer* l3;
    Layer* l4;
    Layer* l5;
    int input_dimension = 5;
    int obs = 3;
    Input i1(input_dimension);
    Dense d1(10, input_dimension, handle);
    Relu relu1(handle);
    Dense d2(2, 10, handle);
    Matrix target = Matrix::Zero(obs, 2);
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for (int i = 0; i < obs; i++) {
        double abc = distribution(generator);
        if (abc > 0.5) {
            target(i, 0) = 0;
            target(i, 1) = 1;
        }
        else {
            target(i, 0) = 1;
            target(i, 1) = 0;
        }
    }
    SharedStorage SharedTarget = std::make_shared<Storage>(target.transpose());
    Softmax  s1(handle);
    l1 = &i1;
    l2 = &d1;
    l3 = &relu1;
    l4 = &d2;
    l5 = &s1;
    std::vector<Layer*> vec = {l1, l2, l3, l4, l5};
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    NeuralNetwork n1(vec, loss, "CPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    //the forward part
    SharedStorage inp = std::make_shared<Storage>(in);
    vector<SharedStorage> vals = n1.allocate_forward(in.rows());
    vector<SharedStorage> gradients = n1.allocate_backward(in.rows());
    n1.fill_hiddens(vals, in);
    n1.forward(vals);
    SharedStorage& grad_in = gradients[gradients.size() -1];
    loss->grad_loss_cpu(grad_in, vals[vals.size() -1], SharedTarget, SharedTarget);
    n1.backwards(gradients, vals);
    //std::cout << l2->return_gradients()[0]->return_data_const() << std::endl;
}

TEST_CASE("NeuralNetwork backward equivalence", "[backward equivalence]") {
    srand((unsigned int) time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* l1;
    Layer* l2;
    Layer* l3;
    Layer* l4;
    Layer* l5;
    int input_dimension = 1024;
    int obs = 32;
    Input i1(input_dimension);
    Dense d1(900, input_dimension, handle);
    Relu relu1(handle);
    Dense d2(2, 900, handle);
    Matrix target = Matrix::Zero(obs, 2);
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for (int i = 0; i < obs; i++) {
        double abc = distribution(generator);
        if (abc > 0.5) {
            target(i, 0) = 0;
            target(i, 1) = 1;
        }
        else {
            target(i, 0) = 1;
            target(i, 1) = 0;
        }
    }
    SharedStorage SharedTarget = std::make_shared<Storage>(target.transpose());
    Softmax  s1(handle);
    l1 = &i1;
    l2 = &d1;
    l3 = &relu1;
    l4 = &d2;
    l5 = &s1;
    std::vector<Layer*> vec = {l1, l2, l3, l4, l5};
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    NeuralNetwork n_cpu(vec, loss, "CPU");
    NeuralNetwork n_gpu(vec, loss, "GPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    //the forward part
    SharedStorage inp = std::make_shared<Storage>(in);
    vector<SharedStorage> vals = n_cpu.allocate_forward(in.rows());
    vector<SharedStorage> gradients_cpu = n_cpu.allocate_backward(in.rows());
    vector<SharedStorage> gradients_gpu = n_cpu.allocate_backward(in.rows());
    n_cpu.fill_hiddens(vals, in);
    // CPU CODE
    n_cpu.forward(vals);
    SharedStorage& grad_in_cpu = gradients_cpu[gradients_cpu.size() -1];
    loss->grad_loss_cpu(grad_in_cpu, vals[vals.size() -1], SharedTarget, SharedTarget);
    n_cpu.backwards(gradients_cpu, vals);
    // GPU CODE
    n_gpu.forward(vals);
    SharedStorage& grad_in = gradients_gpu[gradients_gpu.size() -1];
    loss->grad_loss_gpu(grad_in, vals[vals.size() -1], SharedTarget, SharedTarget);
    n_cpu.backwards(gradients_gpu, vals);
    // Compare difference
    Matrix diff = gradients_gpu[0]->return_data_const() -
        gradients_cpu[0]->return_data_const();
    dtype maxDiff = diff.array().abs().maxCoeff();
    std::cout << "the max diff is:" << maxDiff << std::endl;

    //std::cout << l2->return_gradients()[0]->return_data_const() << std::endl;
}
