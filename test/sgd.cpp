#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <sys/time.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <iostream>
#include <memory>
#include <random>
#include "../include/neural_network.h"
#include "../third_party/catch/catch.hpp"

using std::vector;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

TEST_CASE("NeuralNetwork backward cpu", "[backward cpu]") {
    srand((unsigned int)time(0));
    Layer* l1;
    Layer* l2;
    Layer* l3;
    Layer* l4;
    Layer* l5;
    int input_dimension = 5;
    int obs = 3;
    Input i1(input_dimension);
    Dense d1(10, input_dimension);
    Relu relu1;
    Dense d2(2, 10);
    Matrix target = Matrix::Zero(obs, 2);
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < obs; i++) {
        double abc = distribution(generator);
        if (abc > 0.5) {
            target(i, 0) = 0;
            target(i, 1) = 1;
        } else {
            target(i, 0) = 1;
            target(i, 1) = 0;
        }
    }
    SharedStorage SharedTarget = std::make_shared<Storage>(target.transpose());
    Softmax s1;
    l1 = &i1;
    l2 = &d1;
    l3 = &relu1;
    l4 = &d2;
    l5 = &s1;
    std::vector<Layer*> vec = {l1, l2, l3, l4, l5};
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    std::shared_ptr<GradientDescent> sgd =
        std::make_shared<StochasticGradientDescent>(
            StochasticGradientDescent(0.1));
    NeuralNetwork n1(vec, loss, "CPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    // the forward part
    SharedStorage inp = std::make_shared<Storage>(in);
    vector<SharedStorage> vals = n1.allocate_forward(in.rows());
    vector<SharedStorage> gradients = n1.allocate_backward(in.rows());
    n1.fill_hiddens(vals, in.transpose());
    n1.forward(vals);
    SharedStorage& grad_in = gradients[gradients.size() - 1];
    loss->grad_loss_cpu(grad_in, vals[vals.size() - 1], SharedTarget,
                        SharedTarget);
    n1.backwards(gradients, vals);
    Matrix before_weight = l2->return_parameters()[0]->return_data_const();
    Matrix before_bias = l2->return_parameters()[1]->return_data_const();
    n1.update_weights(sgd, obs);
    Matrix after_weight = l2->return_parameters()[0]->return_data_const();
    Matrix after_bias = l2->return_parameters()[1]->return_data_const();
    Matrix WeightDiff = before_weight - after_weight;
    Matrix BiasDiff = before_bias - after_bias;
    dtype bias_diff = BiasDiff.cwiseAbs().sum();
    dtype weight_diff = WeightDiff.cwiseAbs().sum();
    REQUIRE(bias_diff > 0);
    REQUIRE(weight_diff > 0);
    REQUIRE(l2->return_gradients()[0]->return_data_const()(0, 0) == 0);
    REQUIRE(l2->return_gradients()[1]->return_data_const()(0, 0) == 0);
}

TEST_CASE("NeuralNetwork backward gpu", "[backward gpu]") {
    srand((unsigned int)time(0));
    Layer* l1;
    Layer* l2;
    Layer* l3;
    Layer* l4;
    Layer* l5;
    int input_dimension = 5;
    int obs = 3;
    Input i1(input_dimension);
    Dense d1(10, input_dimension);
    Relu relu1;
    Dense d2(2, 10);
    Matrix target = Matrix::Zero(obs, 2);
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < obs; i++) {
        double abc = distribution(generator);
        if (abc > 0.5) {
            target(i, 0) = 0;
            target(i, 1) = 1;
        } else {
            target(i, 0) = 1;
            target(i, 1) = 0;
        }
    }
    SharedStorage SharedTarget = std::make_shared<Storage>(target.transpose());
    Softmax s1;
    l1 = &i1;
    l2 = &d1;
    l3 = &relu1;
    l4 = &d2;
    l5 = &s1;
    std::vector<Layer*> vec = {l1, l2, l3, l4, l5};
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    std::shared_ptr<GradientDescent> sgd =
        std::make_shared<StochasticGradientDescent>(
            StochasticGradientDescent(0.1));
    NeuralNetwork n1(vec, loss, "GPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    // the forward part
    SharedStorage inp = std::make_shared<Storage>(in);
    vector<SharedStorage> vals = n1.allocate_forward(in.rows());
    vector<SharedStorage> gradients = n1.allocate_backward(in.rows());
    n1.fill_hiddens(vals, in.transpose());
    n1.forward(vals);
    SharedStorage& grad_in = gradients[gradients.size() - 1];
    loss->grad_loss_gpu(grad_in, vals[vals.size() - 1], SharedTarget,
                        SharedTarget);
    n1.backwards(gradients, vals);
    Matrix before_weight = l2->return_parameters()[0]->return_data_const();
    Matrix before_bias = l2->return_parameters()[1]->return_data_const();
    n1.update_weights(sgd, obs);
    Matrix after_weight = l2->return_parameters()[0]->return_data_const();
    Matrix after_bias = l2->return_parameters()[1]->return_data_const();
    Matrix WeightDiff = before_weight - after_weight;
    Matrix BiasDiff = before_bias - after_bias;
    dtype bias_diff = BiasDiff.cwiseAbs().sum();
    dtype weight_diff = WeightDiff.cwiseAbs().sum();
    REQUIRE(bias_diff > 0);
    REQUIRE(weight_diff > 0);
    REQUIRE(l2->return_gradients()[0]->return_data_const()(0, 0) == 0);
    REQUIRE(l2->return_gradients()[1]->return_data_const()(0, 0) == 0);
}

TEST_CASE("NeuralNetwork equivalance", "[equivalance]") {
    srand((unsigned int)time(0));
    Layer* l1;
    Layer* l1_gpu;
    Layer* l2;
    Layer* l2_gpu;
    Layer* l3;
    Layer* l3_gpu;
    Layer* l4;
    Layer* l4_gpu;
    Layer* l5;
    Layer* l5_gpu;
    int input_dimension = 1024;
    int obs = 32;
    Input i1(input_dimension);
    Dense d1(900, input_dimension);
    Relu relu1;
    Dense d2(2, 900);
    Matrix target = Matrix::Zero(obs, 2);
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < obs; i++) {
        double abc = distribution(generator);
        if (abc > 0.5) {
            target(i, 0) = 0;
            target(i, 1) = 1;
        } else {
            target(i, 0) = 1;
            target(i, 1) = 0;
        }
    }
    SharedStorage SharedTarget = std::make_shared<Storage>(target.transpose());
    Softmax s1;
    l1 = &i1;
    l2 = &d1;
    l3 = &relu1;
    l4 = &d2;
    l5 = &s1;
    std::vector<Layer*> vec = {l1, l2, l3, l4, l5};
    l1_gpu = &i1;
    l2_gpu = &d1;
    l3_gpu = &relu1;
    l4_gpu = &d2;
    l5_gpu = &s1;
    std::vector<Layer*> vec_gpu = {l1_gpu, l2_gpu, l3_gpu, l4_gpu, l5_gpu};
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    std::shared_ptr<GradientDescent> sgd =
        std::make_shared<StochasticGradientDescent>(
            StochasticGradientDescent(0.01));
    NeuralNetwork n_cpu(vec, loss, "CPU");
    NeuralNetwork n_gpu(vec_gpu, loss, "GPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    SharedStorage inp = std::make_shared<Storage>(in);
     //CPU CODE
    double cpuStart = cpuSecond();
    vector<SharedStorage> vals_cpu = n_cpu.allocate_forward(in.rows());
    vector<SharedStorage> grad_cpu = n_cpu.allocate_backward(in.rows());
    n_cpu.fill_hiddens(vals_cpu, in.transpose());
    n_cpu.forward(vals_cpu);
    SharedStorage& grad_in = grad_cpu[grad_cpu.size() - 1];
    loss->grad_loss_cpu(grad_in, vals_cpu[vals_cpu.size() - 1], SharedTarget,
                        SharedTarget);
    n_cpu.backwards(grad_cpu, vals_cpu);
    n_cpu.update_weights(sgd, obs);
    Matrix after_weight = l2->return_parameters()[0]->return_data_const();
    double cpuEnd = cpuSecond() - cpuStart;
     //GPU PART
    double gpuStart = cpuSecond();
    vector<SharedStorage> vals_gpu = n_gpu.allocate_forward(in.rows());
    vector<SharedStorage> grad_gpu = n_gpu.allocate_backward(in.rows());
    n_gpu.fill_hiddens(vals_gpu, in.transpose());
    n_gpu.forward(vals_gpu);
    SharedStorage& grad_in_gpu = grad_gpu[grad_gpu.size() - 1];
    loss->grad_loss_gpu(grad_in_gpu, vals_gpu[vals_gpu.size() - 1],
                        SharedTarget, SharedTarget);
    n_gpu.backwards(grad_gpu, vals_gpu);
    n_gpu.update_weights(sgd, obs);
    double gpuEnd = cpuSecond() - gpuStart;
    Matrix diff = l2->return_parameters()[0]->return_data_const() -
                  l2_gpu->return_parameters()[0]->return_data_const();
    std::cout << "The CPU took " << cpuEnd << " and hte GPU took " << gpuEnd
              << std::endl;
    dtype maxDiff = diff.array().abs().maxCoeff();
    REQUIRE(cpuEnd > gpuEnd);
    REQUIRE(maxDiff < 1e-6);
}
