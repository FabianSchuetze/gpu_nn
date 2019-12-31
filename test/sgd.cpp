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
typedef std::shared_ptr<Layer> s_Layer;
using std::make_shared;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

std::vector<std::vector<SharedStorage>> prepare_helpers(
    std::initializer_list<s_Layer> layers) {
    std::vector<std::vector<SharedStorage>> helpers;
    for (s_Layer layer : layers) {
        std::vector<SharedStorage> helper;
        for (SharedStorage store : layer->return_gradients()) {
            int rows = store->get_rows();
            int cols = store->get_cols();
            Matrix tmp = Matrix::Zero(rows, cols);
            helper.push_back(std::make_shared<Storage>(tmp));
        }
        helpers.push_back(helper);
    }
    return helpers;
}

TEST_CASE("NeuralNetwork backward cpu", "[backward cpu]") {
    srand((unsigned int)time(0));
    int input_dimension = 5;
    int obs = 3;
    Init* init = new Glorot();
    s_Layer l1 = make_shared<Input>(Features(input_dimension));
    s_Layer l2 = make_shared<Dense>(Features(10), l1, init);
    s_Layer l3 = make_shared<Relu>(l2);
    s_Layer l4 = make_shared<Dense>(Features(2), l3, init);
    s_Layer l5 = make_shared<Softmax>(l4);
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
    DebugInfo no_debugging = DebugInfo("", "");
    SharedStorage SharedTarget = std::make_shared<Storage>(target.transpose());
    // std::vector<Layer*> vec = {l1, l2, l3, l4, l5};
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    std::shared_ptr<GradientDescent> sgd =
        std::make_shared<StochasticGradientDescent>(
            StochasticGradientDescent(LearningRate(0.1)));
    NeuralNetwork n1(l5, loss, "CPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    // the forward part
    SharedStorage inp = std::make_shared<Storage>(in);
    vector<SharedStorage> vals = n1.allocate_forward(in.rows());
    vector<SharedStorage> gradients = n1.allocate_backward(in.rows());
    n1.fill_hiddens(vals, in.transpose());
    n1.forward(vals, "train", no_debugging);
    SharedStorage& grad_in = gradients[gradients.size() - 1];
    loss->grad_loss_cpu(grad_in, vals[vals.size() - 1], SharedTarget,
                        SharedTarget);
    n1.backwards(gradients, vals, no_debugging);
    std::vector<std::vector<SharedStorage>> helpers = prepare_helpers({l2, l4});
    Matrix before_weight = l2->return_parameters()[0]->return_data_const();
    Matrix before_bias = l2->return_parameters()[1]->return_data_const();
    n1.update_weights(sgd, helpers, obs);
    Matrix after_weight = l2->return_parameters()[0]->return_data_const();
    Matrix after_bias = l2->return_parameters()[1]->return_data_const();
    Matrix WeightDiff = before_weight - after_weight;
    Matrix BiasDiff = before_bias - after_bias;
    dtype bias_diff = BiasDiff.cwiseAbs().sum();
    dtype weight_diff = WeightDiff.cwiseAbs().sum();
    REQUIRE(bias_diff > 0);
    REQUIRE(weight_diff > 0);
}

TEST_CASE("NeuralNetwork backward gpu", "[backward gpu]") {
    srand((unsigned int)time(0));
    int input_dimension = 5;
    int obs = 3;
    Init* init = new Glorot();
    s_Layer l1 = make_shared<Input>(Features(input_dimension));
    s_Layer l2 = make_shared<Dense>(Features(10), l1, init);
    s_Layer l3 = make_shared<Relu>(l2);
    s_Layer l4 = make_shared<Dense>(Features(2), l3, init);
    s_Layer l5 = make_shared<Softmax>(l4);
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
    DebugInfo no_debugging = DebugInfo("", "");
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    std::shared_ptr<GradientDescent> sgd =
        std::make_shared<StochasticGradientDescent>(
            StochasticGradientDescent(LearningRate(0.1)));
    NeuralNetwork n1(l5, loss, "GPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    // the forward part
    SharedStorage inp = std::make_shared<Storage>(in);
    vector<SharedStorage> vals = n1.allocate_forward(in.rows());
    vector<SharedStorage> gradients = n1.allocate_backward(in.rows());
    n1.fill_hiddens(vals, in.transpose());
    n1.forward(vals, "train", no_debugging);
    SharedStorage& grad_in = gradients[gradients.size() - 1];
    loss->grad_loss_gpu(grad_in, vals[vals.size() - 1], SharedTarget,
                        SharedTarget);
    n1.backwards(gradients, vals, no_debugging);
    Matrix before_weight = l2->return_parameters()[0]->return_data_const();
    Matrix before_bias = l2->return_parameters()[1]->return_data_const();
    std::vector<std::vector<SharedStorage>> helpers = prepare_helpers({l2, l4});
    n1.update_weights(sgd, helpers, obs);
    Matrix after_weight = l2->return_parameters()[0]->return_data_const();
    Matrix after_bias = l2->return_parameters()[1]->return_data_const();
    Matrix WeightDiff = before_weight - after_weight;
    Matrix BiasDiff = before_bias - after_bias;
    dtype bias_diff = BiasDiff.cwiseAbs().sum();
    dtype weight_diff = WeightDiff.cwiseAbs().sum();
    REQUIRE(bias_diff > 0);
    REQUIRE(weight_diff > 0);
}

TEST_CASE("NeuralNetwork equivalance", "[equivalance]") {
    srand((unsigned int)time(0));
    int input_dimension = 1024;
    Init* init = new Glorot();
    s_Layer l1 = make_shared<Input>(Features(input_dimension));
    s_Layer l2 = make_shared<Dense>(Features(10), l1, init);
    s_Layer l3 = make_shared<Relu>(l2);
    s_Layer l4 = make_shared<Dense>(Features(2), l3, init);
    s_Layer l5 = make_shared<Softmax>(l4);
    s_Layer l1_gpu = make_shared<Input>(Features(input_dimension));
    s_Layer l2_gpu = make_shared<Dense>(Features(10), l1_gpu, init);
    s_Layer l3_gpu = make_shared<Relu>(l2_gpu);
    s_Layer l4_gpu = make_shared<Dense>(Features(2), l3_gpu, init);
    s_Layer l5_gpu = make_shared<Softmax>(l4_gpu);
    int obs = 32;
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
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    std::shared_ptr<GradientDescent> sgd =
        std::make_shared<StochasticGradientDescent>(
            StochasticGradientDescent(LearningRate(0.01)));
    NeuralNetwork n_cpu(l5, loss, "CPU");
    NeuralNetwork n_gpu(l5_gpu, loss, "GPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    SharedStorage inp = std::make_shared<Storage>(in);
    std::vector<std::vector<SharedStorage>> helpers_cpu =
        prepare_helpers({l2, l4});
    std::vector<std::vector<SharedStorage>> helpers_gpu =
        prepare_helpers({l2, l4});
    // CPU CODE
    DebugInfo no_debugging = DebugInfo("", "");
    double cpuStart = cpuSecond();
    vector<SharedStorage> vals_cpu = n_cpu.allocate_forward(in.rows());
    vector<SharedStorage> grad_cpu = n_cpu.allocate_backward(in.rows());
    n_cpu.fill_hiddens(vals_cpu, in.transpose());
    n_cpu.forward(vals_cpu, "train", no_debugging);
    SharedStorage& grad_in = grad_cpu[grad_cpu.size() - 1];
    loss->grad_loss_cpu(grad_in, vals_cpu[vals_cpu.size() - 1], SharedTarget,
                        SharedTarget);
    n_cpu.backwards(grad_cpu, vals_cpu, no_debugging);
    n_cpu.update_weights(sgd, helpers_cpu, obs);
    Matrix after_weight = l2->return_parameters()[0]->return_data_const();
    double cpuEnd = cpuSecond() - cpuStart;
    // GPU PART
    double gpuStart = cpuSecond();
    vector<SharedStorage> vals_gpu = n_gpu.allocate_forward(in.rows());
    vector<SharedStorage> grad_gpu = n_gpu.allocate_backward(in.rows());
    n_gpu.fill_hiddens(vals_gpu, in.transpose());
    n_gpu.forward(vals_gpu, "train", no_debugging);
    SharedStorage& grad_in_gpu = grad_gpu[grad_gpu.size() - 1];
    loss->grad_loss_gpu(grad_in_gpu, vals_gpu[vals_gpu.size() - 1],
                        SharedTarget, SharedTarget);
    n_gpu.backwards(grad_gpu, vals_gpu, no_debugging);
    n_gpu.update_weights(sgd, helpers_gpu, obs);
    double gpuEnd = cpuSecond() - gpuStart;
    Matrix diff = l2->return_parameters()[0]->return_data_const() -
                  l2_gpu->return_parameters()[0]->return_data_const();
    //std::cout << diff << std::endl;
    std::cout << "The CPU took " << cpuEnd << " and hte GPU took " << gpuEnd
              << std::endl;
    dtype maxDiff = diff.array().abs().maxCoeff();
    REQUIRE(cpuEnd > gpuEnd);
    REQUIRE(maxDiff < 1e-6);
}
