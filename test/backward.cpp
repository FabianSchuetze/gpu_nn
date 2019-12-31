#define CATCH_CONFIG_MAIN
#include "../third_party/catch/catch.hpp"
#include <cuda_runtime.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <sys/time.h>
#include <iostream>
#include <memory>
#include "../include/neural_network.h"

using std::vector;
typedef std::shared_ptr<Layer> s_Layer;
using std::make_shared;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

TEST_CASE("NeuralNetwork backward gpu", "[backward gpu]") {
    srand((unsigned int) time(0));
    int input_dimension = 5;
    int obs = 3;
    DebugInfo no_debugging = DebugInfo("", "");
    Init* init = new Glorot();
    s_Layer l1 = make_shared<Input>(Features(input_dimension));
    s_Layer l2 = make_shared<Dense>(Features(10), l1, init);
    s_Layer l3 = make_shared<Relu>(l2) ;
    s_Layer l4 = make_shared<Dense>(Features(2), l3, init);
    s_Layer l5 = make_shared<Softmax>(l4);
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
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    NeuralNetwork n1(l5, loss, "GPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    //the forward part
    SharedStorage inp = std::make_shared<Storage>(in);
    vector<SharedStorage> vals = n1.allocate_forward(in.rows());
    vector<SharedStorage> gradients = n1.allocate_backward(in.rows());
    n1.fill_hiddens(vals, in.transpose());
    n1.forward(vals, "train", no_debugging);
    SharedStorage& grad_in = gradients[gradients.size() -1];
    loss->grad_loss_gpu(grad_in, vals[vals.size() -1], SharedTarget, SharedTarget);
    n1.backwards(gradients, vals, no_debugging);
}

TEST_CASE("NeuralNetwork backward cpu", "[backward cpu]") {
    srand((unsigned int) time(0));
    DebugInfo no_debugging = DebugInfo("", "");
    int input_dimension = 5;
    int obs = 3;
    Init* init = new Glorot();
    s_Layer l1 = make_shared<Input>(Features(input_dimension));
    s_Layer l2 = make_shared<Dense>(Features(10), l1, init);
    s_Layer l3 = make_shared<Relu>(l2) ;
    s_Layer l4 = make_shared<Dense>(Features(2), l3, init);
    s_Layer l5 = make_shared<Softmax>(l4);
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
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    NeuralNetwork n1(l5, loss, "CPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    //the forward part
    SharedStorage inp = std::make_shared<Storage>(in);
    vector<SharedStorage> vals = n1.allocate_forward(in.rows());
    vector<SharedStorage> gradients = n1.allocate_backward(in.rows());
    n1.fill_hiddens(vals, in.transpose());
    n1.forward(vals, "train", no_debugging);
    SharedStorage& grad_in = gradients[gradients.size() -1];
    loss->grad_loss_cpu(grad_in, vals[vals.size() -1], SharedTarget, SharedTarget);
    n1.backwards(gradients, vals, no_debugging);
    //std::cout << l2->return_gradients()[0]->return_data_const() << std::endl;
}

TEST_CASE("NeuralNetwork backward equivalence", "[backward equivalence]") {
    srand((unsigned int) time(0));
    int input_dimension = 1024;
    int obs = 32;
    DebugInfo no_debugging = DebugInfo("", "");
    Init* init = new Glorot();
    s_Layer l1 = make_shared<Input>(Features(input_dimension));
    s_Layer l2 = make_shared<Dense>(Features(10), l1, init);
    s_Layer l3 = make_shared<Relu>(l2) ;
    s_Layer l4 = make_shared<Dense>(Features(2), l3, init);
    s_Layer l5 = make_shared<Softmax>(l4);
    s_Layer l1_gpu = make_shared<Input>(Features(input_dimension));
    s_Layer l2_gpu = make_shared<Dense>(Features(10), l1_gpu, init);
    s_Layer l3_gpu = make_shared<Relu>(l2_gpu) ;
    s_Layer l4_gpu = make_shared<Dense>(Features(2), l3_gpu, init);
    s_Layer l5_gpu = make_shared<Softmax>(l4_gpu);
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
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    NeuralNetwork n_cpu(l5, loss, "CPU");
    NeuralNetwork n_gpu(l5_gpu, loss, "GPU");
    Matrix in = Matrix::Random(obs, input_dimension);
    //the forward part
    SharedStorage inp = std::make_shared<Storage>(in);
    vector<SharedStorage> vals_cpu = n_cpu.allocate_forward(in.rows());
    vector<SharedStorage> vals_gpu = n_gpu.allocate_forward(in.rows());
    vector<SharedStorage> gradients_cpu = n_cpu.allocate_backward(in.rows());
    vector<SharedStorage> gradients_gpu = n_cpu.allocate_backward(in.rows());
    n_cpu.fill_hiddens(vals_cpu, in.transpose());
    n_gpu.fill_hiddens(vals_gpu, in.transpose());
    // CPU CODE
    double cpuStart = cpuSecond();
    n_cpu.forward(vals_cpu, "train", no_debugging);
    SharedStorage& grad_in_cpu = gradients_cpu[gradients_cpu.size() -1];
    loss->grad_loss_cpu(grad_in_cpu, vals_cpu[vals_cpu.size() -1], SharedTarget, SharedTarget);
    n_cpu.backwards(gradients_cpu, vals_cpu, no_debugging);
    double cpuEnd = cpuSecond() - cpuStart;
    // GPU CODE
    double gpuStart = cpuSecond();
    n_gpu.forward(vals_gpu, "train", no_debugging);
    SharedStorage& grad_in = gradients_gpu[gradients_gpu.size() -1];
    loss->grad_loss_gpu(grad_in, vals_gpu[vals_gpu.size() -1], SharedTarget, SharedTarget);
    n_gpu.backwards(gradients_gpu, vals_gpu, no_debugging);
    double gpuEnd = cpuSecond() - gpuStart;
    // Compare difference
    Matrix diff = gradients_cpu[0]->return_data_const() -
        gradients_gpu[0]->return_data_const();
    std::cout << "The CPU took " << cpuEnd << " and hte GPU took " <<
        gpuEnd << std::endl;
    dtype maxDiff = diff.array().abs().maxCoeff();
    REQUIRE(cpuEnd > gpuEnd);
    REQUIRE(maxDiff < 1e-6);
}
