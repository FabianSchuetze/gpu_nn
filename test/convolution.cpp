#include <eigen-git-mirror/Eigen/Core>
//#define CATCH_CONFIG_MAIN
#include <sys/time.h>
#include <chrono>
#include <iostream>
#include <thread>
#include "../include/common.h"
#include "../include/neural_network.h"
#include "../third_party/catch/catch.hpp"
#include "../third_party/mnist/include/mnist/get_data.h"
int main() {
    srand((unsigned int)time(0));
    Filters filters(2);
    Mnist data = Mnist();
    srand((unsigned int)time(0));
    // Layer* l1 = new Input(data.get_x_train().cols());
    Layer* inp1 = new Convolution(FilterShape(3, 3), Pad(1), Stride(1),
                                  Filters(2), ImageShape(3, 3), Channels(1));
    Layer* inp2 = new Convolution(FilterShape(3, 3), Pad(1), Stride(1),
                                  Filters(2), ImageShape(3, 3), Channels(1));
    // Layer* l2 = new Dense(10, data.get_x_train().cols() * 3);
    // Layer* l3 = new Softmax;
    // std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy(
    //"GPU"));
    // NeuralNetwork n1({l1, inp1, l2, l3}, loss, "GPU");
    // std::shared_ptr<GradientDescent> sgd =
    // std::make_shared<StochasticGradientDescent>(0.001);
    // n1.train(data.get_x_train(), data.get_y_train(), sgd, Epochs(10),
    // Patience(10), BatchSize(32));
    // delete l1;
    // delete l2;
    // delete l3;
    //}
    // Layer* inp2 = new Convolution(FilterShape(3, 3), Pad(1), Stride(1),
    // filters, ImageShape(5, 5), Channels(2));
    int batch_size(2);
    Matrix in = Matrix::Random(3 * 3 * 1, batch_size);
    Matrix out = Matrix::Zero(3 * 3 * filters.get(), batch_size);
    Matrix grad_out = Matrix::Zero(3 * 3 * filters.get(), batch_size);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_in_cpu = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    std::shared_ptr<Storage> storage_out_cpu =
        std::make_shared<Storage>(grad_out);
    std::cout << storage_in->return_data_const() << std::endl;
    inp1->forward_gpu(storage_in, storage_out, "train");
    // inp1->backward_gpu(storage_in, storage_out, storage_grad_out);
    std::cout << "gpu data\n" << storage_out->return_data_const() << std::endl;
    inp2->forward_cpu(storage_in_cpu, storage_out_cpu, "train");
    std::cout << "cpu data\n"
              << storage_out_cpu->return_data_const() << std::endl;
}
