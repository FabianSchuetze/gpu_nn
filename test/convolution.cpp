#include <eigen-git-mirror/Eigen/Core>
#define CATCH_CONFIG_MAIN
#include <sys/time.h>
#include <chrono>
#include <iostream>
#include <thread>
#include "../include/common.h"
#include "../include/neural_network.h"
#include "../third_party/catch/catch.hpp"
#include "../third_party/mnist/include/mnist/get_data.h"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
TEST_CASE("NeuralNetwork cpu", "[cpu]") {
    srand((unsigned int)time(0));
    FilterShape kernel(2, 2);
    Stride stride(1);
    Pad pad(1);
    Filters filters(3);
    ImageShape image(2, 2);
    Channels channels(2);
    int batches(3);
    int out_height =
        (image.first() + 2 * pad.get() - kernel.first()) / stride.get() + 1;
    int out_width =
        (image.second() + 2 * pad.get() - kernel.second()) / stride.get() + 1;
    Layer* inp1 =
        new Im2ColLayer(kernel, pad, stride, filters, image, channels);
    Layer* l2 = new Convolution(kernel, pad, stride, filters, image, channels);
    Matrix _input = Matrix::Random(
        image.first() * image.second() * channels.get(), batches);
    std::shared_ptr<Storage> input = std::make_shared<Storage>(_input);
    Matrix _out =
        Matrix::Zero(out_height * out_width, channels.get() * kernel.first() *
                                                 kernel.second() * batches);
    Matrix _out2 =
        Matrix::Zero(out_height * out_width * filters.get(), batches);
    std::shared_ptr<Storage> output_cpu = std::make_shared<Storage>(_out);
    std::shared_ptr<Storage> conv_out = std::make_shared<Storage>(_out2);
    inp1->forward_cpu(input, output_cpu, "train");
    l2->forward_cpu(output_cpu, conv_out, "train");
}

TEST_CASE("NeuralNetwork gpu", "[gpu]") {
    srand((unsigned int)time(0));
    FilterShape kernel(2, 2);
    Stride stride(1);
    Pad pad(1);
    Filters filters(3);
    ImageShape image(2, 2);
    Channels channels(2);
    int batches(3);
    int out_height =
        (image.first() + 2 * pad.get() - kernel.first()) / stride.get() + 1;
    int out_width =
        (image.second() + 2 * pad.get() - kernel.second()) / stride.get() + 1;
    Layer* inp1 =
        new Im2ColLayer(kernel, pad, stride, filters, image, channels);
    Layer* l2 = new Convolution(kernel, pad, stride, filters, image, channels);
    Matrix _input = Matrix::Random(
        image.first() * image.second() * channels.get(), batches);
    std::shared_ptr<Storage> input = std::make_shared<Storage>(_input);
    Matrix _out =
        Matrix::Zero(out_height * out_width, channels.get() * kernel.first() *
                                                 kernel.second() * batches);
    Matrix _out2 =
        Matrix::Zero(out_height * out_width * filters.get(), batches);
    std::shared_ptr<Storage> output_cpu = std::make_shared<Storage>(_out);
    std::shared_ptr<Storage> conv_out = std::make_shared<Storage>(_out2);
    inp1->forward_gpu(input, output_cpu, "train");
    l2->forward_gpu(output_cpu, conv_out, "train");
}

TEST_CASE("NeuralNetwork forward equivalence", "[forward equivalance]") {
    srand((unsigned int)time(0));
    FilterShape kernel(3, 3);
    Stride stride(1);
    Pad pad(1);
    Filters filters(3);
    ImageShape image(200, 200);
    Channels channels(3);
    int batches(32);
    int out_height =
        (image.first() + 2 * pad.get() - kernel.first()) / stride.get() + 1;
    int out_width =
        (image.second() + 2 * pad.get() - kernel.second()) / stride.get() + 1;
    Layer* im2col_cpu =
        new Im2ColLayer(kernel, pad, stride, filters, image, channels);
    Layer* im2col_gpu =
        new Im2ColLayer(kernel, pad, stride, filters, image, channels);
    Layer* conv_cpu =
        new Convolution(kernel, pad, stride, filters, image, channels);
    Layer* conv_gpu =
        new Convolution(kernel, pad, stride, filters, image, channels);
    Matrix _input = Matrix::Random(
        image.first() * image.second() * channels.get(), batches);
    std::shared_ptr<Storage> input = std::make_shared<Storage>(_input);
    Matrix _out_cpu =
        Matrix::Zero(out_height * out_width, channels.get() * kernel.first() *
                                                 kernel.second() * batches);
    Matrix _out_gpu =
        Matrix::Zero(out_height * out_width, channels.get() * kernel.first() *
                                                 kernel.second() * batches);
    Matrix _out2_cpu =
        Matrix::Zero(out_height * out_width * filters.get(), batches);
    Matrix _out2_gpu =
        Matrix::Zero(out_height * out_width * filters.get(), batches);
    std::shared_ptr<Storage> output_gpu = std::make_shared<Storage>(_out_gpu);
    std::shared_ptr<Storage> conv_out_cpu =
        std::make_shared<Storage>(_out2_cpu);
    std::shared_ptr<Storage> conv_out_gpu =
        std::make_shared<Storage>(_out2_gpu);
    double cpuStart(0);
    std::shared_ptr<Storage> output_cpu = std::make_shared<Storage>(_out_cpu);
    im2col_cpu->forward_cpu(input, output_cpu, "train");
    cpuStart = cpuSecond();
    conv_cpu->forward_cpu(output_cpu, conv_out_cpu, "train");
    double cpuEnd = cpuSecond() - cpuStart;
    delete conv_cpu;
    delete im2col_cpu;
    // GPU
    im2col_gpu->forward_gpu(input, output_gpu, "train");
    delete im2col_gpu;
    double gpuStart = cpuSecond();
    conv_gpu->forward_gpu(output_gpu, conv_out_gpu, "train");
    double gpuEnd = cpuSecond() - gpuStart;
    Matrix diff = conv_out_cpu->return_data_const() -
        conv_out_gpu->return_data_const();
    dtype maxDiff = diff.array().abs().maxCoeff();
    std::cout << "The CPU took " << cpuEnd << " and hte GPU took " <<
        gpuEnd << std::endl;
    REQUIRE(cpuEnd > gpuEnd);
    REQUIRE(maxDiff < 1e-5);
}
