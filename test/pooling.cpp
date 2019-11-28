#include <eigen-git-mirror/Eigen/Core>
#include <memory>
#define CATCH_CONFIG_MAIN
#include "../include/common.h"
#include "../include/neural_network.h"
#include "../third_party/catch/catch.hpp"
//#include "../include/layer/softmax.h"
//#include "../include/storage.h"
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
TEST_CASE("Pooling cpu", "[cpu]") {
    Window window(2);
    Stride stride(1);
    ImageShape image(2, 4);
    Channels channels(2);
    Pad pad(1);
    int batches(3);
    int out_height = (image.first() - window.get()) / stride.get() + 1;
    int out_width = (image.second() - window.get()) / stride.get() + 1;
    srand((unsigned int)time(0));
    Layer* inp1 =
        new Pooling(window, stride, ImageShape(2, 4), Channels(2));
    Matrix test = Matrix::Random(
        image.first() * image.second() * channels.get(), batches);
    std::shared_ptr<Storage> input = std::make_shared<Storage>(test);
    // std::cout << input->return_data_const() << std::endl;
    Matrix out(out_height * out_width * channels.get(), batches);
    out.setZero();
    std::shared_ptr<Storage> output_cpu = std::make_shared<Storage>(out);
    inp1->forward_cpu(input, output_cpu, "train");
    //std::cout << "out cpu\n" << output_cpu->return_data_const() << std::endl;
}

TEST_CASE("Pooling gpu", "[gpu]") {
    Window window(2);
    Stride stride(1);
    ImageShape image(2, 4);
    Channels channels(2);
    int batches(3);
    int out_height = (image.first() - window.get()) / stride.get() + 1;
    int out_width = (image.second() - window.get()) / stride.get() + 1;
    srand((unsigned int)time(0));
    Layer* inp1 =
        new Pooling(Window(2), Stride(1), ImageShape(2, 4), Channels(2));
    Matrix test = Matrix::Random(
        image.first() * image.second() * channels.get(), batches);
    std::shared_ptr<Storage> input = std::make_shared<Storage>(test);
    // std::cout << input->return_data_const() << std::endl;
    Matrix out(out_height * out_width * channels.get(), batches);
    out.setZero();
    std::shared_ptr<Storage> output_gpu = std::make_shared<Storage>(out);
    inp1->forward_gpu(input, output_gpu, "train");
    //std::cout << "gpu\n" << output_gpu->return_data_const() << std::endl;
}

TEST_CASE("Pooling forward comparison", "[comparison]") {
    Window window(2);
    Stride stride(2);
    ImageShape image(270, 270);
    Channels channels(20);
    int batches(32);
    int out_height = (image.first() - window.get()) / stride.get() + 1;
    int out_width = (image.second() - window.get()) / stride.get() + 1;
    srand((unsigned int)time(0));
    Layer* inp_cpu = new Pooling(window, stride, image, channels);
    Layer* inp_gpu = new Pooling(window, stride, image, channels);
    Matrix test = Matrix::Random(
        image.first() * image.second() * channels.get(), batches);
    std::shared_ptr<Storage> input_cpu = std::make_shared<Storage>(test);
    std::shared_ptr<Storage> input_gpu = std::make_shared<Storage>(test);
    Matrix out(out_height * out_width * channels.get(), batches);
    out.setZero();
    std::shared_ptr<Storage> output_cpu = std::make_shared<Storage>(out);
    std::shared_ptr<Storage> output_gpu = std::make_shared<Storage>(out);
    inp_cpu->forward_cpu(input_cpu, output_cpu, "train");
    double cpuStart = cpuSecond();
    inp_cpu->forward_cpu(input_cpu, output_cpu, "train");
    double cpuEnd = cpuSecond() - cpuStart;
    inp_gpu->forward_gpu(input_gpu, output_gpu, "train");
    delete inp_cpu;
    double gpuStart = cpuSecond();
    inp_gpu->forward_gpu(input_gpu, output_gpu, "train");
    double gpuEnd = cpuSecond() - gpuStart;
    Matrix diff =
        output_cpu->return_data_const() - output_gpu->return_data_const();
    dtype maxDiff = diff.cwiseAbs().maxCoeff();
    REQUIRE(cpuEnd > gpuEnd);
    REQUIRE(maxDiff < 1e-6);
    // std::cout << "maxdiff " << maxdiff << std::endl;
    std::cout << "The CPU took " << cpuEnd << " and hte GPU took " << gpuEnd
              << std::endl;
}
TEST_CASE("Pooling cpu backwards", "[cpu]") {
    Window window(2);
    Stride stride(1);
    ImageShape image(2, 4);
    Channels channels(2);
    int batches(3);
    int out_height = (image.first() - window.get()) / stride.get() + 1;
    int out_width = (image.second() - window.get()) / stride.get() + 1;
    srand((unsigned int)time(0));
    Layer* inp1 =
        new Pooling(Window(2), Stride(1), ImageShape(2, 4), Channels(2));
    Matrix test = Matrix::Random(
        image.first() * image.second() * channels.get(), batches);
    std::shared_ptr<Storage> input = std::make_shared<Storage>(test);
    // std::cout << input->return_data_const() << std::endl;
    Matrix out(out_height * out_width * channels.get(), batches);
    out.setZero();
    std::shared_ptr<Storage> output_cpu = std::make_shared<Storage>(out);
    inp1->forward_cpu(input, output_cpu, "train");
    // std::cout << "out cpu\n" << output_cpu->return_data_const() << std::endl;
}

TEST_CASE("Pooling gpu backwards", "[gpu]") {
    Window window(2);
    Stride stride(1);
    ImageShape image(270, 270);
    Channels channels(1);
    int batches(32);
    int out_height = (image.first() - window.get()) / stride.get() + 1;
    int out_width = (image.second() - window.get()) / stride.get() + 1;
    srand((unsigned int)time(0));
    Layer* inp1 = new Pooling(window, stride, image, channels);
    Matrix test = Matrix::Random(
        image.first() * image.second() * channels.get(), batches);
    std::shared_ptr<Storage> input = std::make_shared<Storage>(test);
    Matrix grad_out =
        Matrix::Zero(image.first() * image.second() * channels.get(), batches);
    std::shared_ptr<Storage> gradient_out = std::make_shared<Storage>(grad_out);
    // std::cout << input->return_data_const() << std::endl;
    Matrix out(out_height * out_width * channels.get(), batches);
    out.setZero();
    std::shared_ptr<Storage> output_gpu = std::make_shared<Storage>(out);
    inp1->forward_gpu(input, output_gpu, "train");
    inp1->backward_gpu(input, output_gpu, gradient_out);
    // std::cout << "gpu\n" << output_gpu->return_data_const() << std::endl;
}
TEST_CASE("Pooling backward cpu", "[cpu]") {
    Window window(2);
    Stride stride(2);
    ImageShape image(270, 270);
    Channels channels(2);
    int batches(3);
    int out_height = (image.first() - window.get()) / stride.get() + 1;
    int out_width = (image.second() - window.get()) / stride.get() + 1;
    srand((unsigned int)time(0));
    Layer* inp_cpu = new Pooling(window, stride, image, channels);
    Matrix test = Matrix::Random(
        image.first() * image.second() * channels.get(), batches);
    Matrix grad_out =
        Matrix::Zero(image.first() * image.second() * channels.get(), batches);
    std::shared_ptr<Storage> input_cpu = std::make_shared<Storage>(test);
    // std::cout << "input\n" << input_cpu->return_data_const() << std::endl;
    std::shared_ptr<Storage> gradient_out = std::make_shared<Storage>(grad_out);
    Matrix out(out_height * out_width * channels.get(), batches);
    out.setZero();
    std::shared_ptr<Storage> output_cpu = std::make_shared<Storage>(out);
    inp_cpu->forward_cpu(input_cpu, output_cpu, "train");

    inp_cpu->backward_cpu(input_cpu, output_cpu, gradient_out);
    // std::cout << gradient_out->return_data_const() << std::endl;
    dtype maxdiff = gradient_out->return_data_const().maxCoeff() -
                    input_cpu->return_data_const().maxCoeff();
    dtype mindiff = gradient_out->return_data_const().minCoeff() -
                    input_cpu->return_data_const().minCoeff();
    REQUIRE(maxdiff < 1e-5);
    REQUIRE(mindiff > 0);
}

TEST_CASE("Pooling backward comparison", "[backward comparison]") {
    Window window(2);
    Stride stride(2);
    ImageShape image(270, 270);
    Channels channels(20);
    int batches(32);
    int out_height = (image.first() - window.get()) / stride.get() + 1;
    int out_width = (image.second() - window.get()) / stride.get() + 1;
    srand((unsigned int)time(0));
    Layer* inp_cpu = new Pooling(window, stride, image, channels);
    Layer* inp_gpu = new Pooling(window, stride, image, channels);
    Matrix test = Matrix::Random(
        image.first() * image.second() * channels.get(), batches);
    Matrix grad_out =
        Matrix::Zero(image.first() * image.second() * channels.get(), batches);
    Matrix grad_out_gpu =
        Matrix::Zero(image.first() * image.second() * channels.get(), batches);
    std::shared_ptr<Storage> input_cpu = std::make_shared<Storage>(test);
    std::shared_ptr<Storage> input_gpu = std::make_shared<Storage>(test);
    // std::cout << "input\n" << input_cpu->return_data_const() << std::endl;
    std::shared_ptr<Storage> gradient_cpu = std::make_shared<Storage>(grad_out);
    std::shared_ptr<Storage> gradient_gpu =
        std::make_shared<Storage>(grad_out_gpu);
    Matrix out(out_height * out_width * channels.get(), batches);
    out.setZero();
    Matrix out_gpu(out_height * out_width * channels.get(), batches);
    out_gpu.setZero();
    std::shared_ptr<Storage> output_cpu = std::make_shared<Storage>(out);
    std::shared_ptr<Storage> output_gpu = std::make_shared<Storage>(out_gpu);
    inp_cpu->forward_cpu(input_cpu, output_cpu, "train");
    inp_gpu->forward_gpu(input_gpu, output_gpu, "train");
    double cpuStart = cpuSecond();
    inp_cpu->backward_cpu(input_cpu, output_cpu, gradient_cpu);
    double cpuEnd = cpuSecond() - cpuStart;
    double gpuStart = cpuSecond();
    inp_gpu->backward_gpu(input_gpu, output_gpu, gradient_gpu);
    double gpuEnd = cpuSecond() - gpuStart;
    Matrix diff =
        output_cpu->return_data_const() - output_gpu->return_data_const();
    dtype maxDiff = diff.cwiseAbs().maxCoeff();
    std::cout << "The CPU took " << cpuEnd << " and hte GPU took " << gpuEnd
              << std::endl;
    REQUIRE(maxDiff < 1e-5);
    REQUIRE(gpuEnd < cpuEnd);
}
