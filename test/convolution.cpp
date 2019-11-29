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
    Layer* l2 = 
        new Convolution(kernel, pad, stride, filters, image, channels);
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
    l2->forward_cpu(output_cpu,  conv_out, "train");
    std::cout << conv_out->return_data_const() << std::endl;
}
