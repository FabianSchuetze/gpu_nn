#include <eigen-git-mirror/Eigen/Core>
//#define CATCH_CONFIG_MAIN
#include <sys/time.h>
#include <chrono>
#include <iostream>
#include <thread>
#include "../include/common.h"
#include "../include/neural_network.h"
#include "../third_party/catch/catch.hpp"
int main() {
    srand((unsigned int)time(0));
    Filters filters(2);
    Layer* inp1 = new Convolution(FilterShape(3, 3), Pad(1), Stride(1), filters,
                                  ImageShape(5, 5), Channels(2));
    //Layer* inp2 = new Convolution(FilterShape(3, 3), Pad(1), Stride(1), filters,
                                  //ImageShape(5, 5), Channels(2));
    int batch_size(2);
    Matrix in = Matrix::Random(5 * 5 * 2, batch_size);
    Matrix out = Matrix::Zero(5 * 5 * filters.get(), batch_size);
    Matrix grad_out = Matrix::Zero(5 * 5 * 2, batch_size);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    std::shared_ptr<Storage> storage_grad_out =
        std::make_shared<Storage>(grad_out);
    std::cout << storage_in->return_data_const() << std::endl;
    inp1->forward_gpu(storage_in, storage_out, "train");
    inp1->backward_gpu(storage_in, storage_out, storage_grad_out);
    // std::cout << "gpu data\n" <<
    // storage_out->return_data_const() << std::endl;
    // inp2->forward_cpu(storage_in, storage_out_cpu, "train");
    // std::cout << "cpu data\n" << storage_out_cpu->return_data_const() <<
    // std::endl;
}
