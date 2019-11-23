#include <eigen-git-mirror/Eigen/Core>
//#define CATCH_CONFIG_MAIN
#include "../third_party/catch/catch.hpp"
#include "../include/common.h"
#include "../include/neural_network.h"
#include <sys/time.h>
#include <iostream>
int main() {
    srand((unsigned int) time(0));
    Layer* inp1 = new Convolution(FilterShape(3,3), Pad(1), Stride(1),
            Filters(3), ImageShape(5,5), Channels(3));
    int batch_size(3);
    Matrix in = Matrix::Random(5*5*3, batch_size);
    Matrix out = Matrix::Zero(5*5*3, batch_size);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    std::cout << storage_in->return_data_const() << std::endl;
    inp1->forward_gpu(storage_in, storage_out, "train");
    std::cout << storage_out->return_data_const() << std::endl;
}
