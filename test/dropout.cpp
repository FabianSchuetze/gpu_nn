//#define CATCH_CONFIG_MAIN
#include "../include/neural_network.h"
#include <eigen-git-mirror/Eigen/Core>
#include <memory>
#include "../third_party/catch/catch.hpp"
#include <sys/time.h>
#include <iostream>

using std::make_shared;
using std::shared_ptr;
using std::vector;
typedef std::shared_ptr<Storage> SharedStorage;

//double cpuSecond() {
    //struct timeval tp;
    //gettimeofday(&tp, NULL);
    //return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
//}
//TEST_CASE("Dense forward_gpu", "[gpu]") {
int main() {
    srand((unsigned int)time(0));
    Matrix in = Matrix::Random(10, 3);
    Matrix out = Matrix::Zero(10, 3);
    //dtype begin = out(0, 0);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    std::shared_ptr<Storage> storage_out2 = std::make_shared<Storage>(out);
    std::cout << storage_in->return_data_const() << std::endl;
    Layer* inp1 = new Dropout(0.5);
    inp1->forward_gpu(storage_in, storage_out, "train");
    std::cout << "fist storage\n" << storage_out->return_data_const() << std::endl;
    inp1->forward_gpu(storage_in, storage_out2, "train");
    std::cout << "second storage\n" <<
        storage_out2->return_data_const() << std::endl;
    //dtype end = storage_out->return_data_const()(0, 0);
    //REQUIRE(begin != end);
}
//}
