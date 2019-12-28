#include <eigen-git-mirror/Eigen/Core>
//#define CATCH_CONFIG_MAIN
//#include "../third_party/catch/catch.hpp"
#include "../include/common.h"
//#include "../include/layer/lstm.hpp"
//#include "../include/storage.h"
#include "../include/neural_network.h"
#include <iostream>
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
int main() {
//TEST_CASE("NeuralNetwork cpu", "[cpu]") {
    srand((unsigned int)0);
    Init* init = new Glorot();
    Layer* inp1 = new LSTM(Features(10), Features(12),init);
    //std::cout << "para1\n" <<
        //inp1->return_parameters()[0]->return_data_const() << std::endl;
    //Layer* inp1 = new LSTM;
    Matrix in = Matrix::Zero(12, 5);
    in(0, 0) = 1;
    in(2, 1) = 1;
    in(10, 2) = 1;
    in(5, 3) = 1;
    in(11, 4) = 1;
    Matrix out = Matrix::Zero(10, 5);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    inp1->forward_cpu(storage_in, storage_out, "train");
    std::cout << storage_out->return_data_const() << std::endl;
    std::cout << std::endl;
//}

//TEST_CASE("NeuralNetwork gpu", "[gpu]") {
//int main() {
    srand((unsigned int)0);
    Init* init2 = new Glorot();
    Layer* inp12 = new LSTM(Features(10), Features(12),init2);
    //std::cout << "para1\n" <<
        //inp12->return_parameters()[0]->return_data_const() << std::endl;
    Matrix in2 = Matrix::Zero(12, 5);
    in2(0, 0) = 1;
    in2(2, 1) = 1;
    in2(10, 2) = 1;
    in2(5, 3) = 1;
    in2(11, 4) = 1;
    Matrix out2 = Matrix::Zero(10, 5);
    std::shared_ptr<Storage> storage_in2 = std::make_shared<Storage>(in2);
    std::shared_ptr<Storage> storage_out2 = std::make_shared<Storage>(out2);
    inp12->forward_gpu(storage_in2, storage_out2, "train");
    std::cout << storage_out2->return_data_const() << std::endl;
}
