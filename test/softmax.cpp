#include <eigen-git-mirror/Eigen/Core>
#define CATCH_CONFIG_MAIN
#include "../third_party/catch/catch.hpp"
#include "../include/common.h"
#include "../include/layer/softmax.h"
#include "../include/storage.h"
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
TEST_CASE("NeuralNetwork gpu", "[gpu]") {
    srand((unsigned int) time(0));
    Layer* inp1;
    Softmax s1;
    inp1 = &s1;
    Matrix in = Matrix::Random(6, 5);
    Matrix out = Matrix::Zero(6, 5);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    inp1->forward_gpu(storage_in, storage_out);
    Vector sum = storage_out->return_data_const().colwise().sum();
    REQUIRE(sum(0) == Approx(1.0));
    REQUIRE(sum.sum() == Approx(5));
}

TEST_CASE("NeuralNetwork cpu", "[cpu]") {
    srand((unsigned int) time(0));
    Layer* inp1;
    Softmax s1;
    inp1 = &s1;
    Matrix in = Matrix::Random(6, 5);
    Matrix out = Matrix::Zero(6, 5);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    inp1->forward_cpu(storage_in, storage_out);
    Vector sum = storage_out->return_data_const().colwise().sum();
    REQUIRE(sum(0) == Approx(1.0));
    REQUIRE(sum.sum() == Approx(5.0));
}

TEST_CASE("NeuralNetwork equivalence", "[equivalence]") {
    srand((unsigned int) time(0));
    Layer* inp1;
    Softmax s1;
    inp1 = &s1;
    Matrix in = Matrix::Random(1024, 100);
    Matrix out = Matrix::Zero(1024, 100);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_gpu = std::make_shared<Storage>(out);
    std::shared_ptr<Storage> storage_cpu = std::make_shared<Storage>(out);
    double cpuStart = cpuSecond();
    inp1->forward_cpu(storage_in, storage_cpu);
    double cpuEnd = cpuSecond() - cpuStart;
    double gpuStart = cpuSecond();
    inp1->forward_gpu(storage_in, storage_gpu);
    double gpuEnd = cpuSecond() - gpuStart;
    std::cout << "The CPU took " << cpuEnd << " and hte GPU took " <<
        gpuEnd << std::endl;
    Matrix diff = storage_cpu->return_data_const() -
        storage_gpu->return_data_const();
    dtype maxDiff = diff.array().abs().maxCoeff();
    REQUIRE(cpuEnd > gpuEnd);
    REQUIRE(maxDiff < 1e-6);
}
