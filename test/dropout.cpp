#define CATCH_CONFIG_MAIN
#include <sys/time.h>
#include <eigen-git-mirror/Eigen/Core>
#include <iostream>
#include <memory>
#include "../include/neural_network.h"
#include "../third_party/catch/catch.hpp"

using std::make_shared;
using std::shared_ptr;
using std::vector;
typedef std::shared_ptr<Storage> SharedStorage;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

TEST_CASE("Dropout forward_gpu", "[gpu]") {
    srand((unsigned int)time(0));
    Matrix in = Matrix::Random(10, 3);
    Matrix out = Matrix::Zero(10, 3);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    std::shared_ptr<Storage> storage_out2 = std::make_shared<Storage>(out);
    Layer* inp1 = new Dropout(0.5);
    inp1->forward_gpu(storage_in, storage_out, "train");
    inp1->forward_gpu(storage_in, storage_out2, "train");
    bool equal = true;
    const Matrix& first_out = storage_out->return_data_const();
    const Matrix& second_out = storage_out2->return_data_const();
    for (int i = 0; i < storage_out->get_rows(); ++i) {
        for (int j = 0; j < storage_out->get_cols(); ++j) {
            if ((first_out(i, j) != 0) and (second_out(i, j) != 0))
                equal = first_out(i, j) == second_out(i, j);
            if (not equal) break;
        }
    }
    REQUIRE(equal);
}

TEST_CASE("Dropout forward_gpu test", "[gpu test]") {
    srand((unsigned int)time(0));
    Matrix in = Matrix::Random(10, 3);
    Matrix out = Matrix::Zero(10, 3);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    std::shared_ptr<Storage> storage_out2 = std::make_shared<Storage>(out);
    Layer* inp1 = new Dropout(0.5);
    inp1->forward_gpu(storage_in, storage_out, "predict");
    inp1->forward_gpu(storage_in, storage_out2, "predict");
    const Matrix& diff =
        storage_out->return_data_const() - storage_in->return_data_const();
    dtype max_diff = diff.cwiseAbs().array().maxCoeff();
    REQUIRE(max_diff < 1e-5);
}

TEST_CASE("Dropout forward_cpu", "[cpu]") {
    srand((unsigned int)time(0));
    Matrix in = Matrix::Random(10, 3);
    Matrix out = Matrix::Zero(10, 3);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    std::shared_ptr<Storage> storage_out2 = std::make_shared<Storage>(out);
    Layer* inp1 = new Dropout(0.5);
    inp1->forward_cpu(storage_in, storage_out, "train");
    inp1->forward_cpu(storage_in, storage_out2, "train");
    bool equal = true;
    const Matrix& first_out = storage_out->return_data_const();
    const Matrix& second_out = storage_out2->return_data_const();
    for (int i = 0; i < storage_out->get_rows(); ++i) {
        for (int j = 0; j < storage_out->get_cols(); ++j) {
            if ((first_out(i, j) != 0) and (second_out(i, j) != 0))
                equal = first_out(i, j) == second_out(i, j);
            if (not equal) break;
        }
    }
    REQUIRE(equal);
}

TEST_CASE("Dropout forward_cpu test", "[cpu test]") {
    srand((unsigned int)time(0));
    Matrix in = Matrix::Random(10, 3);
    Matrix out = Matrix::Zero(10, 3);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    std::shared_ptr<Storage> storage_out2 = std::make_shared<Storage>(out);
    Layer* inp1 = new Dropout(0.5);
    inp1->forward_cpu(storage_in, storage_out, "predict");
    inp1->forward_cpu(storage_in, storage_out2, "predict");
    const Matrix& diff =
        storage_out->return_data_const() - storage_in->return_data_const();
    dtype max_diff = diff.cwiseAbs().array().maxCoeff();
    REQUIRE(max_diff < 1e-5);
}

TEST_CASE("Dropout backward gpu", "[backward gpu]") {
    srand((unsigned int)time(0));
    Matrix in = Matrix::Random(10, 3);
    Matrix in_pred = Matrix::Random(10, 3);
    Matrix out = Matrix::Zero(10, 3);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    std::shared_ptr<Storage> grad_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> grad_out_cpu = std::make_shared<Storage>(out);
    std::shared_ptr<Storage> grad_out_gpu = std::make_shared<Storage>(out);
    Layer* inp1 = new Dropout(0.5);
    inp1->forward_cpu(storage_in, storage_out, "train");
    inp1->backward_cpu(storage_in, grad_in, grad_out_cpu);
    inp1->backward_gpu(storage_in, grad_in, grad_out_gpu);
    const Matrix& diff =
        grad_out_cpu->return_data_const() - grad_out_gpu->return_data_const();
    dtype max_diff = diff.cwiseAbs().array().maxCoeff();
    REQUIRE(max_diff < 1e-5);
}

TEST_CASE("Dropout comparison", "[comparison]") {
    srand((unsigned int)time(0));
    Matrix in = Matrix::Random(1024, 32);
    Matrix out = Matrix::Zero(1024, 32);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    std::shared_ptr<Storage> storage_out2 = std::make_shared<Storage>(out);
    Layer* inp_cpu = new Dropout(0.5);
    Layer* inp_gpu = new Dropout(0.5);
    inp_gpu->forward_gpu(storage_in, storage_out2, "train");
    double cpuStart = cpuSecond();
    inp_cpu->forward_cpu(storage_in, storage_out, "train");
    double cpuEnd = cpuSecond() - cpuStart;
    double gpuStart = cpuSecond();
    inp_gpu->forward_gpu(storage_in, storage_out2, "train");
    double gpuEnd = cpuSecond() - gpuStart;
    std::cout << "The CPU took " << cpuEnd << " and hte GPU took " << gpuEnd
              << std::endl;
}
