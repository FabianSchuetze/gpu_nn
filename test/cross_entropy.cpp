#include <eigen-git-mirror/Eigen/Core>
#define CATCH_CONFIG_MAIN
#include <sys/time.h>
#include <cmath>
#include "../include/common.h"
#include "../include/layer/softmax.h"
#include "../include/neural_network.h"
#include "../include/storage.h"
#include "../third_party/catch/catch.hpp"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

TEST_CASE("CrossEntropy cpu", "[cpu]") {
    srand((unsigned int)time(0));
    Loss* inp1;
    CrossEntropy cross_entropy;
    inp1 = &cross_entropy;
    Matrix tmp = Matrix::Random(6, 1);
    Matrix prediction = Matrix::Zero(6, 2);
    Matrix tmp_target = Matrix::Random(6, 1);
    Matrix target = Matrix::Random(6, 2);
    for (int i = 0; i < tmp.rows(); ++i) {
        dtype pred = tmp(i, 0) * 0.5 + 0.5;
        prediction(i, 0) = 1 - pred;
        prediction(i, 1) = pred;
    }
    for (int i = 0; i < target.rows(); ++i) {
        if (tmp_target(i) > 0.0) {
            target(i, 0) = 0.;
            target(i, 1) = 1.;
        } else {
            target(i, 0) = 1.;
            target(i, 1) = 0.;
        }
    }
    std::shared_ptr<Storage> SharedTarget =
        std::make_shared<Storage>(target.transpose());
    std::shared_ptr<Storage> SharedPrediction =
        std::make_shared<Storage>(prediction.transpose());
    dtype loss = inp1->loss_cpu(SharedPrediction, SharedTarget);
    REQUIRE(loss > 0.);
}

TEST_CASE("CrossEntropy gpu", "[gpu]") {
    srand((unsigned int)time(0));
    Loss* inp1;
    CrossEntropy cross_entropy;
    inp1 = &cross_entropy;
    Matrix tmp = Matrix::Random(6, 1);
    Matrix prediction = Matrix::Zero(6, 2);
    Matrix tmp_target = Matrix::Random(6, 1);
    Matrix target = Matrix::Random(6, 2);
    for (int i = 0; i < tmp.rows(); ++i) {
        dtype pred = tmp(i, 0) * 0.5 + 0.5;
        prediction(i, 0) = 1 - pred;
        prediction(i, 1) = pred;
    }
    for (int i = 0; i < target.rows(); ++i) {
        if (tmp_target(i) > 0.0) {
            target(i, 0) = 0.;
            target(i, 1) = 1.;
        } else {
            target(i, 0) = 1.;
            target(i, 1) = 0.;
        }
    }
    std::shared_ptr<Storage> SharedTarget =
        std::make_shared<Storage>(target.transpose());
    std::shared_ptr<Storage> SharedPrediction =
        std::make_shared<Storage>(prediction.transpose());
    dtype loss = inp1->loss_gpu(SharedPrediction, SharedTarget);
    REQUIRE(loss > 0.);
}

TEST_CASE("CrossEntropy equivalence loss", "[equivalence]") {
    srand((unsigned int)time(0));
    Loss* inp1;
    CrossEntropy cross_entropy;
    inp1 = &cross_entropy;
    Matrix tmp = Matrix::Random(1024, 1);
    Matrix prediction = Matrix::Zero(1024, 2);
    Matrix tmp_target = Matrix::Random(1024, 1);
    Matrix target = Matrix::Random(1024, 2);
    for (int i = 0; i < tmp.rows(); ++i) {
        dtype pred = tmp(i, 0) * 0.5 + 0.5;
        prediction(i, 0) = 1 - pred;
        prediction(i, 1) = pred;
    }
    for (int i = 0; i < target.rows(); ++i) {
        if (tmp_target(i) > 0.0) {
            target(i, 0) = 0.;
            target(i, 1) = 1.;
        } else {
            target(i, 0) = 1.;
            target(i, 1) = 0.;
        }
    }
    std::shared_ptr<Storage> SharedTarget =
        std::make_shared<Storage>(target.transpose());
    std::shared_ptr<Storage> SharedPrediction =
        std::make_shared<Storage>(prediction.transpose());
    double cpuStart = cpuSecond();
    dtype loss_cpu = inp1->loss_cpu(SharedPrediction, SharedTarget);
    double cpuEnd = cpuSecond() - cpuStart;
    double gpuStart = cpuSecond();
    dtype loss_gpu = inp1->loss_gpu(SharedPrediction, SharedTarget);
    double gpuEnd = cpuSecond() - gpuStart;
    std::cout << "loss cpu\n" << loss_cpu << std::endl;
    std::cout << "loss gpu\n" << loss_gpu << std::endl;
    dtype diff = std::abs(loss_cpu / 1024. - loss_gpu / 1024.);
    REQUIRE(gpuEnd < cpuEnd);
    REQUIRE(diff < 1e-5);
}

TEST_CASE("CrossEntropy equivalence gradient", "[gradient]") {
    srand((unsigned int)time(0));
    Loss* inp1;
    CrossEntropy cross_entropy;
    inp1 = &cross_entropy;
    Matrix tmp = Matrix::Random(10024, 1);
    Matrix prediction = Matrix::Zero(10024, 2);
    Matrix gradient_cpu = Matrix::Zero(10024, 2);
    Matrix gradient_gpu = Matrix::Zero(10024, 2);
    Matrix values = Matrix::Zero(10024, 2);
    Matrix tmp_target = Matrix::Random(10024, 1);
    Matrix target = Matrix::Random(10024, 2);
    for (int i = 0; i < tmp.rows(); ++i) {
        dtype pred = tmp(i, 0) * 0.5 + 0.5;
        prediction(i, 0) = 1 - pred;
        prediction(i, 1) = pred;
    }
    for (int i = 0; i < target.rows(); ++i) {
        if (tmp_target(i) > 0.0) {
            target(i, 0) = 0.;
            target(i, 1) = 1.;
        } else {
            target(i, 0) = 1.;
            target(i, 1) = 0.;
        }
    }
    std::shared_ptr<Storage> SharedTarget =
        std::make_shared<Storage>(target.transpose());
    std::shared_ptr<Storage> SharedPrediction =
        std::make_shared<Storage>(prediction.transpose());
    std::shared_ptr<Storage> SharedGradient_GPU =
        std::make_shared<Storage>(gradient_gpu.transpose());
    std::shared_ptr<Storage> SharedGradient_CPU =
        std::make_shared<Storage>(gradient_cpu.transpose());
    std::shared_ptr<Storage> SharedValues =
        std::make_shared<Storage>(values.transpose());
    double cpuStart = cpuSecond();
    inp1->grad_loss_cpu(SharedGradient_CPU, SharedPrediction, SharedTarget,
                        SharedValues);
    double cpuEnd = cpuSecond() - cpuStart;
    double gpuStart = cpuSecond();
    inp1->grad_loss_gpu(SharedGradient_GPU, SharedPrediction, SharedTarget,
                        SharedValues);
    double gpuEnd = cpuSecond() - gpuStart;
    Matrix diff = SharedGradient_CPU->return_data_const() -
        SharedGradient_GPU->return_data_const();
    dtype maxDiff = diff.array().abs().maxCoeff();
    REQUIRE(gpuEnd < cpuEnd);
    // THR GPU IS RELATIVELY SLOW HERE!!!
    REQUIRE(maxDiff < 1e-5);
}
