#define CATCH_CONFIG_MAIN
#include "../include/layer/dense.h"
#include <eigen-git-mirror/Eigen/Core>
#include <memory>
#include "../include/common.h"
#include "../include/layer/layer.h"
#include "../third_party/catch/catch.hpp"
//#include "../include/math.h"
#include <sys/time.h>
#include "../include/storage.h"

using std::make_shared;
using std::shared_ptr;
using std::vector;
typedef std::shared_ptr<Storage> SharedStorage;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
TEST_CASE("Dense forward_gpu", "[gpu]") {
    srand((unsigned int)time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* inp1;
    Dense s1(6, 5, handle);
    inp1 = &s1;
    Eigen::MatrixXd in = Eigen::MatrixXd::Random(5, 3);
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(6, 3);
    double begin = out(0, 0);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    inp1->forward_gpu(storage_in, storage_out);
    double end = storage_out->return_data_const()(0, 0);
    REQUIRE(begin != end);
}

TEST_CASE("Dense backward_gpu", "[gpu]") {
    srand((unsigned int)time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* inp1;
    Dense s1(6, 5, handle);
    inp1 = &s1;
    Eigen::MatrixXd gradient_in = Eigen::MatrixXd::Random(6, 3);
    Eigen::MatrixXd gradient_out = Eigen::MatrixXd::Zero(5, 3);
    Eigen::MatrixXd values = Eigen::MatrixXd::Random(5, 3);
    SharedStorage shared_gradient_in = make_shared<Storage>(gradient_in);
    SharedStorage shared_gradient_out = make_shared<Storage>(gradient_out);
    SharedStorage shared_values = make_shared<Storage>(values);
    vector<SharedStorage> grad_vec = {shared_gradient_out, shared_gradient_in};
    int layer = 1;
    inp1->backward_gpu(layer, shared_values, grad_vec);
    REQUIRE(inp1->return_gradients()[1]->return_data_const() ==
            gradient_in.rowwise().sum());
}

TEST_CASE("Dense forward_cpu", "[cpu]") {
    srand((unsigned int)time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* inp1;
    Dense s1(6, 5, handle);
    inp1 = &s1;
    Eigen::MatrixXd in = Eigen::MatrixXd::Random(5, 3);
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(6, 3);
    double begin = out(0, 0);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    inp1->forward_cpu(storage_in, storage_out);
    double end = storage_out->return_data_const()(0, 0);
    REQUIRE(begin != end);
}

TEST_CASE("Dense backard_cpu", "[cpu]") {
    srand((unsigned int)time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* inp1;
    Dense s1(6, 5, handle);
    inp1 = &s1;
    Eigen::MatrixXd gradient_in = Eigen::MatrixXd::Random(6, 3);
    Eigen::MatrixXd gradient_out = Eigen::MatrixXd::Zero(5, 3);
    Eigen::MatrixXd values = Eigen::MatrixXd::Random(5, 3);
    SharedStorage shared_gradient_in = make_shared<Storage>(gradient_in);
    SharedStorage shared_gradient_out = make_shared<Storage>(gradient_out);
    SharedStorage shared_values = make_shared<Storage>(values);
    vector<SharedStorage> grad_vec = {shared_gradient_out, shared_gradient_in};
    int layer = 1;
    inp1->backward_cpu(layer, shared_values, grad_vec);
    REQUIRE(inp1->return_gradients()[1]->return_data_const() ==
            gradient_in.rowwise().sum());
}

TEST_CASE("Dense backard equivalence", "[backward equivalence]") {
    srand((unsigned int)time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* inp1;
    Dense s1(1024, 1000, handle);
    inp1 = &s1;
    Eigen::MatrixXd gradient_in = Eigen::MatrixXd::Random(1024, 32);
    Eigen::MatrixXd gradient_out_cpu = Eigen::MatrixXd::Zero(1000, 32);
    Eigen::MatrixXd gradient_out_gpu = Eigen::MatrixXd::Zero(1000, 32);
    Eigen::MatrixXd values = Eigen::MatrixXd::Random(1000, 32);
    SharedStorage shared_gradient_in = make_shared<Storage>(gradient_in);
    SharedStorage shared_gradient_out_cpu =
        make_shared<Storage>(gradient_out_cpu);
    SharedStorage shared_gradient_out_gpu =
        make_shared<Storage>(gradient_out_gpu);
    SharedStorage shared_values = make_shared<Storage>(values);
    vector<SharedStorage> grad_vec_cpu = {shared_gradient_out_cpu,
                                          shared_gradient_in};
    vector<SharedStorage> grad_vec_gpu = {shared_gradient_out_gpu,
                                          shared_gradient_in};
    int layer = 1;
    double cpuStart = cpuSecond();
    inp1->backward_cpu(layer, shared_values, grad_vec_cpu);
    double cpuEnd = cpuSecond() - cpuStart;
    layer = 1;
    double gpuStart = cpuSecond();
    inp1->backward_gpu(layer, shared_values, grad_vec_gpu);
    double gpuEnd = cpuSecond() - gpuStart;
    Eigen::MatrixXd diff = shared_gradient_out_cpu->return_data_const() -
                           shared_gradient_out_gpu->return_data_const();
    double out = diff.array().abs().maxCoeff();
    std::cout << "The CPU took " << cpuEnd << " and hte GPU took " << gpuEnd
              << std::endl;
    std::cout << "maximum difference: " << out << std::endl;
    REQUIRE(out < 1e-7);
    REQUIRE(gpuEnd < cpuEnd);
}

TEST_CASE("Dense forward equivalence", "[forward equivalence]") {
    srand((unsigned int)time(0));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    CHECK_CUBLAS(stat);
    Layer* inp1;
    Dense s1(1024, 1000, handle);
    inp1 = &s1;
    Eigen::MatrixXd in = Eigen::MatrixXd::Random(1000, 32);
    Eigen::MatrixXd out_cpu = Eigen::MatrixXd::Zero(1024, 32);
    Eigen::MatrixXd out_gpu = Eigen::MatrixXd::Zero(1024, 32);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out_cpu =
        std::make_shared<Storage>(out_cpu);
    std::shared_ptr<Storage> storage_out_gpu =
        std::make_shared<Storage>(out_gpu);
    double cpuStart = cpuSecond();
    inp1->forward_cpu(storage_in, storage_out_cpu);
    double cpuEnd = cpuSecond() - cpuStart;
    double gpuStart = cpuSecond();
    inp1->forward_gpu(storage_in, storage_out_gpu);
    double gpuEnd = cpuSecond() - gpuStart;
    Eigen::MatrixXd diff = storage_out_cpu->return_data_const() -
                           storage_out_gpu->return_data_const();
    double out = diff.array().abs().maxCoeff();
    std::cout << "The CPU took " << cpuEnd << " and hte GPU took " << gpuEnd
              << std::endl;
    std::cout << "maximum difference: " << out << std::endl;
    REQUIRE(out < 1e-7);
}
