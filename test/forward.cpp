#include <cuda_runtime.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <eigen-git-mirror/Eigen/src/Core/util/ForwardDeclarations.h>
#include <sys/time.h>
#include <iostream>
#include <memory>
#include "../include/common.h"
#include "../include/layer/dense.h"
#include "../include/layer/softmax.h"
#include "../include/layer/layer.h"
#include "../include/math.h"
#include "../include/storage.h"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
int main() {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    int incoming_rows = 10;
    int outgoing_rows = 12;
    int outgoing_rows2 = 6;
    int incoming_obs = 5;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(incoming_rows, incoming_obs);
    //Eigen::MatrixXd Vec = Eigen::MatrixXd::Random(incoming_rows, 1);
    Eigen::MatrixXd C1 = Eigen::MatrixXd::Zero(outgoing_rows, incoming_obs);
    Eigen::MatrixXd C2 = Eigen::MatrixXd::Zero(outgoing_rows2, incoming_obs);
    Eigen::MatrixXd C3 = Eigen::MatrixXd::Zero(outgoing_rows2, incoming_obs);
    Eigen::MatrixXd C4 = Eigen::MatrixXd::Zero(outgoing_rows2, incoming_obs);
    std::shared_ptr<Storage> storageA = std::make_shared<Storage>(A);
    std::shared_ptr<Storage> storageC1 = std::make_shared<Storage>(C1);
    std::shared_ptr<Storage> storageC2 = std::make_shared<Storage>(C2);
    std::shared_ptr<Storage> storageC3 = std::make_shared<Storage>(C3);
    std::shared_ptr<Storage> storageC4 = std::make_shared<Storage>(C4);
    //std::cout << std::fixed;
    //std::cout << std::setprecision(10);
    //std::cout << storageC->return_data_const() << std::endl;
    Layer* inp1;
    Layer* inp2;
    Layer* inp3;
    Dense d1(outgoing_rows, incoming_rows, handle);
    Dense d2(outgoing_rows2, outgoing_rows, handle);
    Softmax s1(handle);
    inp1 = &d1;
    inp2 = &d2;
    inp3 = &s1;
    double beg = cpuSecond();
    inp1->forward_gpu(storageA, storageC1);
    inp2->forward_gpu(storageC1, storageC2);
    inp3->forward_gpu(storageC2, storageC3);
    inp3->forward_cpu(storageC2, storageC4);
    double end = cpuSecond()- beg;
    std::cout << end << std::endl;
    std::cout << storageC3->return_data_const() << std::endl;
    std::cout << "CPU CALC\n" << storageC4->return_data_const() << std::endl;
    double beg2 = cpuSecond();
    Eigen::MatrixXd tmp = d1.return_parameters()[0]->return_data_const() * A;
    Eigen::MatrixXd res1 = tmp.colwise() + Eigen::VectorXd(
            inp1->return_parameters()[1]->return_data_const());
    Eigen::MatrixXd tmp2 = d2.return_parameters()[0]->return_data_const() * 
        storageC1->return_data_const();
    Eigen::MatrixXd res2 = tmp2.colwise() + Eigen::VectorXd(
            inp2->return_parameters()[1]->return_data_const());
    double end2 = cpuSecond()- beg2;
    std::cout << end2 << std::endl;
    Eigen::MatrixXd diff = res2 - storageC2->return_data_const();
    std::cout <<diff.array().abs().maxCoeff() << std::endl;
}
