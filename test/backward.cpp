#include <cuda_runtime.h>
#include <eigen-git-mirror/Eigen/Core>
#include <sys/time.h>
#include <eigen-git-mirror/Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>
#include "../include/common.h"
#include "../include/layer/dense.h"
#include "../include/layer/layer.h"
#include "../include/math.h"
#include "../include/storage.h"
using Eigen::MatrixXd;
using std::make_shared;
using std::shared_ptr;
using std::vector;
int main() {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    int incoming = 12;
    int outgoing = 10;
    int obs = 6;
    MatrixXd gradient = MatrixXd::Ones(outgoing, obs);
    MatrixXd gradient_out = MatrixXd::Ones(incoming, obs);
    MatrixXd values = MatrixXd::Random(incoming, obs);
    // Eigen::MatrixXd Vec = Eigen::MatrixXd::Random(incoming_rows, 1);
    MatrixXd weight = MatrixXd::Random(outgoing, incoming);
    MatrixXd grad_bias = gradient.rowwise().sum();
    MatrixXd grad_weight = gradient * values.transpose();
    MatrixXd gradient_out_test = weight.transpose() * gradient;
    std::cout << std::fixed;
    std::cout << std::setprecision(2);
    std::cout << "incoming gradient\n" << gradient << std::endl;
    std::cout << " gradient bias\n" << grad_bias << std::endl;
    std::cout << "gradient weight\n" << grad_weight << std::endl;
    std::cout << "outgoing gradient\n" << gradient_out_test << std::endl;
    // Eigen::MatrixXd C2 = Eigen::MatrixXd::Zero(outgoing_rows2, incoming_obs);
    //shared_ptr<Storage> storage_grad = make_shared<Storage>(gradient);
    //shared_ptr<Storage> storage_grad_out = make_shared<Storage>(gradient_out);
    //shared_ptr<Storage> storage_vals = make_shared<Storage>(values);
    //vector<shared_ptr<Storage>> vec(2);
    //vector<shared_ptr<Storage>> vals_vec(1);
    //vec[1] = storage_grad;
    //vec[0] = storage_grad_out;
    //vals_vec[0] = storage_vals;
    // std::shared_ptr<Storage> storageC2 = std::make_shared<Storage>(C2);
    ////std::cout << std::fixed;
    ////std::cout << std::setprecision(10);
    ////std::cout << storageC->return_data_const() << std::endl;
    //Layer* inp1;
    // Layer* inp2;
    //Dense d1(outgoing, incoming, handle);
    // Dense d2(outgoing_rows2, outgoing_rows, handle);
    //inp1 = &d1;
    //inp1->backward_gpu(0, vals_vec, vec);
    //std::cout << "gpu grad bias\n"
              //<< inp1->return_gradients()[1]->return_data_const() << std::endl;
    // inp2 = &d2;
    // double beg = cpuSecond();
    // inp1->forward_gpu(storageA, storageC1);
    // inp2->forward_gpu(storageC1, storageC2);
    // double end = cpuSecond()- beg;
    // std::cout << end << std::endl;
    ////std::cout << storageC1->return_data_const() << std::endl;
    // double beg2 = cpuSecond();
    // Eigen::MatrixXd tmp = d1.return_parameters()[0]->return_data_const() * A;
    // Eigen::MatrixXd res1 = tmp.colwise() + Eigen::VectorXd(
    // inp1->return_parameters()[1]->return_data_const());
    // Eigen::MatrixXd tmp2 = d2.return_parameters()[0]->return_data_const() *
    // storageC1->return_data_const();
    // Eigen::MatrixXd res2 = tmp2.colwise() + Eigen::VectorXd(
    // inp2->return_parameters()[1]->return_data_const());
    // double end2 = cpuSecond()- beg2;
    // std::cout << end2 << std::endl;
    // Eigen::MatrixXd diff = res2 - storageC2->return_data_const();
    // std::cout <<diff.array().abs().maxCoeff() << std::endl;
}
