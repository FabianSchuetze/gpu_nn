#include <eigen-git-mirror/Eigen/Core>
#define CATCH_CONFIG_MAIN
#include "../third_party/catch/catch.hpp"
#include "../include/common.h"
//#include "../include/layer/lstm.hpp"
//#include "../include/storage.h"
#include <sys/time.h>
#include <iostream>
#include "../include/neural_network.h"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
// int main() {
//TEST_CASE("NeuralNetwork cpu", "[cpu]") {
    //srand((unsigned int)0);
    //int outf = 10;
    //int inf = 12;
    //Init* init = new Glorot();
    //Layer* inp1 = new LSTM(Features(outf), Features(inf), init);
    //Matrix in = Matrix::Zero(inf, 5);
    //in(0, 0) = 1;
    //in(2, 1) = 1;
    //in(10, 2) = 1;
    //in(5, 3) = 1;
    //in(11, 4) = 1;
    //Matrix out = Matrix::Zero(outf, 5);
    //std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    //std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    //inp1->forward_cpu(storage_in, storage_out, "train");
    //REQUIRE(storage_out->return_data_const().cwiseAbs().minCoeff() > 1e-8);
//}

//TEST_CASE("NeuralNetwork gpu", "[gpu]") {
    //// int main() {
    //srand((unsigned int)0);
    //int outf = 10;
    //int inf = 12;
    //Init* init2 = new Glorot();
    //Layer* inp12 = new LSTM(Features(outf), Features(inf), init2);
    //Matrix in2 = Matrix::Zero(inf, 5);
    //in2(0, 0) = 1;
    //in2(2, 1) = 1;
    //in2(10, 2) = 1;
    //in2(5, 3) = 1;
    //in2(11, 4) = 1;
    //Matrix out2 = Matrix::Zero(outf, 5);
    //std::shared_ptr<Storage> storage_in2 = std::make_shared<Storage>(in2);
    //std::shared_ptr<Storage> storage_out2 = std::make_shared<Storage>(out2);
    //inp12->forward_gpu(storage_in2, storage_out2, "train");
    //REQUIRE(storage_out2->return_data_const().cwiseAbs().minCoeff() > 1e-8);
//}

//TEST_CASE("NeuralNetwork forward equivalance", "[forward]") {
    //// int main() {
    //srand((unsigned int)0);
    //int outf = 200;
    //int inf = 66;
    //std::random_device rd;
    //std::mt19937 gen(rd());
    //std::uniform_int_distribution<> dis(1, inf - 1);
    //Init* init2 = new Glorot();
    //Layer* inp1_cpu = new LSTM(Features(outf), Features(inf), init2);
    //Layer* inp1_gpu = new LSTM(Features(outf), Features(inf), init2);
    //Matrix in2 = Matrix::Zero(inf, 32);
    //for (int i = 0; i < 32; ++i) {
        //int rint = dis(gen);
        //in2(rint, i) = 1.0;
    //}
    //Matrix out_gpu = Matrix::Zero(outf, 32);
    //Matrix out_cpu = Matrix::Zero(outf, 32);
    //std::shared_ptr<Storage> storage_in2 = std::make_shared<Storage>(in2);
    //std::shared_ptr<Storage> storage_gpu = std::make_shared<Storage>(out_gpu);
    //std::shared_ptr<Storage> storage_cpu = std::make_shared<Storage>(out_cpu);
    //double gpuStart = cpuSecond();
    //inp1_gpu->forward_gpu(storage_in2, storage_gpu, "train");
    //double gpuEnd = cpuSecond() - gpuStart;
    //double cpuStart = cpuSecond();
    //inp1_cpu->forward_cpu(storage_in2, storage_cpu, "train");
    //double cpuEnd = cpuSecond() - cpuStart;
    //Matrix diff =
        //storage_cpu->return_data_const() - storage_gpu->return_data_const();
    //dtype out = diff.array().abs().maxCoeff();
    //dtype allowed = 1e-5;
    //std::cout << "The CPU took " << cpuEnd << " and hte GPU took " << gpuEnd
              //<< std::endl;
    //std::cout << "maximum difference: " << out << std::endl;
    //REQUIRE(out < allowed);
//}

 //int main() {
TEST_CASE("NeuralNetwork back cpu", "[back cpu]") {
    srand((unsigned int)0);
    int outf = 5;
    int inf = 4;
    int batches = 2;
    Init* init = new Glorot();
    Layer* inp1 = new LSTM(Features(outf), Features(inf), init);
    Matrix in = Matrix::Zero(inf, batches);
    in(0, 0) = 1;
    in(2, 1) = 1;
    Matrix out = Matrix::Zero(outf, batches);
    Matrix gin = Matrix::Random(outf, batches);
    Matrix gout = Matrix::Zero(inf, batches);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    std::shared_ptr<Storage> grad_in = std::make_shared<Storage>(gin);
    std::shared_ptr<Storage> grad_out_cpu = std::make_shared<Storage>(gout);
    inp1->forward_cpu(storage_in, storage_out, "train");
    inp1->backward_cpu(storage_in, grad_in, grad_out_cpu);
    std::cout << "grad out cpu\n" <<
        grad_out_cpu->return_data_const() << std::endl;
    REQUIRE(grad_out_cpu->return_data_const().cwiseAbs().minCoeff() > 1e-8);
}

TEST_CASE("NeuralNetwork back gpu", "[back gpu]") {
    srand((unsigned int)0);
    int batches = 2;
    int outf = 10;
    int inf = 12;
    Init* init2 = new Glorot();
    Layer* inp12 = new LSTM(Features(outf), Features(inf), init2);
    Matrix in2 = Matrix::Zero(inf, batches);
    in2(0, 0) = 1;
    in2(2, 1) = 1;
    Matrix out2 = Matrix::Zero(outf, batches);
    Matrix gin_gpu = Matrix::Random(outf, batches);
    Matrix gout_gpu = Matrix::Zero(inf, batches);
    std::shared_ptr<Storage> storage_in2 = std::make_shared<Storage>(in2);
    std::shared_ptr<Storage> storage_out2 = std::make_shared<Storage>(out2);
    std::shared_ptr<Storage> grad_in_gpu = std::make_shared<Storage>(gin_gpu);
    std::shared_ptr<Storage> grad_out_gpu = std::make_shared<Storage>(gout_gpu);
    inp12->forward_gpu(storage_in2, storage_out2, "train");
    inp12->backward_gpu(storage_in2, grad_in_gpu, grad_out_gpu);
    std::cout << "gpu\n" << grad_out_gpu->return_data_const() << std::endl;
    REQUIRE(grad_out_gpu->return_data_const().cwiseAbs().minCoeff() > 1e-8);
}

TEST_CASE("NeuralNetwork backward equivalance", "[backward]") {
    // int main() {
    srand((unsigned int)0);
    int outf = 200;
    int inf = 66;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, inf - 1);
    Init* init2 = new Glorot();
    Layer* inp1_cpu = new LSTM(Features(outf), Features(inf), init2);
    Layer* inp1_gpu = new LSTM(Features(outf), Features(inf), init2);
    Matrix in2 = Matrix::Zero(inf, 32);
    for (int i = 0; i < 32; ++i) {
        int rint = dis(gen);
        in2(rint, i) = 1.0;
    }
    Matrix grad_in = Matrix::Random(outf, 32);
    std::shared_ptr<Storage> storage_grad_in =
        std::make_shared<Storage>(grad_in);
    Matrix out_gpu = Matrix::Zero(outf, 32);
    Matrix out_cpu = Matrix::Zero(outf, 32);
    Matrix gout_cpu = Matrix::Zero(inf, 32);
    Matrix gout_gpu = Matrix::Zero(inf, 32);
    std::shared_ptr<Storage> storage_in2 = std::make_shared<Storage>(in2);
    std::shared_ptr<Storage> storage_gpu = std::make_shared<Storage>(out_gpu);
    std::shared_ptr<Storage> storage_cpu = std::make_shared<Storage>(out_cpu);
    std::shared_ptr<Storage> grad_out_cpu = std::make_shared<Storage>(gout_cpu);
    std::shared_ptr<Storage> grad_out_gpu = std::make_shared<Storage>(gout_gpu);
    inp1_gpu->forward_gpu(storage_in2, storage_gpu, "train");
    double gpuStart = cpuSecond();
    inp1_gpu->backward_gpu(storage_in2, storage_grad_in, grad_out_gpu);
    double gpuEnd = cpuSecond() - gpuStart;
    inp1_cpu->forward_cpu(storage_in2, storage_cpu, "train");
    double cpuStart = cpuSecond();
    inp1_cpu->backward_cpu(storage_in2, storage_grad_in, grad_out_cpu);
    double cpuEnd = cpuSecond() - cpuStart;
    Matrix diff =
        grad_out_cpu->return_data_const() - grad_out_gpu->return_data_const();
    dtype out = diff.array().abs().maxCoeff();
    dtype allowed = 1e-5;
    std::cout << "The CPU took " << cpuEnd << " and hte GPU took " << gpuEnd
              << std::endl;
    std::cout << "maximum difference: " << out << std::endl;
    REQUIRE(out < allowed);
}
