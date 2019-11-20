#include "../../include/layer/dropout.h"
#include <curand.h>
#include <curand_kernel.h>
//#include <ctime>
#include <chrono>
#include <iostream>
#include "../../include/cuda_math.h"
#include "../../include/math.h"
#include <cuda.h>
#include <curand.h>

Dropout::Dropout(dtype prob)
    : assistance_parameters() {
    initialize_random();
    initialize_probability(prob);
}

Dropout::~Dropout() {CHECK_CURAND(curandDestroyGenerator(gen));}

void Dropout::initialize_random() {
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
}

void Dropout::initialize_probability(const dtype& prob) {
    Matrix ones = Matrix::Constant(1, 1, prob);
    assistance_parameters.push_back(std::make_shared<Storage>(ones));
}

void Dropout::forward_cpu(const SharedStorage& in, SharedStorage& out) {
    int rows = in->get_rows();
    int cols = in->get_cols();
    Matrix bern = Matrix::Random(rows, cols);
    Matrix& tmp = out->return_data();
    bern = bern.array() * 0.5 + 0.5;
    const dtype prob = assistance_parameters[0]->return_data_const()(0, 0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; j++) {
            bern(i, j) = (bern(i, j) < prob) ? 1. / prob : 0.;
        }
    }
    tmp = (in->return_data_const().array() * bern.array()).matrix();
}

void Dropout::forward_gpu(const SharedStorage& in, SharedStorage& out) {
    int rows = in->get_rows();
    int cols = in->get_cols();
    curandGenerateUniform(gen, out->gpu_pointer(), rows * cols);
    my_cuda_dropout(in, assistance_parameters[0], out);
}

void Dropout::backward_gpu(const SharedStorage&, const SharedStorage&,
                           SharedStorage&){};
void Dropout::backward_cpu(const SharedStorage&, const SharedStorage&,
                           SharedStorage&){};
void Dropout::clear_gradients_cpu(){};
void Dropout::clear_gradients_gpu(){};
