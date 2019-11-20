#include "../../include/layer/dropout.h"
//#include <chrono>
#include <curand.h>
//#include <functional>
#include <iostream>
#include <stdexcept>
//#include <thread>
#include "../../include/cuda_math.h"
#include "../../include/math.h"

Dropout::Dropout(dtype prob) : probability(prob) {
    initialize_random();
    initialize_masking();
}

void Dropout::initialize_masking() {
    masking = std::make_shared<Storage>();
}

Dropout::~Dropout() { CHECK_CURAND(curandDestroyGenerator(gen)); }

void Dropout::initialize_random() {
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
}

void Dropout::check_masking(const SharedStorage& in) {
    if (!same_size(in, masking)) {
        masking = std::make_shared<Storage>(Matrix::Zero(in->get_rows(),
                    in->get_cols()));
    }
}

void Dropout::forward_cpu(const SharedStorage& in, SharedStorage& out,
                          const std::string& type) {
    if (type == "predict") {
        out->update_cpu_data(in->return_data_const());
        return;
    }
    check_masking(in);
    int rows = in->get_rows();
    int cols = in->get_cols();
    Matrix bern = Matrix::Random(rows, cols);
    bern = bern.array() * 0.5 + 0.5;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; j++) {
            masking->return_data()(i, j) =
                (bern(i, j) < probability) ? 1. / probability : 0.;
        }
    }
    out->return_data() =
        (in->return_data_const().array() * masking->return_data_const().array())
            .matrix();
}

void Dropout::forward_gpu(const SharedStorage& in, SharedStorage& out,
                          const std::string& type) {
    if (type == "predict") {
        out->update_gpu_data(in->gpu_pointer_const());
        return;
    }
    check_masking(in);
    int rows = in->get_rows();
    int cols = in->get_cols();
    curandGenerateUniform(gen, masking->gpu_pointer(), rows * cols);
    my_cuda_masking(probability, masking);
    my_mult_elementwise(in, masking, out);
}

void Dropout::check_backward() {
    if (!masking->is_set()) {
        std::string m("Was forward called first?, in\n");
        throw std::runtime_error(m + __PRETTY_FUNCTION__);
    }
}

void Dropout::backward_gpu(const SharedStorage& values,
                           const SharedStorage& grad_in,
                           SharedStorage& grad_out){
    check_backward();
    my_mult_elementwise(grad_in, masking, grad_out);
};

void Dropout::backward_cpu(const SharedStorage& values,
                           const SharedStorage& grad_in,
                           SharedStorage& grad_out) {
    check_backward();
    grad_out->return_data() = (masking->return_data_const().array() *
                               grad_in->return_data_const().array())
                                  .matrix();
}
void Dropout::clear_gradients_cpu(){};
void Dropout::clear_gradients_gpu(){};
