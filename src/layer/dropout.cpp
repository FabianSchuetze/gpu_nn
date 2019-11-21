#include "../../include/layer/dropout.h"
#include <curand.h>
#include <iostream>
#include <stdexcept>
#include "../../include/cuda_math.h"
#include "../../include/math.h"
#include <random>
#include <iomanip>

Dropout::Dropout(dtype prob) : Layer("Dropout"), probability(prob) {
    initialize_random();
    initialize_masking();
    //_name = "Dropout";
    //std::random_device rd;
    //gen_host = std::mt19937(rd());
    //gen_host.seed(0);
    dis = std::uniform_real_distribution<float>(0.0, 1.0);
}

void Dropout::initialize_masking() {
    masking = std::make_shared<Storage>();
}

Dropout::~Dropout() { CHECK_CURAND(curandDestroyGenerator(gen_device)); }

void Dropout::initialize_random() {
    std::random_device rd;
    gen_host = std::mt19937(rd());
    gen_host.seed(0);
    dis = std::uniform_real_distribution<float>(0.0, 1.0);
    CHECK_CURAND(curandCreateGenerator(&gen_device, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen_device, 1234ULL));
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
    Matrix bern = Matrix::NullaryExpr(rows,cols,[&](){return dis(gen_host);});
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
    };
    srand((unsigned int) 0);
    check_masking(in);
    int rows = in->get_rows();
    int cols = in->get_cols();
    curandGenerateUniform(gen_device, masking->gpu_pointer(), rows * cols);
    my_cuda_masking(probability, masking);
    my_mult_elementwise(in, masking, out);
}

void Dropout::check_backward() {
    if (!masking->is_set()) {
        std::string m("Was forward called first?, in\n");
        throw std::runtime_error(m + __PRETTY_FUNCTION__);
    }
}

void Dropout::backward_gpu(const SharedStorage&, const SharedStorage& grad_in,
                           SharedStorage& grad_out){
    check_backward();
    my_mult_elementwise(grad_in, masking, grad_out);
};

void Dropout::backward_cpu(const SharedStorage&, const SharedStorage& grad_in,
                           SharedStorage& grad_out) {
    check_backward();
    grad_out->return_data() = (masking->return_data_const().array() *
                               grad_in->return_data_const().array())
                                  .matrix();
}

//void Dropout::clear_gradients_cpu(){};
//void Dropout::clear_gradients_gpu(){};
