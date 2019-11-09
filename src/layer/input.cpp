#include "../../include/layer/input.h"

Input::Input(int output_dimension)
    : Layer(), _output_dimension(output_dimension) {
    _name = "Input";
}

void Input::forward_gpu(const std::shared_ptr<Storage>&,
                        std::shared_ptr<Storage>&) {}
void Input::forward_cpu(const std::shared_ptr<Storage>&,
                        std::shared_ptr<Storage>&) {}
void Input::backward_gpu(const SharedStorage&, const SharedStorage&,
                         SharedStorage&) {};
void Input::backward_cpu(const SharedStorage&, const SharedStorage&,
                         SharedStorage&) {};
