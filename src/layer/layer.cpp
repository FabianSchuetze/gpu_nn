#include "../../include/layer/layer.h"
typedef std::vector<std::shared_ptr<Storage>> VecSharedStorage;
int Layer::input_dimension() { return 0; }
int Layer::input_dimension() const { return 0; }
int Layer::output_dimension() { return 0; }
int Layer::output_dimension() const { return 0; }
void Layer::forward_gpu(const SharedStorage&, SharedStorage&) { ; };
void Layer::forward_cpu(const SharedStorage&, SharedStorage&) { ; };
void Layer::backward_gpu(const SharedStorage&, const SharedStorage&,
                         SharedStorage&) {
    ;
};
void Layer::backward_cpu(const SharedStorage&, const SharedStorage&,
                         SharedStorage&) {
    ;
};
VecSharedStorage Layer::return_parameters() { return parameters; };
VecSharedStorage Layer::return_gradients() { return gradients; };
VecSharedStorage Layer::return_parameters() const { return parameters; };
VecSharedStorage Layer::return_gradients() const { return gradients; };
void Layer::clear_gradients_cpu() { ; };
void Layer::clear_gradients_gpu() { ; };
