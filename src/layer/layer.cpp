#include "../../include/layer/layer.h"

Layer::Layer(const std::string& s)
    : _name(s), _out_dim(1), parameters(), gradients(), _previous(NULL) {
    _out_dim[0] = 0;
}

Layer::Layer()
    : _name("Layer"), _out_dim(1), parameters(), gradients(), _previous(NULL) {
    _out_dim[0] = 0;
}

typedef std::vector<std::shared_ptr<Storage>> VecSharedStorage;

void Layer::forward_gpu(const SharedStorage&, SharedStorage&,
                        const std::string&) {
    ;
};
void Layer::forward_cpu(const SharedStorage&, SharedStorage&,
                        const std::string&) {
    ;
};
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

void Layer::initialize_output_dimension(
    const std::shared_ptr<Layer>& previous) {
    if (previous) {
        int i = 1;
        for (int shape : previous->output_dimension()) i *= shape;
        _out_dim[0] = i;
    }
}

void Layer::initialize_output_dimension() { _out_dim.push_back(0); }
