#include "../include/debug_info.hpp"
#include <algorithm>
#include <memory>
#include <ostream>

DebugInfo::DebugInfo(std::string fwd, std::string bwd)
    : _ffw_stream(fwd), _bwd_stream(bwd) {
    if ((fwd.size() > 0) and (bwd.size() > 0)) {
        _is_set = true;
    }
}

bool DebugInfo::is_set() { return _is_set; }

void DebugInfo::backward_debug_info(
    const std::vector<std::shared_ptr<Storage>>& grad,
    const std::deque<std::shared_ptr<Layer>>& layers, int batch_size) {
    _bwd_stream << grad[0]->return_data_const().mean();
    for (size_t i = 1; i < grad.size(); ++i) {
        _bwd_stream << " ";
        _bwd_stream << grad[i]->return_data_const().mean();
        _bwd_stream << " ";
        _bwd_stream << grad[i]->return_data_const().lpNorm<1>() / batch_size;
        for (const SharedStorage& para : layers[i]->return_gradients()) {
            _bwd_stream << " ";
            _bwd_stream << para->return_data_const().mean();
            _bwd_stream << " ";
            _bwd_stream << para->return_data_const().lpNorm<1>();
        }
    }
    _bwd_stream << "\n";
}

void DebugInfo::forward_debug_info(
    const std::vector<std::shared_ptr<Storage>>& values,
    const std::deque<std::shared_ptr<Layer>>& layers, int batch_size) {
    _ffw_stream << values[0]->return_data_const().mean();
    for (size_t i = 1; i < values.size(); ++i) {
        _ffw_stream << " ";
        _ffw_stream << values[i]->return_data_const().mean();
        _ffw_stream << " ";
        _ffw_stream << values[i]->return_data_const().lpNorm<1>() / batch_size;
        for (const SharedStorage& para : layers[i]->return_parameters()) {
            _ffw_stream << " ";
            _ffw_stream << para->return_data_const().mean();
            _ffw_stream << " ";
            _ffw_stream << para->return_data_const().lpNorm<1>();
        }
    }
    _ffw_stream << "\n";
}

void DebugInfo::print_layers(const std::deque<std::shared_ptr<Layer>>& layers) {
    _print_layers(layers, _ffw_stream);
    _print_layers(layers, _bwd_stream);
}

void DebugInfo::_print_layers(const std::deque<std::shared_ptr<Layer>>& layers,
                              std::ostream& stream) {
    stream << layers[0]->name();
    for (size_t i = 1; i < layers.size(); ++i) {
        stream << " ";
        stream << layers[i]->name() << "_mean_";
        stream << " ";
        stream << layers[i]->name() << "_l1_";
        for (size_t j = 0; j < layers[i]->return_parameters().size(); ++j) {
            stream << " ";
            stream << layers[i]->name() << "_param_mena" << j;
            stream << " ";
            stream << layers[i]->name() << "_param_l1" << j;
        }
    }
    stream << "\n";
}
