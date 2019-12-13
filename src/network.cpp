#include "../include/network.h"
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <thread>
#include "../include/layer/im2col_layer.h"
#include "../include/loss/cross_entropy.h"
using std::shared_ptr;
using std::vector;

// void print_Matrix_to_stdout2(const Matrix& val, std::string loc) {
// int rows(val.rows()), cols(val.cols());
// std::ofstream myfile(loc);
// myfile << "dimensions: rows, cols: " << rows << ", " << cols << std::endl;
// myfile << std::fixed;
// myfile << std::setprecision(2);
// for (int row = 0; row < rows; ++row) {
// myfile << val(row, 0);
// for (int col = 1; col < cols; ++col) {
// myfile << ", " << val(row, col);
//}
// myfile << std::endl;
//}
//}
NeuralNetwork::NeuralNetwork(const std::shared_ptr<Layer>& last_layer,
                             std::shared_ptr<Loss>& _loss)
    : layers(), loss(_loss) {
    construct_layers(last_layer);
    fun_forward = &NeuralNetwork::forward_gpu;
    fun_backward = &NeuralNetwork::backward_gpu;
    fun_update = &NeuralNetwork::update_weights_gpu;
    print_network();
};

NeuralNetwork::NeuralNetwork(const std::shared_ptr<Layer>& last_layer,
                             std::shared_ptr<Loss>& _loss,
                             const std::string& device)
    : layers(), loss(_loss) {
    construct_layers(last_layer);
    if (device == "GPU") {
        fun_forward = &NeuralNetwork::forward_gpu;
        fun_backward = &NeuralNetwork::backward_gpu;
        fun_update = &NeuralNetwork::update_weights_gpu;
    } else {
        fun_forward = &NeuralNetwork::forward_cpu;
        fun_backward = &NeuralNetwork::backward_cpu;
        fun_update = &NeuralNetwork::update_weights_cpu;
    }
    print_network();
};

void NeuralNetwork::print_network() {
    int i = 0;
    std::vector<int> input_dim;
    for (const std::shared_ptr<Layer> layer : layers) {
        input_dim = layer->output_dimension();
        std::stringstream ss;
        ss << "Layer " << i << ": " << layer->name() << ", output size: ";
        std::copy(input_dim.begin(), input_dim.end(),
                  std::ostream_iterator<int>(ss, " "));
        std::cout << ss.str() << std::endl;
        i++;
    }
}

void NeuralNetwork::insert_cnn_layer(const std::shared_ptr<Layer>& layer) {
    std::shared_ptr<Convolution> derived =
        std::dynamic_pointer_cast<Convolution>(layer);
    std::shared_ptr<Layer> im2col = std::make_shared<Im2ColLayer>(derived);
    // std::shared_ptr<Layer> tmp = layer->_previous;
    layer->_previous = im2col;
    layers.push_front(layer);
    layers.push_front(im2col);
}

void NeuralNetwork::construct_layers(std::shared_ptr<Layer> curr) {
    while (curr->previous()) {
        if (curr->name() == "Convolution") {
            insert_cnn_layer(curr);
        } else if (curr->name() == "Im2ColLayer") {
            ;
        } else
            layers.push_front(curr);
        std::shared_ptr<Layer> tmp = curr->previous();
        curr.swap(tmp);
    }
    if (curr->name() == "Input")
        layers.push_front(curr);
    else {
        std::stringstream ss;
        ss << "Cannot recognize the layer name " << curr->name() << " in:\n"
           << __PRETTY_FUNCTION__ << "\ncalled from " << __FILE__ << " at "
           << __LINE__;
        throw std::invalid_argument(ss.str());
    }
}

int NeuralNetwork::convert_output_dimension(const shared_ptr<Layer>& layer) {
    int i = 1;
    for (int shape : layer->output_dimension()) i *= shape;
    return i;
}

void NeuralNetwork::allocate_storage(int obs, std::vector<SharedStorage>& inp,
                                     const std::shared_ptr<Layer>& layer) {
    if (layer->name() == "Im2ColLayer") {
        obs *= layer->input_dimension();
    }
    int out_dim = convert_output_dimension(layer);
    inp.push_back(std::make_shared<Storage>(Matrix::Zero(out_dim, obs)));
}

vector<SharedStorage> NeuralNetwork::allocate_forward(int obs) {
    vector<SharedStorage> vals;
    for (shared_ptr<Layer> layer : layers) {
        allocate_storage(obs, vals, layer);
    }
    return vals;
}

vector<SharedStorage> NeuralNetwork::allocate_backward(int obs) {
    vector<SharedStorage> vals;
    std::deque<shared_ptr<Layer>>::iterator layer = layers.begin();
    std::deque<shared_ptr<Layer>>::iterator end = layers.end();
    --end;
    while (layer != end) {
        allocate_storage(obs, vals, *layer);
        ++layer;
    }
    return vals;
}

void NeuralNetwork::fill_hiddens(vector<SharedStorage>& values,
                                 const Matrix& features) {
    values[0]->update_cpu_data(features);
}

void NeuralNetwork::forward(vector<SharedStorage>& values,
                            const std::string& type) {
    (this->*fun_forward)(values, type);
}

void NeuralNetwork::forward_gpu(vector<SharedStorage>& values,
                                const std::string& type) {
    int i = 0;
    std::deque<shared_ptr<Layer>>::iterator layer = layers.begin();
    ++layer;
    while (layer != layers.end()) {
        (*layer)->forward_gpu(values[i], values[i + 1], type);
        i++;
        ++layer;
    }
}

void NeuralNetwork::forward_cpu(vector<SharedStorage>& values,
                                const std::string& type) {
    int i = 0;
    std::deque<shared_ptr<Layer>>::iterator layer = layers.begin();
    ++layer;
    while (layer != layers.end()) {
        (*layer)->forward_gpu(values[i], values[i + 1], type);
        i++;
        ++layer;
    }
}

void NeuralNetwork::get_new_predict_sample(const vector<int>& samples,
                                           const Matrix& all, Matrix& subset) {
    subset = all(samples, Eigen::all).transpose();
}

vector<int> NeuralNetwork::predict_sample(int& iter, int total) {
    int at_least_remaining = std::min(32, total - iter);
    vector<int> samples(at_least_remaining);
    for (size_t i = 0; i < samples.size(); i++) samples[i] = iter++;
    return samples;
}

void NeuralNetwork::producer_predict(
    const Matrix& input, threadsafe_queue<vector<SharedStorage>>* pred_queue) {
    int iter = 0;
    int total = input.rows();
    Matrix x;
    vector<SharedStorage> vals;
    int batch_size(0);
    while (iter < total) {
        if (pred_queue->size() < 5) {
            vector<int> samples = predict_sample(iter, total);
            get_new_predict_sample(samples, input, x);
            SharedStorage inp = std::make_shared<Storage>(x);
            if (inp->get_cols() != batch_size) {
                vals = allocate_forward(x.cols());
                batch_size = inp->get_cols();
            }
            vals[0] = inp;
            pred_queue->push(vals);
        }
    }
}

void NeuralNetwork::consumer_predict(
    SharedStorage& target,
    threadsafe_queue<vector<SharedStorage>>* pred_queue) {
    int iter = 0;
    int total = target->get_cols();
    while (iter < total) {
        std::shared_ptr<vector<SharedStorage>> out = pred_queue->wait_and_pop();
        unsigned int start_position = iter * out->back()->get_rows();
        forward(*out, "predict");
        unsigned int obs = (*out)[0]->get_cols();
        target->update_gpu_data(out->back()->gpu_pointer_const(),
                                start_position, obs * out->back()->get_rows());
        iter += obs;
    }
}

void NeuralNetwork::predict(const Matrix& input, SharedStorage& SharedTarget) {
    threadsafe_queue<vector<SharedStorage>> pred_queue;
    threadsafe_queue<vector<SharedStorage>>* ppred_queue = &pred_queue;
    std::thread produce([&]() { producer_predict(input, ppred_queue); });
    std::thread consume([&]() { consumer_predict(SharedTarget, ppred_queue); });
    produce.join();
    consume.join();
}

Matrix NeuralNetwork::predict(const Matrix& input) {
    // THIS IS A HUGE BUG ! IT MUST BE DEPENDTENT ON THE TARGET!!! I NEED TO
    // FIX THE LAYERS BUSINESS!!!
    Matrix output = Matrix::Zero(10, input.rows());
    SharedStorage SharedTarget = std::make_shared<Storage>(output);
    predict(input, SharedTarget);
    return SharedTarget->return_data_const().transpose();
}
