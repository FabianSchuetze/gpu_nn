#include "../include/network.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include "../include/loss/cross_entropy.h"
using std::vector;

void print_Matrix_to_stdout2(const Matrix& val, std::string loc) {
    int rows(val.rows()), cols(val.cols());
    std::ofstream myfile(loc);
    myfile << "dimensions: rows, cols: " << rows << ", " << cols << std::endl;
    myfile << std::fixed;
    myfile << std::setprecision(2);
    for (int row = 0; row < rows; ++row) {
        myfile << val(row, 0);
        for (int col = 1; col < cols; ++col) {
            myfile << ", " << val(row, col);
        }
        myfile << std::endl;
    }
}
NeuralNetwork::NeuralNetwork(vector<Layer*> _layers,
                             std::shared_ptr<Loss> _loss)
    : layers(_layers), loss(_loss) {
    fun_forward = &NeuralNetwork::forward_gpu;
    fun_backward = &NeuralNetwork::backward_gpu;
    fun_update = &NeuralNetwork::update_weights_gpu;
};

NeuralNetwork::NeuralNetwork(vector<Layer*> _layers,
                             std::shared_ptr<Loss> _loss,
                             const std::string& device)
    : layers(_layers), loss(_loss) {
    if (device == "GPU") {
        fun_forward = &NeuralNetwork::forward_gpu;
        fun_backward = &NeuralNetwork::backward_gpu;
        fun_update = &NeuralNetwork::update_weights_gpu;
    } else {
        fun_forward = &NeuralNetwork::forward_cpu;
        fun_backward = &NeuralNetwork::backward_cpu;
        fun_update = &NeuralNetwork::update_weights_cpu;
    }
};

void NeuralNetwork::allocate_storage(int obs, int& out_dim,
                                     std::vector<SharedStorage>& inp,
                                     const Layer* layer) {
    if (layer->name() == "Dense") {
        if (layer->input_dimension() != out_dim) {
            int input_dim = layer->input_dimension();
            std::stringstream ss;
            ss << "Dimension do not fit, in:\n"
               << __PRETTY_FUNCTION__ << "\n Previous output:" << out_dim
               << " expected input " << input_dim << "\ncalled from "
               << __FILE__ << " at " << __LINE__;
            throw std::invalid_argument(ss.str());
        }
        out_dim = layer->output_dimension();
    } else if (layer->name() == "Activation")
        ;
    else if (layer->name() == "BatchNorm") {
        ;
    } else if (layer->name() == "Convolution") {
        out_dim = layer->output_dimension();
    } else if (layer->name() == "Dropout") {
        ;
    } else if (layer->name() == "Im2ColLayer") {
        out_dim = layer->output_dimension();
        obs *= layer->n_cols();
    } else if (layer->name() == "Input") {
        out_dim = layer->output_dimension();
    } else {
        std::stringstream ss;
        ss << "Cannot figure out name, in:\n"
           << __PRETTY_FUNCTION__ << "\ncalled from " << __FILE__ << " at "
           << __LINE__;
        throw std::invalid_argument(ss.str());
    }
    inp.push_back(std::make_shared<Storage>(Matrix::Zero(out_dim, obs)));
}

vector<SharedStorage> NeuralNetwork::allocate_forward(int obs) {
    vector<SharedStorage> vals;
    int out_dim(0);
    for (const Layer* layer : layers) {
        allocate_storage(obs, out_dim, vals, layer);
    }
    return vals;
}

vector<SharedStorage> NeuralNetwork::allocate_backward(int obs) {
    vector<SharedStorage> vals;
    int out_dim(0);
    for (size_t i = 0; i < layers.size() - 1; i++) {
        allocate_storage(obs, out_dim, vals, layers[i]);
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
    for (size_t layer_idx = 1; layer_idx < layers.size(); ++layer_idx) {
        // std::cout << "layer name " << layers[layer_idx]->name() << std::endl;
        layers[layer_idx]->forward_gpu(values[i], values[i + 1], type);
        i++;
    }
    // std::cout << "finisehd forward gpu\n" << std::endl;
}

void NeuralNetwork::forward_cpu(vector<SharedStorage>& values,
                                const std::string& type) {
    int i = 0;
    for (size_t layer_idx = 1; layer_idx < layers.size(); ++layer_idx) {
        layers[layer_idx]->forward_cpu(values[i], values[i + 1], type);
        i++;
    }
}

void NeuralNetwork::get_new_predict_sample(const vector<int>& samples,
                                           const Matrix& all, Matrix& subset) {
    subset = all(samples, Eigen::all).transpose();
}

vector<int> NeuralNetwork::predict_sample(int& iter, int total) {
    int at_least_remaining = std::min(train_args->batch_size(), total - iter);
    vector<int> samples(at_least_remaining);
    for (size_t i = 0; i < samples.size(); i++) samples[i] = iter++;
    return samples;
}

Matrix NeuralNetwork::predict(const Matrix& input) {
    int iter = 0;
    const int total = input.rows();
    Matrix output = Matrix::Zero(input.rows(), 10);
    Matrix x;
    while (iter < total) {
        vector<int> samples = predict_sample(iter, total);
        get_new_predict_sample(samples, input, x);
        SharedStorage inp = std::make_shared<Storage>(x);
        vector<SharedStorage> vals = allocate_forward(x.cols());
        vals[0] = inp;
        forward(vals, "predict");
        output(samples, Eigen::all) =
            vals.back()->return_data_const().transpose();
    }
    return output.transpose();
}
