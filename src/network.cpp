#include "../include/network.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include "../include/loss/cross_entropy.h"
using std::vector;

//void print_Matrix_to_stdout2(const Matrix& val, std::string loc) {
    //int rows(val.rows()), cols(val.cols());
    //std::ofstream myfile(loc);
    //myfile << "dimensions: rows, cols: " << rows << ", " << cols << std::endl;
    //myfile << std::fixed;
    //myfile << std::setprecision(2);
    //for (int row = 0; row < rows; ++row) {
        //myfile << val(row, 0);
        //for (int col = 1; col < cols; ++col) {
            //myfile << ", " << val(row, col);
        //}
        //myfile << std::endl;
    //}
//}
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
        layers[layer_idx]->forward_gpu(values[i], values[i + 1], type);
        i++;
    }
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
        //std::cout << "consumer size: " << pred_queue->size() << std::endl;
        unsigned int start_position = iter * 10;
        std::shared_ptr<vector<SharedStorage>> out = pred_queue->wait_and_pop();
        forward(*out, "predict");
        unsigned int obs = (*out)[0]->get_cols();
        target->update_gpu_data(out->back()->gpu_pointer_const(),
                                start_position, obs * 10);
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
    Matrix output = Matrix::Zero(10, input.rows());
    SharedStorage SharedTarget = std::make_shared<Storage>(output);
    predict(input, SharedTarget);
    return SharedTarget->return_data_const().transpose();
}

//Matrix NeuralNetwork::predict(const Matrix& input) {
    //std::cout << "inside predict\n";
    //int iter = 0;
    //const int total = input.rows();
    //Matrix output = Matrix::Zero(10, input.rows());
    //SharedStorage SharedTarget = std::make_shared<Storage>(output);
    //Matrix x;
    //while (iter < total) {
        //unsigned int start_position = iter * 10;
        //vector<int> samples = predict_sample(iter, total);
        //get_new_predict_sample(samples, input, x);
        //unsigned int len = samples.size() * 10;
        //SharedStorage inp = std::make_shared<Storage>(x);
        //vector<SharedStorage> vals = allocate_forward(x.cols());
        //vals[0] = inp;
        //forward(vals, "predict");
        //SharedTarget->update_gpu_data(vals.back()->gpu_pointer_const(),
                                      //start_position, len);
    //}
    //std::cout << "leaving predict\n";
    //return SharedTarget->return_data_const();
    //// return output;
//}
