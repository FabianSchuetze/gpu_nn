#include "../include/network.h"
#include <memory>
#include "../include/loss/cross_entropy.h"
using std::vector;
NeuralNetwork::NeuralNetwork(vector<Layer*> _layers, const std::string& _loss)
    : layers(_layers) {
    create_loss(_loss);
};

void NeuralNetwork::create_loss(const std::string& s) {
    if (s == "Bernoulli")
        ;
    // loss = std::make_shared<Bernoulli>();
    else if (s == "MSE")
        ;
    // loss = std::make_shared<MSE>();
    else if (s == "Categorical_Crossentropy")
        loss = std::make_shared<CrossEntropy>();
    else {
        // string m("Only Bernoulli, MSE and Categorical_Crossentropy, in:\n");
        // throw std::invalid_argument(m + __PRETTY_FUNCTION__);
    }
}

vector<SharedStorage> NeuralNetwork::allocate_shared_storage(int obs) {
    vector<SharedStorage> vals;
    int out_dim(0);
    for (const Layer* layer : layers) {
        if (layer->name() == "Dense") {
            out_dim = layers[0]->output_dimension();
        } else if (layer->name() == "Activation")
            ;
        else if (layer->name() == "BatchNorm") {
            ;
        } else if (layer->name() == "Dropout") {
            ;
        } else if (layer->name() == "Input") {
            out_dim = layers[0]->output_dimension();
        } else {
            std::string m("Cannot figure out name, in:\n");
            throw std::invalid_argument(m + __PRETTY_FUNCTION__);
        }
        vals.push_back(std::make_shared<Storage>(Matrix(out_dim, obs)));
    }
    return vals;
}

void NeuralNetwork::fill_hiddens(vector<SharedStorage>& values,
                                 const Matrix& features) {
    values[0]->copy_cpu_data(features.transpose());
}

void NeuralNetwork::forward(vector<SharedStorage>& values) {
    int i = 0;
    for (Layer* layer : layers) {
        SharedStorage input = values[i];
        SharedStorage output = values[i+1];
        layer->forward_gpu(input, output);
        i++;
    }
}

Matrix NeuralNetwork::predict(const Matrix& input) {
    SharedStorage inp = std::make_shared<Storage>(input);
    vector<SharedStorage> vals = allocate_shared_storage(input.rows());
    fill_hiddens(vals, input);
    forward(vals);
    return vals[vals.size() - 1]->return_data().transpose();
}
