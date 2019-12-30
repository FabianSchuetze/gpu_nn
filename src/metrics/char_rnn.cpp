#include "../../include/metrics/char_rnn.hpp"
#include <iostream>
#include <random>
using std::vector;

CharRNN::CharRNN(int length, NeuralNetwork* nn,
                 const std::map<int, char>& ix_to_char, dtype _temp)
    : Metric("CharRNN", nn),
      _length(length),
      _ix_to_char(ix_to_char),
      temperature(_temp) {
    gen.seed(0);
};

void CharRNN::convert_to_stdout(const vector<int>& sequence) {
    std::stringstream s;
    for (int val : sequence) {
        s << _ix_to_char[val];
    }
    std::cout << s.str() << std::endl;
    std::cout << std::endl;
}
int CharRNN::pick_value(const Matrix& input) {
    std::uniform_real_distribution<dtype> dist(0.0, 1.0);
    dtype number = dist(gen);
    dtype start = input(0);
    for (int i = 0; i < input.cols(); ++i) {
        if (number < start) return i;
        start += input(i + 1);
    }
    return input.rows() - 1;
}

void CharRNN::validate(const Matrix& features, const Matrix& targets) {
    DebugInfo no_debugging("", "");
    std::uniform_int_distribution<int> dist(0, features.rows() - 1);
    int start = dist(gen);
    vector<int> sequence = {start};
    vector<SharedStorage> vals = _nn->allocate_forward(1);
    vals[0]->update_cpu_data(features(start, Eigen::all).transpose());
    const std::string type("predict");
    for (int i = 0; i < _length; ++i) {
        _nn->forward(vals, type, no_debugging);
        if (temperature < 1) {
            vals[vals.size() - 2]->return_data() /= temperature;
            _nn->layers[_nn->layers.size() - 1]->forward_cpu(
                vals[vals.size() - 2], vals[vals.size() - 1], "predict");
        }
        int sample = pick_value(vals.back()->return_data_const().transpose());
        vals[0]->return_data().setZero();
        vals[0]->return_data()(sample, 0) = 1;
        sequence.push_back(sample);
    }
    convert_to_stdout(sequence);
}
