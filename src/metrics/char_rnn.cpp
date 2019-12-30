#include "../../include/metrics/char_rnn.hpp"
#include <iostream>
#include <random>
using std::vector;

CharRNN::CharRNN(int length, NeuralNetwork* nn,
                 const std::map<int, char>& ix_to_char)
    : Metric("CharRNN", nn), _length(length), _ix_to_char(ix_to_char) {
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
    int out_size = targets.cols();
    Matrix input = Matrix::Zero(1, features.cols());
    std::uniform_int_distribution<int> dist(0, features.rows() -1);
    int start = dist(gen);
    input(0, Eigen::all) = features(start, Eigen::all);
    vector<int> sequence = {start};
    for (int i = 0; i < _length; ++i) {
        Matrix res = _nn->predict(input, out_size);
        int sample = pick_value(res);
        input.setZero();
        input(0, sample) = 1;
        sequence.push_back(sample);
    }
    convert_to_stdout(sequence);
}
