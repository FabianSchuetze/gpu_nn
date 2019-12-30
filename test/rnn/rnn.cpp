#include <stdlib.h>
#include <memory>
//#include <type_traits>
#include "../../include/neural_network.h"
#include "../../include/metrics/char_rnn.hpp"
#include "../../include/metrics/metric.hpp"
//#include "../../include/utils/normalization.hpp"
//#include "../../third_party/cifar10/include/cifar/get_data.h"
#include <iostream>
#include "../../include/utils/io.h"

typedef std::shared_ptr<Layer> s_Layer;
using std::make_shared;

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

int main(int argc, char** argv) {
    //std::ifstream t = std::ifstream("shakespeare.txt");
    std::ifstream t = std::ifstream("input.txt");
    std::vector<int> vec;
    char c;
    while (t >> std::noskipws >> c) {
        int val = (int)c;
        vec.push_back((int)val);
    }
    std::vector<int> new_vec(vec.size());
    std::map<int, int> char_to_ix;
    std::map<int, char> ix_to_char;
    int i = 0;
    for (int val : vec) {
        if (char_to_ix.count(val) == 0) {
            int len = char_to_ix.size();
            char_to_ix[val] = len;
            ix_to_char[len] = (char)val;
        }
        new_vec[i++] = char_to_ix[val];
    }
    Matrix input(Matrix::Zero(new_vec.size() -1, char_to_ix.size()));
    Matrix output(Matrix::Zero(new_vec.size() -1, char_to_ix.size()));
    int in = new_vec[0];
    for (size_t i = 0; i < new_vec.size() - 1; ++i) {
        int out = new_vec[i+1];
        input(i, in) = 1.0f;
        output(i, out) = 1.0f;
        in = out;
    }
    Init* init = new Glorot();
    s_Layer l1 = make_shared<Input>(Features(input.cols()));
    s_Layer rnn1 = make_shared<LSTM>(Features(128), l1, init);
    s_Layer rnn2 = make_shared<LSTM>(Features(128), rnn1, init);
    s_Layer d1 = make_shared<Dense>(Features(input.cols()), rnn2,init);
    s_Layer s1 = make_shared<Softmax>(d1);
    std::shared_ptr<Loss> loss =
        std::make_shared<CrossEntropy>(CrossEntropy("CPU"));
    NeuralNetwork n1(s1, loss, "CPU");
    std::vector<Metric*> test_func(0);
    Metric* val = new CharRNN(500, &n1, ix_to_char, 0.5);
    test_func.push_back(val);
    std::shared_ptr<GradientDescent> sgd = std::make_shared<AdaGrad>(
        LearningRate(0.002*100), DecayRate(0.95),
        WeightDecay(0), LearingRateDecay(10, 0.95));
    n1.train(input, output, sgd, Epochs(1000), Patience(1000), BatchSize(100),
            test_func, DebugInfo("", ""), Shuffle(false));
}
