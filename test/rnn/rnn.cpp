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


int main(int argc, char** argv) {
    //std::ifstream t = std::ifstream("ft.txt");
    std::ifstream t = std::ifstream("shakespeare.txt");
    std::cout << "here" << std::endl;
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
    Init* init = new Normal(0., 0.01);
    s_Layer l1 = make_shared<Input>(Features(input.cols()));
    s_Layer rnn1 = make_shared<LSTM>(Features(256), l1, init);
    s_Layer rnn2 = make_shared<LSTM>(Features(256), rnn1, init);
    s_Layer d1 = make_shared<Dense>(Features(input.cols()), rnn2,init);
    s_Layer s1 = make_shared<Softmax>(d1);
    std::shared_ptr<Loss> loss =
        std::make_shared<CrossEntropy>(CrossEntropy("CPU"));
    NeuralNetwork n1(s1, loss, "CPU");
    std::shared_ptr<GradientDescent> sgd = std::make_shared<Momentum>(
        LearningRate(0.01*32), MomentumRate(0.9));
    std::vector<Metric*> test_func(0);
    Metric* val = new CharRNN(200, &n1, ix_to_char);
    test_func.push_back(val);
    //std::shared_ptr<GradientDescent> sgd = std::make_shared<AdaGrad>(
        //LearningRate(0.01*32));
    n1.train(input, output, sgd, Epochs(1000), Patience(1000), BatchSize(32),
            test_func, DebugInfo("", ""), Shuffle(false));
            //DebugInfo("forwards_info", "backwards_info"));
     //Matrix predictions = n1.predict(x_test, 10);
     //n_missclassified(predictions, y_test);
}
