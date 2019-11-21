#include <stdlib.h>
#include <type_traits>
#include "../include/neural_network.h"
#include "../third_party/mnist/include/mnist/get_data.h"

int main(int argc, char** argv) {
    if ((argc != 2) and (argc != 5))
        throw std::invalid_argument("Must have one or four arguemnts");
    Mnist data = Mnist();
    srand((unsigned int)time(0));
    Layer* l1 = new Input(data.get_x_train().cols());
    Layer* l2 = new Dense(100, data.get_x_train().cols());
    Layer* l3 = new Relu;
    Layer* l4 = new Dropout(0.5);
    Layer* l5 = new Dense(10, 100);
    Layer* l6 = new Softmax;
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy(
                argv[1]));
    NeuralNetwork n1({l1, l2, l3, l4, l5, l6}, loss, argv[1]);
    std::shared_ptr<GradientDescent> sgd =
        std::make_shared<StochasticGradientDescent>(0.001);
    if (argc == 5) {
        Epochs epoch(strtol(argv[2], NULL, 10));
        Patience patience(strtol(argv[3], NULL, 10));
        BatchSize batch_size(strtol(argv[4], NULL, 10));
        n1.train(data.get_x_train(), data.get_y_train(), sgd, epoch, patience,
                 batch_size);
    } else
        n1.train(data.get_x_train(), data.get_y_train(), sgd, Epochs(10),
                 Patience(10), BatchSize(32));
}
