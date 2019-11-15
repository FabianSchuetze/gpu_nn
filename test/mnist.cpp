#include "../include/neural_network.h"
#include "../third_party/mnist/include/mnist/get_data.h"
#include <type_traits>
#include <stdlib.h>

int main(int argc, char** argv) {
    if ((argc != 1) and (argc != 4))
        throw std::invalid_argument("Must have no or three arguemnts");
    Mnist data = Mnist();
    srand((unsigned int)time(0));
    Layer* l1;
    Layer* l2;
    Layer* l3;
    Layer* l4;
    Layer* l5;
    Input i1(data.get_x_train().cols());
    Dense d1(100, data.get_x_train().cols());
    Relu r1;
    Dense d2(10, 100);
    Softmax s1;
    l1 = &i1;
    l2 = &d1;
    l3 = &r1;
    l4 = &d2;
    l5 = &s1;
    std::vector<Layer*> vec = {l1, l2, l3, l4, l5};
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    NeuralNetwork n1(vec, loss, "GPU");
    std::shared_ptr<GradientDescent> sgd =
        std::make_shared<StochasticGradientDescent>(0.001);
    if (argc == 4) {
        //int test = argv[1] - '0';
        Epochs epoch(strtol(argv[1], NULL,10));
        Patience patience(strtol(argv[2], NULL,10));
        BatchSize batch_size(strtol(argv[3], NULL,10));
        n1.train(data.get_x_train(), data.get_y_train(), sgd, epoch,
                 patience, batch_size);
    }
    else
        n1.train(data.get_x_train(), data.get_y_train(), sgd, Epochs(1),
                 Patience(10), BatchSize(32));
}
