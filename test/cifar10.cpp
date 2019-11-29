#include <stdlib.h>
//#include <type_traits>
#include "../include/neural_network.h"
#include "../third_party/cifar10/include/cifar/get_data.h"

int main(int argc, char** argv) {
    // if ((argc != 2) and (argc != 5))
    // throw std::invalid_argument("Must have one or four arguemnts");
    Cifar10 data = Cifar10();
    Matrix test = data.get_x_train();
    srand((unsigned int)time(0));
    Layer* l1 = new Input(data.get_x_train().cols());
    Layer* l2 = new Im2ColLayer(FilterShape(3, 3), Pad(1), Stride(1),
                                ImageShape(32, 32), Channels(3));
    Layer* l3 = new Convolution(FilterShape(3, 3), Pad(1), Stride(1),
                                Filters(5), ImageShape(32, 32), Channels(3));
    Layer* l4 = new Relu;
    Layer* l5 = new Dense(10, Filters(5).get() * 32 * 32);
    Layer* l6 = new Softmax;
    std::shared_ptr<Loss> loss =
        std::make_shared<CrossEntropy>(CrossEntropy("GPU"));
    NeuralNetwork n1({l1, l2, l3, l4, l5, l6}, loss, "GPU");
    std::shared_ptr<GradientDescent> sgd =
        std::make_shared<StochasticGradientDescent>(0.001);
    // if (argc == 5) {
    // Epochs epoch(strtol(argv[2], NULL, 10));
    // Patience patience(strtol(argv[3], NULL, 10));
    // BatchSize batch_size(strtol(argv[4], NULL, 10));
    // n1.train(data.get_x_train(), data.get_y_train(), sgd, epoch, patience,
    // batch_size);
    n1.train(data.get_x_train(), data.get_y_train(), sgd, Epochs(10),
             Patience(10), BatchSize(32));
    delete l1;
    delete l2;
    delete l3;
    delete l4;
    delete l5;
    delete l6;
}
