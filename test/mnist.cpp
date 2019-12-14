#include <stdlib.h>
//#include <type_traits>
#include <memory>
#include "../include/neural_network.h"
#include "../third_party/mnist/include/mnist/get_data.h"

typedef std::shared_ptr<Layer> s_Layer;
using std::make_shared;

int main(int argc, char** argv) {
    if ((argc != 2) and (argc != 5))
        throw std::invalid_argument("Must have one or four arguemnts");
    Mnist data = Mnist();
    srand((unsigned int)time(0));
    Init* init = new Glorot();
    s_Layer l1 = make_shared<Input>(Features(data.get_x_train().cols()));
    s_Layer d1 = make_shared<Dense>(Features(1024), l1, init);
    s_Layer r1 = make_shared<Relu>(d1);
    s_Layer drop1 = make_shared<Dropout>(0.5, r1);
    s_Layer d2 = make_shared<Dense>(Features(1024), drop1, init);
    s_Layer r2 = make_shared<Relu>(d2);
    s_Layer drop2 = make_shared<Dropout>(0.5, r2);
    s_Layer d3 = make_shared<Dense>(Features(10), drop2, init);
    s_Layer s1 = make_shared<Softmax>(d3);
    std::shared_ptr<Loss> loss =
        std::make_shared<CrossEntropy>(CrossEntropy("GPU"));
    NeuralNetwork n1(s1, loss, "GPU");
    std::shared_ptr<GradientDescent> sgd =
        std::make_shared<Momentum>(LearningRate(0.001), MomentumRate(0.75));
    if (argc == 5) {
        Epochs epoch(strtol(argv[2], NULL, 10));
        Patience patience(strtol(argv[3], NULL, 10));
        BatchSize batch_size(strtol(argv[4], NULL, 10));
        n1.train(data.get_x_train(), data.get_y_train(), sgd, epoch, patience,
                 batch_size, true);
    } else
        n1.train(data.get_x_train(), data.get_y_train(), sgd, Epochs(10),
                 Patience(10), BatchSize(32));
}
