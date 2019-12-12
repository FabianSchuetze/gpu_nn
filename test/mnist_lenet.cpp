#include <stdlib.h>
//#include <type_traits>
#include "../include/neural_network.h"
#include "../third_party/mnist/include/mnist/get_data.h"

typedef std::shared_ptr<Layer> s_Layer;
using std::make_shared;

int main(int argc, char** argv) {
    Mnist data = Mnist();
    Init* init = new Glorot();
    //s_Layer l1 = make_shared<Input>(Features(data.get_x_train().cols()));
    s_Layer l1 = make_shared<Input>(Channels(1), ImageShape(28, 28));
    s_Layer conv1 = make_shared<Convolution>(FilterShape(5, 5), Pad(2),
                                             Stride(1), Filters(20), l1, init);
    s_Layer pool1 = make_shared<Pooling>(Window(2), Stride(2), conv1);
    s_Layer conv2 = make_shared<Convolution>(
        FilterShape(5, 5), Pad(2), Stride(1), Filters(50), pool1, init);
    s_Layer pool2 = make_shared<Pooling>(Window(2), Stride(2), conv2);
    s_Layer d1 = make_shared<Dense>(Features(500), pool2, init);
    s_Layer d2 = make_shared<Dense>(Features(10), d1, init);
    s_Layer s1 = make_shared<Softmax>(d2);
    std::shared_ptr<Loss> loss =
        std::make_shared<CrossEntropy>(CrossEntropy("GPU"));
    NeuralNetwork n1(s1, loss, "GPU");
    std::shared_ptr<GradientDescent> sgd =
        std::make_shared<Momentum>(LearningRate(0.001), MomentumRate(0.90));
    n1.train(data.get_x_train(), data.get_y_train(), sgd, Epochs(30),
             Patience(10), BatchSize(32));
}
