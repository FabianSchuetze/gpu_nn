#include <stdlib.h>
//#include <type_traits>
#include "../include/neural_network.h"
#include "../third_party/mnist/include/mnist/get_data.h"

int main(int argc, char** argv) {
    Mnist data = Mnist();
    Init* init = new Glorot();
    Layer* l1 = new Input(data.get_x_train().cols());
    Layer* imcol1 = new Im2ColLayer(FilterShape(5, 5), Pad(2), Stride(1),
                                    ImageShape(28, 28), Channels(1));
    Layer* conv1 =
        new Convolution(FilterShape(5, 5), Pad(2), Stride(1), Filters(20),
                        ImageShape(28, 28), Channels(1), init);
    Layer* pool1 =
        new Pooling(Window(2), Stride(2), ImageShape(28, 28), Channels(20));
    Layer* imcol2 = new Im2ColLayer(FilterShape(5, 5), Pad(2), Stride(1),
                                    ImageShape(14, 14), Channels(20));
    Layer* conv2 =
        new Convolution(FilterShape(5, 5), Pad(2), Stride(1), Filters(50),
                        ImageShape(14, 14), Channels(20), init);
    Layer* pool2 =
        new Pooling(Window(2), Stride(2), ImageShape(14, 14), Channels(50));
    Layer* d1 = new Dense(500, 7 * 7 * 50,init);
    Layer* d2 = new Dense(10, 500, init);
    Layer* s1 = new Softmax;
    std::shared_ptr<Loss> loss =
        std::make_shared<CrossEntropy>(CrossEntropy("GPU"));
    NeuralNetwork n1({l1, imcol1, conv1,  pool1, imcol2, conv2, pool2, d1, 
                      d2, s1}, loss, "GPU");
    std::shared_ptr<GradientDescent> sgd =
        std::make_shared<Momentum>(LearningRate(0.001), MomentumRate(0.90));
    n1.train(data.get_x_train(), data.get_y_train(), sgd,
             Epochs(30), Patience(10), BatchSize(32));
}
