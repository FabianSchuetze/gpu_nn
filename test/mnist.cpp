#include "../third_party/mnist/include/mnist/get_data.h"
#include "../include/neural_network.h"


int main() {
    Mnist data = Mnist();
    srand((unsigned int) time(0));
    Layer* l1;
    Layer* l2;
    Layer* l3;
    Layer* l4;
    Layer* l5;
    Input i1(data.get_x_train().cols());
    Dense d1(100, data.get_x_train().cols());
    Relu r1;
    Dense d2(10, 100);
    Softmax  s1;
    l1 = &i1;
    l2 = &d1;
    l3 = &r1;
    l4 = &d2;
    l5 = &s1;
    std::vector<Layer*> vec = {l1, l2, l3, l4, l5};
    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropy>(CrossEntropy());
    NeuralNetwork n1(vec, loss, "CPU");
    std::shared_ptr<GradientDescent> sgd = 
        std::make_shared<StochasticGradientDescent>(0.001);
    n1.train(data.get_x_train(), data.get_y_train(), sgd);
}
