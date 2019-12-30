#include <stdlib.h>
#include <memory>
#include "../../include/neural_network.h"
#include <iostream>
#include "../../include/utils/io.h"
#include "../../include/metrics/missclassified.hpp"

typedef std::shared_ptr<Layer> s_Layer;
using std::make_shared;

//void n_missclassified(const Matrix& y_pred, const Matrix& y_true) {
    //int missclassified(0);
    //for (int i = 0; i < y_pred.rows(); ++i) {
        //int arg_pred = 0;
        //int arg_true = 0;
        //dtype max_pred = y_pred(i, 0);
        //dtype max_true = y_true(i, 0);
        //for (int j = 0; j < 10; ++j) {
            //if (max_pred < y_pred(i, j)) {
                //arg_pred = j;
                //max_pred = y_pred(i, j);
            //}
            //if (max_true < y_true(i, j)) {
                //arg_true = j;
                //max_true = y_true(i, j);
            //}
        //}
        //if (arg_true != arg_pred) missclassified++;
    //}
    //std::cout << "fraction miassclassified : "
              //<< float(missclassified) / y_pred.rows() << " and "
              //<< "number missclassified " << missclassified << std::endl;
//}

int main(int argc, char** argv) {
    Matrix x_train, x_test;
    read_binary("/home/fabian/Documents/work/gpu_nn/test/cifar/x_train.dat",
                x_train);
    read_binary("/home/fabian/Documents/work/gpu_nn/test/cifar/x_test.dat",
                x_test);
    Matrix y_train, y_test;
    read_binary("/home/fabian/Documents/work/gpu_nn/test/cifar/y_train.dat",
                y_train);
    read_binary("/home/fabian/Documents/work/gpu_nn/test/cifar/y_test.dat",
                y_test);
    Init* init = new Glorot();
    s_Layer l1 = make_shared<Input>(Channels(3), ImageShape(32, 32));
    s_Layer conv1 =
        make_shared<Convolution>(FilterShape(5, 5), Pad(2), Stride(1),
                                 Filters(64), l1, init);
    s_Layer pool1 = make_shared<Pooling>(Window(2), Stride(2), conv1);
    s_Layer conv2 =
        make_shared<Convolution>(FilterShape(5, 5), Pad(2), Stride(1),
                                 Filters(64), pool1, init);
    s_Layer pool2 = make_shared<Pooling>(Window(2), Stride(2), conv2);
    s_Layer conv3 =
        make_shared<Convolution>(FilterShape(5, 5), Pad(2), Stride(1),
                                 Filters(128), pool2,init);
    s_Layer pool3 = make_shared<Pooling>(Window(3), Stride(2), conv3);
    s_Layer d1 = make_shared<Dense>(Features(128), pool3,init);
    s_Layer drop1 = make_shared<Dropout>(0.5, d1);
    s_Layer r1 = make_shared<Relu>(drop1);
    s_Layer d2 = make_shared<Dense>(Features(10), r1,init);
    s_Layer drop2 = make_shared<Dropout>(0.5, d2);
    s_Layer s1 = make_shared<Softmax>(drop2);
    std::shared_ptr<Loss> loss =
        std::make_shared<CrossEntropy>(CrossEntropy("GPU"));
    NeuralNetwork n1(s1, loss, "GPU");
    std::shared_ptr<GradientDescent> sgd = std::make_shared<Momentum>(
        LearningRate(0.001), MomentumRate(0.90));
    std::vector<Metric*> metrics;
    Metric* val = new Missclassified(&n1);
    metrics.push_back(val);
    n1.train(x_train, y_train, sgd, Epochs(30), Patience(1), BatchSize(32)
            ,metrics);
     //Matrix predictions = n1.predict(x_test, 10);
     //n_missclassified(predictions, y_test);
}
