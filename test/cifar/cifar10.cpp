#include <stdlib.h>
#include <memory>
//#include <type_traits>
#include "../../include/neural_network.h"
//#include "../../include/utils/normalization.hpp"
//#include "../../third_party/cifar10/include/cifar/get_data.h"
#include <iostream>
#include "../../include/utils/io.h"

typedef std::shared_ptr<Layer> s_Layer;
using std::make_shared;

//void print_Matrix_to_stdout2(const Matrix& val, std::string loc) {
    //int rows(val.rows()), cols(val.cols());
    //std::ofstream myfile(loc);
    //myfile << "dimensions: rows, cols: " << rows << ", " << cols << std::endl;
    ////myfile << std::fixed;
    ////myfile << std::setprecision(2);
    //for (int row = 0; row < rows; ++row) {
        //myfile << val(row, 0);
        //for (int col = 1; col < cols; ++col) {
            //myfile << ", " << val(row, col);
        //}
        //myfile << std::endl;
    //}
//}
void n_missclassified(const Matrix& y_pred, const Matrix& y_true) {
    int missclassified(0);
    for (int i = 0; i < y_pred.rows(); ++i) {
        int arg_pred = 0;
        int arg_true = 0;
        dtype max_pred = y_pred(i, 0);
        dtype max_true = y_true(i, 0);
        for (int j = 0; j < 10; ++j) {
            if (max_pred < y_pred(i, j)) {
                arg_pred = j;
                max_pred = y_pred(i, j);
            }
            if (max_true < y_true(i, j)) {
                arg_true = j;
                max_true = y_true(i, j);
            }
        }
        if (arg_true != arg_pred) missclassified++;
    }
    std::cout << "fraction miassclassified : "
              << float(missclassified) / y_pred.rows() << " and "
              << "number missclassified " << missclassified << std::endl;
}

int main(int argc, char** argv) {
    Matrix x_train;
    read_binary("/home/fabian/Documents/work/gpu_nn/test/cifar/x_train.dat",
                x_train);
    Matrix y_train;
    read_binary("/home/fabian/Documents/work/gpu_nn/test/cifar/y_train.dat",
                y_train);
    s_Layer l1 = make_shared<Input>(Channels(3), ImageShape(32, 32));
    s_Layer conv1 =
        make_shared<Convolution>(FilterShape(5, 5), Pad(2), Stride(1),
                                 Filters(32), l1, new Normal(0, 0.0001));
    s_Layer pool1 = make_shared<Pooling>(Window(3), Stride(2), conv1);
    s_Layer conv2 =
        make_shared<Convolution>(FilterShape(5, 5), Pad(2), Stride(1),
                                 Filters(32), pool1, new Normal(0, 0.01));
    s_Layer pool2 = make_shared<Pooling>(Window(3), Stride(2), conv2);
    s_Layer conv3 =
        make_shared<Convolution>(FilterShape(5, 5), Pad(2), Stride(1),
                                 Filters(64), pool2, new Normal(0., 0.01));
    s_Layer pool3 = make_shared<Pooling>(Window(3), Stride(2), conv3);
    s_Layer d1 = make_shared<Dense>(Features(64), pool3, new Normal(0., 0.1));
    //s_Layer r1 = make_shared<Relu>(d1);
    s_Layer d2 = make_shared<Dense>(Features(10), d1, new Normal(0., 0.1));
    s_Layer s1 = make_shared<Softmax>(d2);
    std::shared_ptr<Loss> loss =
        std::make_shared<CrossEntropy>(CrossEntropy("GPU"));
    NeuralNetwork n1(s1, loss, "GPU");
    std::shared_ptr<GradientDescent> sgd = std::make_shared<Momentum>(
        LearningRate(0.001), MomentumRate(0.90), WeightDecay(0.004));
    n1.train(x_train, y_train, sgd, Epochs(30), Patience(10), BatchSize(32),
            true);
    // Matrix predictions = n1.predict(transform_data(data.get_x_test()));
    // n_missclassified(predictions, data.get_y_test());
}
