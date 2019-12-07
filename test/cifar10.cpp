#include <stdlib.h>
//#include <type_traits>
#include "../include/neural_network.h"
#include "../include/utils/normalization.hpp"
#include "../third_party/cifar10/include/cifar/get_data.h"

void print_Matrix_to_stdout2(const Matrix& val, std::string loc) {
    int rows(val.rows()), cols(val.cols());
    std::ofstream myfile(loc);
    myfile << "dimensions: rows, cols: " << rows << ", " << cols << std::endl;
    myfile << std::fixed;
    myfile << std::setprecision(2);
    for (int row = 0; row < rows; ++row) {
        myfile << val(row, 0);
        for (int col = 1; col < cols; ++col) {
            myfile << ", " << val(row, col);
        }
        myfile << std::endl;
    }
}
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

Matrix transform_data(const Matrix& input) {
    GCN gcn(32, 32, 3);
    StandardNormalization scaler;
    ZCAWhitening zca;
    Matrix norm = gcn.transform(input);
    Matrix norm2 = scaler.transform(norm);
    return norm2;
    //zca.fit(norm2);
    //Matrix white = zca.transform(norm2);
    //return white;
}

int main(int argc, char** argv) {
    Cifar10 data = Cifar10();
    srand((unsigned int)time(0));
    Layer* l1 = new Input(data.get_x_train().cols());
    Layer* imcol1 = new Im2ColLayer(FilterShape(5, 5), Pad(2), Stride(1),
                                    ImageShape(32, 32), Channels(3));
    Layer* conv1 =
        new Convolution(FilterShape(5, 5), Pad(2), Stride(1), Filters(32),
                        ImageShape(32, 32), Channels(3));
    Layer* relu1 = new Relu;
    Layer* pool1 =
        new Pooling(Window(2), Stride(2), ImageShape(32, 32), Channels(32));
    // Stide < Window: Overlapping pooling, as in AlexNet
    Layer* imcol2 = new Im2ColLayer(FilterShape(5, 5), Pad(2), Stride(1),
                                    ImageShape(16, 16), Channels(32));
    Layer* conv2 =
        new Convolution(FilterShape(5, 5), Pad(2), Stride(1), Filters(32),
                        ImageShape(16, 16), Channels(32));
    Layer* relu2 = new Relu;
    Layer* pool2 =
        new Pooling(Window(2), Stride(2), ImageShape(16, 16), Channels(32));
    Layer* imcol3 = new Im2ColLayer(FilterShape(5, 5), Pad(2), Stride(1),
                                    ImageShape(8, 8), Channels(32));
    Layer* conv3 =
        new Convolution(FilterShape(5, 5), Pad(2), Stride(1), Filters(64),
                        ImageShape(8, 8), Channels(32));
    Layer* relu3 = new Relu;
    Layer* pool3 =
        new Pooling(Window(2), Stride(2), ImageShape(8, 8), Channels(64));
    Layer* d1 = new Dense(64, 4 * 4 * 64);
    Layer* relu4 = new Relu;
    //Layer* d2 = new Dense(2048, 2048);
    //Layer* relu5 = new Relu;
    Layer* d3 = new Dense(10, 64);
    Layer* s1 = new Softmax;
    std::shared_ptr<Loss> loss =
        std::make_shared<CrossEntropy>(CrossEntropy("GPU"));
    NeuralNetwork n1({l1,    imcol1, conv1, relu1,  pool1, imcol2, conv2,
                      relu2, pool2,  relu3, imcol3, conv3, relu3,  pool3,
                      d1,    relu4,  d3,    s1},
                     loss, "GPU");
    std::shared_ptr<GradientDescent> sgd =
        std::make_shared<Momentum>(LearningRate(0.001), MomentumRate(0.90),
                                   WeightDecay(0.004));
    //transform_data(data.get_x_train());
    n1.train(transform_data(data.get_x_train()), data.get_y_train(), sgd,
             Epochs(20), Patience(10), BatchSize(32));
     Matrix predictions = n1.predict(transform_data(data.get_x_test()));
     n_missclassified(predictions, data.get_y_test());
}
