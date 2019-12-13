#include <memory>
#include <stdlib.h>
//#include <type_traits>
#include "../../include/neural_network.h"
#include "../../include/utils/normalization.hpp"
#include "../../third_party/cifar10/include/cifar/get_data.h"
#include "../../include/utils/io.h"

Matrix transform_data(const Matrix& input) {
    GCN gcn(32, 32, 3);
    StandardNormalization scaler;
    Matrix norm = gcn.transform(input);
    Matrix norm2 = scaler.transform(input);
    return norm2;
}

int main() {
    Cifar10 data = Cifar10();
    Matrix x_train = data.get_x_train();
    Matrix x_test = data.get_x_test();
    Matrix y_train = data.get_y_train();
    Matrix y_test = data.get_y_test();
    write_binary("../test/cifar/x_train.dat", transform_data(x_train));
    write_binary("../test/cifar/x_test.dat", transform_data(x_test));
    write_binary("../test/cifar/y_train.dat", y_train);
    write_binary("../test/cifar/y_test.dat", y_test);
    //Matrix test;
    //read_binary("../test/Cifar/x_train.dat", test);
}
