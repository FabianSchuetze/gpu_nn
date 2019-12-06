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

int main() {
    Matrix test = Matrix::Random(3, 4);
    print_Matrix_to_stdout2(
        test, "/home/fabian/Documents/work/gpu_nn/debug/test.txt");
    //int rows = 2;
    //int cols = 2;
    //int channels = 2;
    StandardNormalization scaler;
    Matrix out = scaler.transform(test);
    print_Matrix_to_stdout2(
        out, "/home/fabian/Documents/work/gpu_nn/debug/out.txt");

}
