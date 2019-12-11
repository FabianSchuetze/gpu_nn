#include <fstream>
#include <iostream>
#include "../include/neural_network.h"

void print_Matrix_to_stdout3(const Matrix& val, std::string loc) {
    int rows(val.rows()), cols(val.cols());
    std::ofstream myfile(loc);
    myfile << "dimensions: rows, cols: " << rows << ", " << cols << std::endl;
    myfile << std::fixed;
    myfile << std::setprecision(3);
    for (int row = 0; row < rows; ++row) {
        myfile << val(row, 0);
        for (int col = 1; col < cols; ++col) {
            myfile << ", " << val(row, col);
        }
        myfile << std::endl;
    }
}

int main() {
    Init* norm = new Normal(0, 0.01);
    Matrix test = norm->weights(75, 32);
    print_Matrix_to_stdout3(
        test, "/home/fabian/Documents/work/gpu_nn/debug/norm.txt");
    Init* glorot = new Glorot();
    Matrix glo = glorot->weights(75, 32);
    print_Matrix_to_stdout3(
        glo, "/home/fabian/Documents/work/gpu_nn/debug/glo.txt");
}
