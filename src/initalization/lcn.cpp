#include "../../include/initalization/lcn.hpp"
#include <random>

LCN::LCN(dtype value)
    : Init("LCN"), _value(value){};

Matrix LCN::weights(int rows, int cols) const {
    Matrix mat = Matrix::Zero(rows, cols);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; ++col) {
            mat(row, col) = _value;
        }
    }
    return mat;
}
