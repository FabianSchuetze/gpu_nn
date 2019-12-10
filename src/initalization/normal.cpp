#include "../../include/initalization/normal.hpp"
#include <random>

Normal::Normal(dtype mean, dtype std)
    : Init("Normal"), _mean(mean), _std(std){};

Matrix Normal::weights(int rows, int cols) const {
    std::mt19937 gen;
    std::normal_distribution<> normal(_mean, _std);
    Matrix mat = Matrix::Zero(rows, cols);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; ++col) {
            mat(row, col) = normal(gen);
        }
    }
    return mat;
}
