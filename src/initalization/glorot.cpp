#include "../../include/initalization/glorot.hpp"
#include <random>

Matrix Glorot::weights(int rows, int cols) const {
    std::mt19937 gen;
    gen.seed(0);
    std::uniform_real_distribution<dtype> dis(-1.0, 1.0);
    Matrix mat = Matrix::NullaryExpr(rows, cols, [&]() { return dis(gen); });
    dtype glorot_scale = std::sqrt(6.) / std::sqrt(rows + cols);
    mat *= glorot_scale;
    return mat;
}
