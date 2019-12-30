#include "../../include/initalization/glorot.hpp"

Matrix Glorot::weights(int rows, int cols) const {
    srand((unsigned int)time(0));
    Matrix mat = Matrix::Random(rows, cols);
    dtype glorot_scale = std::sqrt(6.) / std::sqrt(rows + cols);
    mat *= glorot_scale;
    return mat;
}
