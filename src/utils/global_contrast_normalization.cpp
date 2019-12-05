//#include <eigen-git-mirror/Eigen/Core/Map>
#include "../../include/utils/normalization.hpp"
using Eigen::Map;

GCN::GCN(int rows, int cols, int channels)
    : _rows(rows), _cols(cols), _channels(channels){};

Matrix GCN::transform(const Matrix& input) {
    Matrix tmp = input;
    Matrix reshaped = reshape(tmp);
    Eigen::VectorXf means = reshaped.colwise().mean();
    reshaped.rowwise() -= means.transpose();
    return inv_reshape(reshaped, input.rows());
}

Matrix GCN::inv_reshape(Matrix& input, int batches) {
    Map<Matrix> map(input.data(), _rows * _cols * _channels, batches);
    Matrix tmp = map.transpose();
    return tmp;
}

Matrix GCN::reshape(Matrix& input) {
    Matrix tmp = input.transpose();
    Map<Matrix> map(tmp.data(), _rows * _cols, input.rows() * _channels);
    return map;
}
