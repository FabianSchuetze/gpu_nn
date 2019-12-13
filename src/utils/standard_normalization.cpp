#include "../../include/utils/normalization.hpp"

StandardNormalization::StandardNormalization() {;};
Vector StandardNormalization::colwise_std(const Matrix& diff) {
    Matrix out = (diff.transpose() * diff) / float(diff.rows());
    out = out.array().sqrt();
    return out.diagonal();
}

Matrix StandardNormalization::transform(const Matrix& input) {
    Matrix tmp = input;
    Eigen::VectorXf means = tmp.colwise().mean();
    tmp.rowwise() -= means.transpose();
    //Vector std = colwise_std(tmp);
    //tmp.array().rowwise() /= std.transpose().array();
    return tmp;
}
