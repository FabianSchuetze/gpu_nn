#include "../../include/utils/normalization.hpp"
#include <eigen-git-mirror/Eigen/Eigenvalues>
#include <eigen-git-mirror/Eigen/src/Eigenvalues/EigenSolver.h>
#include <stdexcept>
#include <fstream>

using Eigen::EigenSolver;
using Eigen::MatrixXd;
ZCAWhitening::ZCAWhitening() {;};


void ZCAWhitening::compute_covariance(const MatrixXd& input) {
    cov = (input.transpose() * input) / double(input.rows());
}

void ZCAWhitening::solve_eigensystem(const MatrixXd& varcov) {
    EigenSolver<MatrixXd> es(varcov);
    eigenvectors = es.eigenvectors().real();
    eigenvalues = es.eigenvalues().real();
    Eigen::VectorXd tmp = 1.0f / eigenvalues.array().sqrt();
    MatrixXd Lambda = tmp.asDiagonal();
    weights =  eigenvectors * Lambda * eigenvectors.transpose();
}

void ZCAWhitening::fit(const Matrix& input) {
    Eigen::MatrixXd tmp = input.cast<double>();
    compute_covariance(tmp);
    solve_eigensystem(cov);
}

Matrix ZCAWhitening::transform(const Matrix& input) {
    Eigen::MatrixXd tmp = input.cast<double>();
    MatrixXd res = tmp * weights;
    return res.cast<float>();
}
