#ifndef GUARD_get_data_h
#define GUARD_get_data_h
#include <eigen-git-mirror/Eigen/Dense>
#include <vector>
#include "mnist_reader.hpp"

typedef float dtype;
typedef Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic> Matrix;
class Mnist {
public:
    Mnist();
    Matrix get_x_train() { return x_train; }
    Matrix get_x_test() { return x_test; }
    Matrix get_y_train() { return y_train; }
    Matrix get_y_test() { return y_test; }

private:
    Matrix x_train, x_test;
    Matrix y_train, y_test;
    Matrix labels(const std::vector<uint8_t>&);
    Matrix features(const std::vector<std::vector<uint8_t>>&);
    std::vector<dtype> read(const std::vector<std::vector<uint8_t>>&);
};

// WHAT IS THE STORAGE ORDER HERE?
std::vector<dtype> Mnist::read(const std::vector<std::vector<uint8_t>>& inp) {
    int size = inp.size();
    std::vector<dtype> res;
    res.reserve(size * 784 + 1);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < 784; ++j) {
            res.push_back(inp[i][j]);
        }
    }
    return res;
}

Matrix Mnist::labels(const std::vector<uint8_t>& inp) {
    int size   = inp.size();
    Matrix res = Matrix::Zero(size, 10);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < 10; ++j) {
            if (inp[i] == j)
                res(i, j) = 1;
        }
    }
    return res;
}

Matrix Mnist::features(const std::vector<std::vector<uint8_t>>& inp) {
    int size               = inp.size();
    std::vector<dtype> tmp = read(inp);
    Matrix tmp2            = Eigen::Map<Matrix>(tmp.data(), 784, size);
    Matrix tmp3            = tmp2.transpose();
    tmp3                   = (tmp3.array() / 255).matrix();
    return tmp3;
}

Mnist::Mnist() : x_train(), x_test(), y_train(), y_test() {
    auto dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
                "/home/fabian/Documents/work/gpu_nn/third_party/mnist");
    y_train = labels(dataset.training_labels);
    y_test  = labels(dataset.test_labels);
    x_train = features(dataset.training_images);
    x_test  = features(dataset.test_images);
}
#endif
