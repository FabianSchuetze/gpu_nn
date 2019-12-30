#include "../../include/metrics/missclassified.hpp"
#include <iostream>

Missclassified::Missclassified(NeuralNetwork* nn)
    : Metric("Missclassified", nn) { };

void Missclassified::validate(const Matrix& features, const Matrix& targets) {
    int out_size = targets.cols();
    Matrix res = _nn->predict(features, out_size);
    n_missclassified(res, targets);
}

void Missclassified::n_missclassified(const Matrix& y_pred,
                                      const Matrix& y_true) {
    int missclassified(0);
    for (int i = 0; i < y_pred.rows(); ++i) {
        int arg_pred = 0;
        int arg_true = 0;
        dtype max_pred = y_pred(i, 0);
        dtype max_true = y_true(i, 0);
        for (int j = 0; j < 10; ++j) {
            if (max_pred < y_pred(i, j)) {
                arg_pred = j;
                max_pred = y_pred(i, j);
            }
            if (max_true < y_true(i, j)) {
                arg_true = j;
                max_true = y_true(i, j);
            }
        }
        if (arg_true != arg_pred) missclassified++;
    }
    std::cout << "fraction miassclassified : "
              << float(missclassified) / y_pred.rows() << " and "
              << "number missclassified " << missclassified << std::endl;
}
