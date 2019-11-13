#include "../include/neural_network.h"
#include "../third_party/pybind11/include/pybind11/eigen.h"
//#include "../pybind11/include/pybind11/functional.h"
//#include "../pybind11/include/pybind11/pybind11.h"
#include <memory>
#include "../third_party/pybind11/include/pybind11/stl.h"

PYBIND11_MODULE(NeuralNetwork, m) {
    pybind11::class_<Loss>(m, "Loss").def(pybind11::init<>());
    pybind11::class_<CrossEntropy, Loss>(m, "CrossEntropy").def(pybind11::init<>());
}
