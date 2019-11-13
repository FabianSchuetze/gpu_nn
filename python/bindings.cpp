#include "../include/neural_network.h"
#include "../third_party/pybind11/include/pybind11/eigen.h"
#include "../third_party/pybind11/include/pybind11/functional.h"
#include "../third_party/pybind11/include/pybind11/pybind11.h"
#include <algorithm>
#include <memory>
#include "../third_party/pybind11/include/pybind11/stl.h"

using pybind11::class_;
using pybind11::init;

template<typename T> struct Deleter { void operator() (T *o) const { delete o; } };

PYBIND11_MODULE(NeuralNetwork, m) {
    class_<Loss>(m, "Loss").def(init<>());
    class_<CrossEntropy, Loss>(m, "CrossEntropy").def(init<>());
    class_<Layer>(m, "Layer").def(init<>());
    class_<Dense, Layer>(m, "Dense").def(init<int, int>());
    class_<Input, Layer>(m, "Input").def(init<int>());
    class_<Relu, Layer>(m, "Relu").def(init<>());
    class_<Softmax, Layer>(m, "Softmax").def(init<>());
    class_<GradientDescent>(m, "GradientDescent").def(init<dtype>());
    class_<StochasticGradientDescent, GradientDescent>(
        m, "StochasticGradientDescent")
        .def(init<dtype>());
    class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(init<std::vector<Layer*>, std::string, std::string>())
        .def("train", &NeuralNetwork::train);
}
