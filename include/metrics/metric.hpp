#pragma once
#ifndef metric_hpp
#define metric_hpp
#include <memory>
#include <vector>
#include "../neural_network.h"
//#include "../../include/layer/layer.h"
#include "../common.h"
#include "../storage.h"
class Metric {
   protected:
    std::string _name;
    NeuralNetwork* _nn;

   public:
    Metric(const std::string& s, NeuralNetwork* nn): _name(s), _nn(nn) {};
    virtual ~Metric() {};
    virtual void validate(const Matrix&, const Matrix&) = 0;
};
#endif
