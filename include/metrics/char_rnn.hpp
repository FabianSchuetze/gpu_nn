#pragma once
#ifndef char_rnn_hpp
#define char_rnn_hpp
#include <memory>
#include <vector>
#include "metric.hpp"
class CharRNN : public Metric {
   public:
    CharRNN(int length, NeuralNetwork*, const std::map<int, char>&,
            dtype temperate = 1.);
    virtual ~CharRNN() {};
    void validate(const Matrix&, const Matrix&) override;
   private:
    int _length;
    std::map<int, char> _ix_to_char;
    std::mt19937 gen;
    dtype temperature;
    void convert_to_stdout(const std::vector<int>&);
    int pick_value(const Matrix&);
};
#endif
