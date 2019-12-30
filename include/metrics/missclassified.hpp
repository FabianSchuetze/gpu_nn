#pragma once
#ifndef missclassified_hpp
#define missclassified_hpp
#include <vector>
#include "metric.hpp"
class Missclassified : public Metric {
   public:
    explicit Missclassified(NeuralNetwork*);
    virtual ~Missclassified(){};
    void validate(const Matrix&, const Matrix&) override;

   private:
    void n_missclassified(const Matrix&, const Matrix&);
};
#endif
