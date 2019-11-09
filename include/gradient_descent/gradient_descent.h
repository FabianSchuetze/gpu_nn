#pragma once
#ifndef gradient_descent_h
#define gradient_descent_h
#include <memory>
#include <vector>
#include "../storage.h"
#include "../common.h"
#include "../../include/layer/layer.h"
class GradientDescent {
    protected:
    typedef std::shared_ptr<Storage> SharedStorage;
    int learing_rate;
   public:
    GradientDescent(int);
    ~GradientDescent() = default;
    virtual void weight_update_gpu(Layer*) = 0;
    virtual void weight_update_cpu(Layer*) = 0;
};
#endif
