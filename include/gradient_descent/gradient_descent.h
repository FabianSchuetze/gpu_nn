#pragma once
#ifndef gradient_descent_h
#define gradient_descent_h
#include <memory>
#include <vector>
#include "../../include/layer/layer.h"
#include "../common.h"
#include "../storage.h"
class GradientDescent {
   protected:
    typedef std::shared_ptr<Storage> SharedStorage;
    typedef std::vector<SharedStorage> VecSharedStorage;
    dtype learing_rate;

   public:
    GradientDescent(dtype);
    virtual ~GradientDescent() {};
    virtual void weight_update_cpu(const VecSharedStorage&,
                                   VecSharedStorage&, int);
    virtual void weight_update_gpu(const VecSharedStorage&,
                                   VecSharedStorage&, int);
};
#endif
