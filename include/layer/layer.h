#pragma once
#ifndef layer_h
#define layer_h
#include <memory>
#include <vector>
#include "../storage.h"
class Layer {
   protected:
    typedef std::shared_ptr<Storage> SharedStorage;
    typedef std::vector<std::shared_ptr<Storage>> VecSharedStorage;

   public:
    Layer() : _name("Template"){};
    ~Layer() = default;
    virtual int input_dimension() = 0;
    virtual int input_dimension() const = 0;
    virtual int output_dimension() = 0;
    virtual int output_dimension() const = 0;
    virtual std::string name() { return _name; };
    virtual std::string name() const { return _name; };
    virtual void forward_gpu(const SharedStorage&, SharedStorage&) = 0;
    virtual void forward_cpu(const SharedStorage&, SharedStorage&) = 0;
    virtual void backward_gpu(const SharedStorage&, const SharedStorage&,
                              SharedStorage&) = 0;
    virtual void backward_cpu(const SharedStorage&, const SharedStorage&,
                              SharedStorage&) = 0;
    virtual VecSharedStorage return_parameters() = 0;
    virtual VecSharedStorage return_gradients() = 0;
    virtual VecSharedStorage return_parameters() const = 0;
    virtual VecSharedStorage return_gradients() const = 0;
    virtual void clear_gradients_cpu() = 0;
    virtual void clear_gradients_gpu()= 0;

   protected:
    std::vector<SharedStorage> parameters;
    std::vector<SharedStorage> gradients;
    std::string _name;
};
#endif
