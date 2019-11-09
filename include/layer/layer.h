#pragma once
#ifndef layer_h
#define layer_h
#include <memory>
#include <vector>
#include "../storage.h"
class Layer {
   protected:
    typedef std::shared_ptr<Storage> SharedStorage;

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
    virtual void backward_gpu(int&, const SharedStorage&, const SharedStorage&,
                              SharedStorage&) = 0;
    virtual void backward_cpu(int&, const SharedStorage&, const SharedStorage&,
                              SharedStorage&) = 0;
    virtual std::vector<SharedStorage> return_parameters() = 0;
    virtual std::vector<SharedStorage> return_gradients() = 0;
    virtual std::vector<SharedStorage> return_parameters() const = 0;
    virtual std::vector<SharedStorage> return_gradients() const = 0;

   protected:
    std::vector<SharedStorage> parameters;
    std::vector<SharedStorage> gradients;
    std::string _name;
};
#endif
