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
    virtual ~Layer() = default;
    virtual int input_dimension();
    virtual int input_dimension() const;
    virtual int output_dimension();
    virtual int output_dimension() const;
    virtual std::string name() { return _name; };
    virtual std::string name() const { return _name; };
    virtual void forward_gpu(const SharedStorage&, SharedStorage&,
                             const std::string&);
    virtual void forward_cpu(const SharedStorage&, SharedStorage&,
                             const std::string&);
    virtual void backward_gpu(const SharedStorage&, const SharedStorage&,
                              SharedStorage&);
    virtual void backward_cpu(const SharedStorage&, const SharedStorage&,
                              SharedStorage&);
    virtual VecSharedStorage return_parameters();
    virtual VecSharedStorage return_gradients();
    virtual VecSharedStorage return_parameters() const;
    virtual VecSharedStorage return_gradients() const;
    virtual void clear_gradients_cpu();
    virtual void clear_gradients_gpu();
    virtual int n_paras() { return parameters.size(); };

   protected:
    std::vector<SharedStorage> parameters;
    std::vector<SharedStorage> gradients;
    std::string _name;
};
#endif
