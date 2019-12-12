#pragma once
#ifndef layer_h
#define layer_h
#include <memory>
#include <vector>
#include "../storage.h"
class Layer {
   protected:
    friend class NeuralNetwork;
    typedef std::shared_ptr<Storage> SharedStorage;
    typedef std::vector<std::shared_ptr<Storage>> VecSharedStorage;
    std::vector<int> _out_dim;
    std::vector<SharedStorage> parameters;
    std::vector<SharedStorage> gradients;
    std::string _name;
    std::shared_ptr<Layer> _previous;
    virtual void initialize_output_dimension(const std::shared_ptr<Layer>&);
    virtual void initialize_output_dimension();

   public:
    Layer() : _name("Template"){};
    explicit Layer(const std::string& s): _name(s) {};
    virtual ~Layer() = default;
    virtual int input_dimension() {return 0;}
    virtual int input_dimension() const {return 0;}
    virtual std::vector<int> output_dimension() {return _out_dim;}
    virtual std::vector<int> output_dimension() const {return _out_dim;}
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
    //virtual void clear_gradients_cpu();
    //virtual void clear_gradients_gpu();
    virtual int n_paras() { return parameters.size(); };
    virtual std::shared_ptr<Layer> previous() {return _previous;};
};
#endif
