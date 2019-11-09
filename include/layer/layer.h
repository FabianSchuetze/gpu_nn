#ifndef layer_h
#define layer_h
#include <algorithm>
#include <memory>
#include <vector>
#include "../storage.h"
class Layer {
   public:
    Layer(): _name("Template") {};
    ~Layer() = default;
    virtual int input_dimension() = 0;
    virtual int input_dimension() const = 0;
    virtual int output_dimension() = 0;
    virtual int output_dimension() const = 0;
    virtual std::string name() {return _name;};
    virtual std::string name() const {return _name;};
    virtual void forward_gpu(const std::shared_ptr<Storage>&,
                             std::shared_ptr<Storage>&) = 0;
    virtual void forward_cpu(const std::shared_ptr<Storage>&,
                             std::shared_ptr<Storage>&) = 0;
    virtual void backward_gpu(int&, const std::shared_ptr<Storage>&,
                              std::vector<std::shared_ptr<Storage>>&) = 0;
    virtual void backward_cpu(int&, const std::shared_ptr<Storage>&,
                              std::vector<std::shared_ptr<Storage>>&) = 0;
    virtual std::vector<std::shared_ptr<Storage>> return_parameters() = 0;
    virtual std::vector<std::shared_ptr<Storage>> return_gradients() = 0;
    virtual std::vector<std::shared_ptr<Storage>> return_parameters() const = 0;
    virtual std::vector<std::shared_ptr<Storage>> return_gradients() const = 0;

   protected:
    std::vector<std::shared_ptr<Storage>> parameters;
    std::vector<std::shared_ptr<Storage>> gradients;
    std::string _name;
};
#endif
