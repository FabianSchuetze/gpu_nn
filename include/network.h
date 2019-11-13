#ifndef network_h
#define network_h
#include <memory>
#include <random>
#include <vector>
#include "gradient_descent/gradient_descent.h"
#include "layer/layer.h"
#include "loss/loss.h"
class NeuralNetwork {
   public:
    NeuralNetwork(std::vector<Layer*>, std::shared_ptr<Loss>);
    NeuralNetwork(std::vector<Layer*>,  const std::string& loss,
                  const std::string&);
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork(NeuralNetwork&&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(NeuralNetwork&&) = delete;
    // What predict function do I need?
    //@brief Returns a prediction from the neural network, calling this
    // function allocates shared storage;
    Matrix predict(const Matrix&);
    //@brief Returns a prediction from the neural network, calling this
    // function presumes that the SharedStorage is appropriate for all the
    // layers
    SharedStorage predict(std::vector<SharedStorage>&);
    void backwards(std::vector<SharedStorage>& gradients,
                   const std::vector<SharedStorage>& values);
    std::vector<SharedStorage> allocate_forward(int);
    std::vector<SharedStorage> allocate_backward(int);
    void forward(std::vector<SharedStorage>&);
    void fill_hiddens(std::vector<SharedStorage>&, const Matrix&);
    void update_weights(GradientDescent*);
    void train(const Matrix&, const Matrix&, GradientDescent*);
    void get_new_sample(const Matrix&, const Matrix&, const std::vector<int>&,
                        Matrix&, Matrix&);
    void random_numbers(std::vector<int>&, std::mt19937&, const Matrix&);

   private:
    typedef void (NeuralNetwork::*update_func)(
        GradientDescent*);
    typedef void (NeuralNetwork::*forward_func)(std::vector<SharedStorage>&);
    typedef void (NeuralNetwork::*backward_func)(
        std::vector<SharedStorage>&, const std::vector<SharedStorage>&);
    NeuralNetwork::forward_func fun_forward;
    NeuralNetwork::backward_func fun_backward;
    NeuralNetwork::update_func fun_update;
    std::vector<Layer*> layers;
    std::shared_ptr<Loss> loss;
    void create_loss(const std::string& s);
    void update_weights_cpu(GradientDescent*);
    void update_weights_gpu(GradientDescent*);
    void forward_gpu(std::vector<SharedStorage>&);
    void forward_cpu(std::vector<SharedStorage>&);
    void backward_cpu(std::vector<SharedStorage>&,
                      const std::vector<SharedStorage>&);
    void backward_gpu(std::vector<SharedStorage>&,
                      const std::vector<SharedStorage>&);
    void allocate_storage(int, int&, std::vector<SharedStorage>&, const Layer*);
};
#endif
