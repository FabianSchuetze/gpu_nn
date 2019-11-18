#ifndef network_h
#define network_h
#include <chrono>
#include <memory>
#include <random>
#include <vector>
#include "gradient_descent/gradient_descent.h"
#include "layer/layer.h"
#include "loss/loss.h"
#include "trainArgs.h"
class NeuralNetwork {
   public:
       // I NEED TO THINK ABOUT THE MOVE CONSTCUTOR
    NeuralNetwork(std::vector<Layer*>, std::shared_ptr<Loss>);
    NeuralNetwork(std::vector<Layer*>, std::shared_ptr<Loss>,
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
    // SharedStorage predict(std::vector<SharedStorage>&);
    // NOT DEFINED !!!
    void backwards(std::vector<SharedStorage>& gradients,
                   const std::vector<SharedStorage>& values);
    std::vector<SharedStorage> allocate_forward(int);
    std::vector<SharedStorage> allocate_backward(int);
    void forward(std::vector<SharedStorage>&);
    void fill_hiddens(std::vector<SharedStorage>&, const Matrix&);
    void update_weights(std::shared_ptr<GradientDescent>, int);
    void train(std::shared_ptr<GradientDescent>);
    void train(const Matrix&, const Matrix&, std::shared_ptr<GradientDescent>,
               Epochs, Patience, BatchSize);
    void validate(std::chrono::milliseconds);
    void random_numbers(std::vector<int>&, std::mt19937&);

   private:
    typedef void (NeuralNetwork::*update_func)(
        std::shared_ptr<GradientDescent>, int);
    typedef void (NeuralNetwork::*forward_func)(std::vector<SharedStorage>&);
    typedef void (NeuralNetwork::*backward_func)(
        std::vector<SharedStorage>&, const std::vector<SharedStorage>&);
    NeuralNetwork::forward_func fun_forward;
    NeuralNetwork::backward_func fun_backward;
    NeuralNetwork::update_func fun_update;
    std::vector<Layer*> layers;
    std::shared_ptr<Loss> loss;
    std::unique_ptr<trainArgs> train_args;
    void create_loss(const std::string& s);
    void update_weights_cpu(std::shared_ptr<GradientDescent>, int);
    void update_weights_gpu(std::shared_ptr<GradientDescent>, int);
    void forward_gpu(std::vector<SharedStorage>&);
    void forward_cpu(std::vector<SharedStorage>&);
    void backward_cpu(std::vector<SharedStorage>&,
                      const std::vector<SharedStorage>&);
    void backward_gpu(std::vector<SharedStorage>&,
                      const std::vector<SharedStorage>&);
    void allocate_storage(int, int&, std::vector<SharedStorage>&, const Layer*);
    void get_new_sample(const std::vector<int>&, Matrix&, Matrix&);
    void consumer(std::shared_ptr<GradientDescent>);
    void producer();
};
#endif
