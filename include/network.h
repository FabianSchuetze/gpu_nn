#ifndef network_h
#define network_h
#include <chrono>
#include <deque>
#include <list>
#include <memory>
#include <random>
#include <vector>
#include "gradient_descent/gradient_descent.h"
#include "layer/layer.h"
#include "loss/loss.h"
#include "threadsafe_queue.hpp"
#include "trainArgs.h"
class NeuralNetwork {
   public:
    typedef std::vector<std::shared_ptr<Storage>> VecSharedStorage;
    // I NEED TO THINK ABOUT THE MOVE CONSTCUTOR
    NeuralNetwork(const std::shared_ptr<Layer>&, std::shared_ptr<Loss>&);
    NeuralNetwork(const std::shared_ptr<Layer>&, std::shared_ptr<Loss>&,
                  const std::string&);
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork(NeuralNetwork&&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(NeuralNetwork&&) = delete;
    // What predict function do I need?
    //@brief Returns a prediction from the neural network, calling this
    // function allocates shared storage;
    Matrix predict(const Matrix&);
    void predict(const Matrix&, SharedStorage&);
    //@brief Returns a prediction from the neural network, calling this
    // function presumes that the SharedStorage is appropriate for all the
    // layers
    // SharedStorage predict(std::vector<SharedStorage>&);
    // NOT DEFINED !!!
    void backwards(std::vector<SharedStorage>& gradients,
                   const std::vector<SharedStorage>& values);
    std::vector<SharedStorage> allocate_forward(int);
    std::vector<SharedStorage> allocate_backward(int);
    void forward(std::vector<SharedStorage>&, const std::string&);
    void fill_hiddens(std::vector<SharedStorage>&, const Matrix&);
    void update_weights(std::shared_ptr<GradientDescent>&,
                        std::vector<VecSharedStorage>&, int);
    void train(std::shared_ptr<GradientDescent>&);
    void train(const Matrix&, const Matrix&, std::shared_ptr<GradientDescent>&,
               Epochs, Patience, BatchSize);
    dtype validate(std::chrono::milliseconds);
    void random_numbers(std::vector<int>&, std::mt19937&);

   private:
    typedef void (NeuralNetwork::*update_func)(
        std::shared_ptr<GradientDescent>&, std::vector<VecSharedStorage>&, int);
    typedef void (NeuralNetwork::*forward_func)(std::vector<SharedStorage>&,
                                                const std::string&);
    typedef void (NeuralNetwork::*backward_func)(
        std::vector<SharedStorage>&, const std::vector<SharedStorage>&);
    NeuralNetwork::forward_func fun_forward;
    NeuralNetwork::backward_func fun_backward;
    NeuralNetwork::update_func fun_update;
    std::deque<std::shared_ptr<Layer>> layers;
    std::shared_ptr<Loss> loss;
    std::unique_ptr<trainArgs> train_args;
    void create_loss(const std::string& s);
    void update_weights_cpu(std::shared_ptr<GradientDescent>&,
                            std::vector<VecSharedStorage>&, int);
    void update_weights_gpu(std::shared_ptr<GradientDescent>&,
                            std::vector<VecSharedStorage>&, int);
    void forward_gpu(std::vector<SharedStorage>&, const std::string&);
    void forward_cpu(std::vector<SharedStorage>&, const std::string&);
    void backward_cpu(std::vector<SharedStorage>&,
                      const std::vector<SharedStorage>&);
    void backward_gpu(std::vector<SharedStorage>&,
                      const std::vector<SharedStorage>&);
    // void allocate_storage(int, int&, std::vector<SharedStorage>&, const
    // Layer*);
    void get_new_sample(const std::vector<int>&, Matrix&, Matrix&);
    void consumer(std::shared_ptr<GradientDescent>&);
    void producer();
    std::vector<int> predict_sample(int&, int);
    void get_new_predict_sample(const std::vector<int>&, const Matrix&,
                                Matrix&);
    void consumer_predict(SharedStorage&,
                          threadsafe_queue<std::vector<SharedStorage>>*);
    void producer_predict(const Matrix&,
                          threadsafe_queue<std::vector<SharedStorage>>*);
    void update_bkp(dtype curr);
    void restore();
    void append_convolution_layer(Layer*);
    void construct_layers(std::vector<Layer*>);
    void insert_cnn_layer(const std::shared_ptr<Layer>&);
    void construct_layers(std::shared_ptr<Layer>);
    int convert_output_dimension(const std::shared_ptr<Layer>&);
    void allocate_storage(int, std::vector<SharedStorage>&,
                          const std::shared_ptr<Layer>&);
    void maybe_shuffle(std::vector<int>&, int&, const int&, const int&,
                       std::mt19937&);
    void prepare_subset(const std::vector<int>&, std::vector<int>&, int&,
                        const int&);
};
#endif
