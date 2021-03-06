#ifndef network_h
#define network_h
#include <chrono>
#include <deque>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <vector>
#include "debug_info.hpp"
#include "gradient_descent/gradient_descent.h"
#include "layer/layer.h"
#include "loss/loss.h"
#include "threadsafe_queue.hpp"
#include "trainArgs.h"
//#include "metrics/metric.hpp"
class Metric;
class NeuralNetwork {
    friend class Metric;
    friend class CharRNN;

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
    //@brief Returns a prediction from the neural network, calling this
    // function allocates shared storage;
    Matrix predict(const Matrix&, DebugInfo&& = DebugInfo("", ""));
    //@brief Returns a prediction from the neural network, calling this
    // function presumes that the SharedStorage is appropriate for all the
    // layers
    // SharedStorage predict(std::vector<SharedStorage>&);
    // NOT DEFINED !!!
    void backwards(std::vector<SharedStorage>& gradients,
                   const std::vector<SharedStorage>& values, DebugInfo&);
    std::vector<SharedStorage> allocate_forward(int);
    std::vector<SharedStorage> allocate_backward(int);
    void forward(std::vector<SharedStorage>&, const std::string&, DebugInfo&);
    void fill_hiddens(std::vector<SharedStorage>&, const Matrix&);
    void update_weights(std::shared_ptr<GradientDescent>&,
                        std::vector<VecSharedStorage>&, int);
    void train(std::shared_ptr<GradientDescent>&);
    void train(const Matrix&, const Matrix&, std::shared_ptr<GradientDescent>&,
               Epochs, Patience, BatchSize, std::vector<Metric*>&,
               DebugInfo&& = DebugInfo("", ""), Shuffle = Shuffle(true));
    dtype validate(std::chrono::milliseconds);

   private:
    typedef void (NeuralNetwork::*update_func)(
        std::shared_ptr<GradientDescent>&, std::vector<VecSharedStorage>&, int);
    typedef void (NeuralNetwork::*forward_func)(std::vector<SharedStorage>&,
                                                const std::string&, DebugInfo&);
    typedef void (NeuralNetwork::*backward_func)(
        std::vector<SharedStorage>&, const std::vector<SharedStorage>&,
        DebugInfo&);
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
    void forward_gpu(std::vector<SharedStorage>&, const std::string&,
                     DebugInfo&);
    void forward_cpu(std::vector<SharedStorage>&, const std::string&,
                     DebugInfo&);
    void backward_cpu(std::vector<SharedStorage>&,
                      const std::vector<SharedStorage>&, DebugInfo&);
    void backward_gpu(std::vector<SharedStorage>&,
                      const std::vector<SharedStorage>&, DebugInfo&);
    void get_new_sample(const std::vector<int>&, Matrix&, Matrix&);
    void consumer(std::shared_ptr<GradientDescent>&, DebugInfo&,
                  std::vector<Metric*>&);
    void producer();
    std::vector<int> predict_sample(int&, int);
    void get_new_predict_sample(const std::vector<int>&, const Matrix&,
                                Matrix&);
    void consumer_predict(SharedStorage&,
                          threadsafe_queue<std::vector<SharedStorage>>*,
                          DebugInfo&);
    void producer_predict(const Matrix&,
                          threadsafe_queue<std::vector<SharedStorage>>*);
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
    int check_input_dimension(const std::vector<int>&);
    void print_network();
    void display_train_loss(dtype&);
    void predict(const Matrix&, SharedStorage&, DebugInfo&);
    bool continue_training();
};
#endif
