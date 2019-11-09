#ifndef network_h
#define network_h
#include <vector>
#include "layer/layer.h"
#include "loss/loss.h"
class NeuralNetwork {
   public:
    NeuralNetwork(std::vector<Layer*>, const std::string&);
    NeuralNetwork(std::vector<Layer*>, const std::string&, const std::string&);
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
    std::vector<SharedStorage> allocate_shared_storage(int);
    // DANGEROUS DO THAT FOR TEST CASE!!
    std::shared_ptr<Loss> loss;
    void forward(std::vector<SharedStorage>&);
    void fill_hiddens(std::vector<SharedStorage>&, const Matrix&);

   private:
    typedef void (NeuralNetwork::*forward_func)(std::vector<SharedStorage>&);
    typedef void (NeuralNetwork::*backward_func)(
        std::vector<SharedStorage>&, const std::vector<SharedStorage>&);
    NeuralNetwork::forward_func fun_forward;
    NeuralNetwork::backward_func fun_backward;
    std::vector<Layer*> layers;
    void create_loss(const std::string& s);
    void forward_gpu(std::vector<SharedStorage>&);
    void forward_cpu(std::vector<SharedStorage>&);
    void backward_cpu(std::vector<SharedStorage>&,
                      const std::vector<SharedStorage>&);
    void backward_gpu(std::vector<SharedStorage>&,
                      const std::vector<SharedStorage>&);
};
#endif
