#ifndef network_h
#define network_h
#include "layer/layer.h"
#include "loss/loss.h"
#include <vector>
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
    //function allocates shared storage;
    Matrix predict(const Matrix&);
    //@brief Returns a prediction from the neural network, calling this
    //function presumes that the SharedStorage is appropriate for all the 
    //layers
    SharedStorage predict(std::vector<SharedStorage>&);
    private:
    typedef void(NeuralNetwork::*forward_func)(std::vector<SharedStorage>&);
    NeuralNetwork::forward_func fun_forward;
    std::vector<Layer*> layers;
    std::shared_ptr<Loss> loss;
    void create_loss(const std::string& s);
    std::vector<SharedStorage> allocate_shared_storage(int);
    void forward(std::vector<SharedStorage>&);
    void forward_gpu(std::vector<SharedStorage>&);
    void forward_cpu(std::vector<SharedStorage>&);
    void fill_hiddens(std::vector<SharedStorage>&, const Matrix&);
};
#endif
