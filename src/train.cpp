#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include "../include/loss/cross_entropy.h"
#include "../include/network.h"
#include "../include/threadsafe_queue.hpp"

using Eigen::all;
using std::make_shared;
using std::shared_ptr;
using std::vector;
// void print_Matrix_to_stdout2(const Matrix& val, std::string loc) {
// int rows(val.rows()), cols(val.cols());
// std::ofstream myfile(loc);
// myfile << "dimensions: rows, cols: " << rows << ", " << cols << std::endl;
// myfile << std::fixed;
// myfile << std::setprecision(2);
// for (int row = 0; row < rows; ++row) {
// myfile << val(row, 0);
// for (int col = 1; col < cols; ++col) {
// myfile << ", " << val(row, col);
//}
// myfile << std::endl;
//}
//}

void NeuralNetwork::backwards(std::vector<SharedStorage>& gradients,
                              const std::vector<SharedStorage>& values,
                              DebugInfo& debug) {
    (this->*fun_backward)(gradients, values, debug);
}

// void NeuralNetwork::backward_debug_info(const vector<SharedStorage>& grad) {
// train_args->bwd_stream() << grad[0]->return_data_const().mean();
// for (size_t i = 1; i < grad.size(); ++i) {
// train_args->bwd_stream() << " ";
// train_args->bwd_stream() << grad[i]->return_data_const().mean();
// train_args->bwd_stream() << " ";
// train_args->bwd_stream() << grad[i]->return_data_const().lpNorm<1>() /
// train_args->batch_size();
// for (const SharedStorage& para : layers[i]->return_gradients()) {
// train_args->bwd_stream() << " ";
// train_args->bwd_stream() << para->return_data_const().mean();
// train_args->bwd_stream() << " ";
// train_args->bwd_stream() << para->return_data_const().lpNorm<1>();
//}
//}
// train_args->bwd_stream() << "\n";
//}

void NeuralNetwork::backward_cpu(std::vector<SharedStorage>& gradients,
                                 const std::vector<SharedStorage>& values,
                                 DebugInfo& debug) {
    int idx = gradients.size() - 1;
    for (int i = layers.size() - 2; i > 0; i--) {
        const SharedStorage& gradient_in = gradients[idx];
        SharedStorage& gradient_out = gradients[idx - 1];
        const SharedStorage& vals = values[idx - 1];
        layers[i]->backward_cpu(vals, gradient_in, gradient_out);
        idx--;
    }
    if (debug.is_set()) debug.backward_debug_info(gradients, layers, 32);
}

void NeuralNetwork::backward_gpu(vector<SharedStorage>& gradients,
                                 const vector<SharedStorage>& values,
                                 DebugInfo& debug) {
    int idx = gradients.size() - 1;
    for (int i = layers.size() - 2; i > 0; i--) {
        const SharedStorage& gradient_in = gradients[idx];
        SharedStorage& gradient_out = gradients[idx - 1];
        const SharedStorage& vals = values[idx - 1];
        layers[i]->backward_gpu(vals, gradient_in, gradient_out);
        idx--;
    }
    if (debug.is_set()) debug.backward_debug_info(gradients, layers, 32);
}

void NeuralNetwork::update_weights(std::shared_ptr<GradientDescent>& opt,
                                   vector<VecSharedStorage>& helper,
                                   int batch_size) {
    (this->*fun_update)(opt, helper, batch_size);
}

void NeuralNetwork::update_weights_cpu(std::shared_ptr<GradientDescent>& opt,
                                       vector<VecSharedStorage>& helpers,
                                       int batch_size) {
    // int i = helpers.size() - 1;
    int i = 0;
    for (std::shared_ptr<Layer> layer : layers) {
        if (layer->n_paras() > 0) {
            vector<SharedStorage> parameters = layer->return_parameters();
            const vector<SharedStorage>& gradients = layer->return_gradients();
            VecSharedStorage& helper = helpers[i++];
            opt->weight_update_cpu(gradients, parameters, batch_size, helper);
            //// WHEN CALCULATING GRADIENTS IN BATCHES THAT"S NOT NEEDED, MIGHT
            // BE DANGEROUS!!!
        }
    }
}

void NeuralNetwork::update_weights_gpu(std::shared_ptr<GradientDescent>& opt,
                                       vector<VecSharedStorage>& helpers,
                                       int batch_size) {
    int i = 0;
    for (std::shared_ptr<Layer> layer : layers) {
        if (layer->n_paras() > 0) {
            vector<SharedStorage> parameters = layer->return_parameters();
            const vector<SharedStorage>& gradients = layer->return_gradients();
            VecSharedStorage& helper = helpers[i++];
            opt->weight_update_gpu(gradients, parameters, batch_size, helper);
            // layer->clear_gradients_cpu();
        }
    }
}

void NeuralNetwork::prepare_subset(const vector<int>& pop, vector<int>& subset,
                                   int& start, const int& n_samples) {
    int i = 0;
    while (start + i < start + n_samples) {
        subset[i] = pop[start + i];
        i++;
    }
    start += n_samples;
}

void NeuralNetwork::maybe_shuffle(vector<int>& pop, int& start, const int& end,
                                  const int& step, std::mt19937& gen) {
    if (start + step >= end) {
        std::shuffle(pop.begin(), pop.end(), gen);
        start = 0;
    }
}

void NeuralNetwork::random_numbers(vector<int>& samples, std::mt19937& gen) {
    if (!train_args) throw std::runtime_error("Train args is not set");
    std::uniform_int_distribution<> uniform(0,
                                            train_args->x_train().rows() - 1);
    for (size_t i = 0; i < samples.size(); i++) samples[i] = uniform(gen);
}

void NeuralNetwork::get_new_sample(const vector<int>& samples, Matrix& x_train,
                                   Matrix& y_train) {
    if (!train_args) throw std::runtime_error("Train args is not set");
    x_train = train_args->x_train()(samples, all).transpose();
    y_train = train_args->y_train()(samples, all).transpose();
}

int NeuralNetwork::check_input_dimension(const std::vector<int>& dim) {
    int i = 1;
    for (int shape : dim) i *= shape;
    return i;
}

// void NeuralNetwork::print_layers(std::ofstream& stream) {
// stream << layers[0]->name();
// for (size_t i = 1; i < layers.size(); ++i) {
// stream << " ";
// stream << layers[i]->name() << "_mean_";
// stream << " ";
// stream << layers[i]->name() << "_l1_";
// for (size_t j = 0; j < layers[i]->return_parameters().size(); ++j) {
// stream << " ";
// stream << layers[i]->name() << "_param_mena" << j;
// stream << " ";
// stream << layers[i]->name() << "_param_l1" << j;
//}
//}
// stream << "\n";
//}

void NeuralNetwork::train(const Matrix& features, const Matrix& targets,
                          std::shared_ptr<GradientDescent>& sgd, Epochs _epoch,
                          Patience _patience, BatchSize _batch_size,
                          DebugInfo&& debug_info) {
    std::vector<int> input_dim = layers[0]->output_dimension();
    int expected_cols = check_input_dimension(input_dim);
    if (expected_cols != features.cols()) {
        std::stringstream ss;
        ss << "The number of input features is: " << features.cols()
           << " but the input layer expects: ";
        std::copy(input_dim.begin(), input_dim.end(),
                  std::ostream_iterator<int>(ss, " "));
        ss << "in:\n"
           << __PRETTY_FUNCTION__ << "\ncalled from " << __FILE__ << " at "
           << __LINE__;
        throw std::invalid_argument(ss.str());
    }
    train_args =
        std::make_unique<trainArgs>(features, targets, _epoch, _patience,
                                    _batch_size, sgd, layers);
    if (debug_info.is_set()) {
        debug_info.print_layers(layers);
        // print_layers(train_args->ffw_stream());
        // print_layers(train_args->bwd_stream());
    }
    // train(sgd);
    std::thread produce([&]() { producer(); });
    std::thread consume([&]() { consumer(sgd, debug_info); });
    produce.join();
    consume.join();
}

void NeuralNetwork::producer() {
    std::mt19937 gen;
    // gen.seed(10);
    Matrix x_train, y_train;
    vector<int> samples(train_args->batch_size());
    vector<int> population(train_args->x_train().rows());
    std::iota(population.begin(), population.end(), 0);
    std::pair<SharedStorage, SharedStorage> data;
    int producer_idx(0);
    int end(train_args->x_train().rows());
    std::shuffle(population.begin(), population.end(), gen);
    while (train_args->current_epoch() < train_args->epochs()) {
        if (train_args->data_queue.size() < 5) {
            maybe_shuffle(population, producer_idx, end,
                          train_args->batch_size(), gen);
            prepare_subset(population, samples, producer_idx,
                           train_args->batch_size());
            get_new_sample(samples, x_train, y_train);
            shared_ptr<Storage> SharedInput = make_shared<Storage>(x_train);
            shared_ptr<Storage> SharedTarget = make_shared<Storage>(y_train);
            data = std::make_pair(SharedInput, SharedTarget);
            train_args->data_queue.push(data);
        }
    }
}

dtype NeuralNetwork::validate(std::chrono::milliseconds diff) {
    dtype total_loss(0.);
    DebugInfo no_debugging("", "");
    int out_size = layers[layers.size() - 1]->output_dimension()[0];
    Matrix output = Matrix::Zero(out_size, train_args->x_val().rows());
    SharedStorage SharedPred = std::make_shared<Storage>(output);
    predict(train_args->x_val(), SharedPred, no_debugging);
    total_loss = loss->loss(SharedPred, train_args->y_val_shared());
    size_t obs = train_args->x_val().rows();
    std::cout << "after iter " << train_args->current_epoch() << "the loss is "
              << total_loss / obs << ", in " << diff.count() << " milliseconds"
              << std::endl;
    train_args->advance_epoch();
    train_args->reset_total_iter();
    return total_loss;
}

void NeuralNetwork::display_train_loss(dtype& train_loss) {
    if (not(train_args->total_iter() % (train_args->batch_size() * 500))) {
        dtype rel_loss = train_loss / (500 * train_args->batch_size());
        train_args->advance_cumulative_iter(500);
        std::cout << "train loss after " << train_args->cummulative_iter()
                  << " iter: " << rel_loss << std::endl;
        train_loss = 0;
    }
}

void NeuralNetwork::consumer(std::shared_ptr<GradientDescent>& sgd, DebugInfo& debug) {
    vector<SharedStorage> vals = allocate_forward(train_args->batch_size());
    vector<SharedStorage> grads = allocate_backward(train_args->batch_size());
    auto begin = std::chrono::system_clock::now();
    const std::string type("train");
    std::chrono::milliseconds diff;
    dtype train_loss(0.);
    while (train_args->current_epoch() < train_args->epochs()) {
        std::shared_ptr<std::pair<SharedStorage, SharedStorage>> out =
            train_args->data_queue.wait_and_pop();
        vals[0] = out->first;
        forward(vals, type, debug);
        loss->grad_loss(grads.back(), vals.back(), out->second, out->second);
        backwards(grads, vals, debug);
        train_loss += loss->loss(vals.back(), out->second);
        update_weights(sgd, train_args->optimizer(), train_args->batch_size());
        train_args->advance_total_iter();
        display_train_loss(train_loss);
        if (train_args->total_iter() > train_args->max_total_iter()) {
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now() - begin);
            validate(diff);
            begin = std::chrono::system_clock::now();
        }
    }
}

// void NeuralNetwork::train(std::shared_ptr<GradientDescent>& sgd) {
// std::mt19937 gen;
// gen.seed(0);
// vector<SharedStorage> vals = allocate_forward(train_args->batch_size());
// vector<SharedStorage> grads = allocate_backward(train_args->batch_size());
// Matrix tmp =
// Matrix::Zero(train_args->y_train().cols(), train_args->batch_size());
// SharedStorage SharedTarget = std::make_shared<Storage>(tmp);
// vector<int> samples(train_args->batch_size());
// Matrix x_train, y_train;
// auto begin = std::chrono::system_clock::now();
// auto begin_fill = std::chrono::system_clock::now();
// auto begin_other = std::chrono::system_clock::now();
// auto end = std::chrono::system_clock::now();
//// auto end_fill = std::chrono::system_clock::now();
// std::chrono::milliseconds diff;
// std::chrono::nanoseconds diff_2;
// std::chrono::nanoseconds diff_3;
// while (train_args->current_epoch() < train_args->epochs()) {
// begin_fill = std::chrono::system_clock::now();
// random_numbers(samples, gen);
// get_new_sample(samples, x_train, y_train);
// SharedTarget->update_cpu_data(y_train);
// fill_hiddens(vals, x_train);
// diff_2 += std::chrono::duration_cast<std::chrono::nanoseconds>(
// std::chrono::system_clock::now() - begin_fill);
// begin_other = std::chrono::system_clock::now();
// forward(vals, "train");
//// SharedStorage& grad_in = grads[grads.size() - 1];
// loss->grad_loss(grads.back(), vals.back(), SharedTarget, SharedTarget);
// backwards(grads, vals);
// update_weights(sgd, train_args->optimizer(), train_args->batch_size());
// train_args->advance_total_iter();
// diff_3 += std::chrono::duration_cast<std::chrono::nanoseconds>(
// std::chrono::system_clock::now() - begin_other);
// if (train_args->total_iter() > train_args->max_total_iter()) {
// end = std::chrono::system_clock::now();
// diff = std::chrono::duration_cast<std::chrono::milliseconds>(end -
// begin);
// std::cout << "the input used " << diff_2.count() << " milliseconds"
//<< std::endl;
// std::cout << "the other part used " << diff_3.count()
//<< " milliseconds" << std::endl;
// diff_2 = std::chrono::nanoseconds::zero();
// diff_3 = std::chrono::nanoseconds::zero();
// validate(diff);
// begin = std::chrono::system_clock::now();
//}
//}
//}
