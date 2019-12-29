#pragma once
#include <memory>
#ifndef lstm_hpp
#define lstm_hpp
#include "../initalization/init.hpp"
#include "cublas_v2.h"
#include "layer.h"
class LSTM : public Layer {
   public:
    LSTM(Features, Features, Init*);
    LSTM(Features, const std::shared_ptr<Layer>&, Init*);
    virtual ~LSTM() { CHECK_CUBLAS(cublasDestroy(_handle)); };
    void forward_gpu(const SharedStorage&, SharedStorage&,
                     const std::string&) override;
    void forward_cpu(const SharedStorage&, SharedStorage&,
                     const std::string&) override;
    void backward_gpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    void backward_cpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    VecSharedStorage return_parameters() override { return parameters; };
    VecSharedStorage return_gradients() override { return gradients; }
    VecSharedStorage return_parameters() const override { return parameters; };
    VecSharedStorage return_gradients() const override { return gradients; }

   private:
    void initialize_weight(Init*);
    void initialize_states();
    void clip_gradients();
    void maybe_resize_state(int);
    void expand_states(int);
    void reorganize_states(int);
    void construct_sigma_cpu();
    void construct_sigma_gpu();
    void initialize_grad();
    void initialize_output_dimension() override;
    void initialize_input_dimension(const std::shared_ptr<Layer>&);
    void resize_assistance(int);
    void nonlinear_transformations(int);
    void compute_next_state(int t);
    void multiply_one_col_fwd(const SharedStorage&, int);
    void multiply_one_col_bwd(SharedStorage&, int);
    void new_hidden_state(int, const SharedStorage&);
    void new_cell_state(int, const SharedStorage&);
    void internal_deriv(int);
    cublasHandle_t _handle;
    Features _out;
    Features _in;
    std::vector<SharedStorage> states;
    std::vector<SharedStorage> assistance_parameters;
};
#endif
