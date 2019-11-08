//#pragma once
#ifndef math_h
#define math_h
#include <memory>
#include "common.h"
#include "cublas_v2.h"
#include "storage.h"

typedef std::shared_ptr<Storage> SharedStorage;
void my_Dgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t,
              const SharedStorage&, const SharedStorage&, SharedStorage&, dtype,
              dtype);
void my_Dgemv(cublasHandle_t, cublasOperation_t, const SharedStorage&,
              const SharedStorage&, SharedStorage&, dtype, dtype);
void my_add_vec_to_mat_colwise(SharedStorage&, const SharedStorage&, dtype);
void my_add_vec_to_mat_colwise(const SharedStorage&, const SharedStorage&,
                               SharedStorage&, dtype);
void my_Exponential(SharedStorage&);
void my_Divide_colwise(SharedStorage&, const SharedStorage&);
void my_relu(SharedStorage&, const SharedStorage&);
void my_relu_backwards(const SharedStorage&, const SharedStorage&,
                       SharedStorage&);
void my_cross_entropy_loss(dtype&, const SharedStorage&, const SharedStorage&);
void my_cross_entropy_gradient(SharedStorage&, const SharedStorage&,
                               const SharedStorage);
#endif
