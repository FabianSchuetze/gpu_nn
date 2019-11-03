//#pragma once
#ifndef math_h
#define math_h
#include <memory>
#include "cublas_v2.h"
#include "storage.h"

typedef std::shared_ptr<Storage> SharedStorage;
void my_Dgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t,
              const SharedStorage&, const SharedStorage&, SharedStorage&,
              double, double);
void my_Dgemv(cublasHandle_t, cublasOperation_t, const SharedStorage&,
              const SharedStorage&, SharedStorage&, double, double);
void my_add_vec_to_mat_colwise(SharedStorage&, const SharedStorage&);
#endif
