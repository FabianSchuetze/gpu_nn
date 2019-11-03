//#pragma once
#ifndef math_h
#define math_h
#include <memory>
#include "cublas_v2.h"
#include "storage.h"

typedef std::shared_ptr<Storage> SharedStorage;
void multonGPU(cublasHandle_t, const SharedStorage&, const SharedStorage&,
               SharedStorage&, double, double);
#endif
