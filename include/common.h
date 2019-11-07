#ifndef common_h
#define common_h
#include <eigen-git-mirror/Eigen/Dense>
#include <eigen-git-mirror/Eigen/src/Core/util/Constants.h>
#include <fstream>
#include <iomanip>

typedef double dtype;
typedef Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Vector<dtype, Eigen::Dynamic> Vector;

// CLEAR THE NAMESPACE MESSS
//namespace CUDA_CHECKS{
#define MY_CHECK(call)                                                         \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}
//}
#endif
