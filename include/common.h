#ifndef common_h
#define common_h
#include <eigen-git-mirror/Eigen/Dense>
#include <fstream>
#include <iomanip>

typedef float dtype;
typedef Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Vector<dtype, Eigen::Dynamic> Vector;

//void print_Matrix_to_stdout(const Eigen::MatrixXd& val, std::string loc) {
    //int rows(val.rows()), cols(val.cols());
    //std::ofstream myfile(loc);
    //myfile << "dimensions: rows, cols: " << rows << ", " << cols << std::endl;
    //myfile << std::fixed;
    //myfile << std::setprecision(2);
    //for (int row = 0; row < rows; ++row) {
        //myfile << val(row, 0);
        //for (int col = 1; col < cols; ++col) {
            //myfile << ", " << val(row, col);
        //}
        //myfile << std::endl;
    //}
//}

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
