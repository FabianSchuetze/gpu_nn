#ifndef common_h
#define common_h
#include <eigen-git-mirror/Eigen/Core>
#include <fstream>
#include <iomanip>
#include <utility>
//#include <iostream>

typedef float dtype;
typedef Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Vector<dtype, Eigen::Dynamic> Vector;

template <typename T, typename Paramter>
class NamedType {
   public:
    explicit NamedType(T const& value) : value_(value) {}
    explicit NamedType(T&& value) : value_(std::move(value)) {}
    T& get() { return value_; }
    T const& get() const { return value_; }

   private:
    T value_;
};

template <typename T, typename Paramter>
class NamedPair {
   public:
    explicit NamedPair(T const v1, T const v2) : value_() {
        value_ = std::make_pair(v1, v2);
    }
    std::pair<T, T>& get() { return value_; }
    T& first() const { return const_cast<T&>(value_.first); }
    T& second()  const { return const_cast<T&> (value_.second); }
    std::pair<T, T> const& get() const { return value_; }

   private:
    std::pair<T, T> value_;
};

template <typename T1, typename T2, typename Paramter>
class NamedPairDiff {
   public:
    explicit NamedPairDiff(T1 const v1, T2 const v2) : value_() {
        value_ = std::make_pair(v1, v2);
    }
    std::pair<T1, T2>& get() { return value_; }
    T1& first() const { return const_cast<T1&>(value_.first); }
    T2& second()  const { return const_cast<T2&> (value_.second); }
    std::pair<T1, T2> const& get() const { return value_; }

   private:
    std::pair<T1, T2> value_;
};

struct ShuffleParameter {};
using Shuffle = NamedType<bool, ShuffleParameter>;
struct MomentumRateParameter {};
using MomentumRate = NamedType<dtype, MomentumRateParameter>;
struct FeaturesParameter {};
using Features = NamedType<int, FeaturesParameter>;
struct LearningRateParameter {};
using LearningRate = NamedType<dtype, LearningRateParameter>;
struct WeightDecayParameter {};
using WeightDecay = NamedType<dtype, WeightDecayParameter>;
struct DecayRateParameter {};
using DecayRate = NamedType<dtype, DecayRateParameter>;
struct LearingRateDecayParameter {};
using LearingRateDecay = NamedPairDiff<int, dtype, LearingRateDecayParameter>;
struct EpochParamter {};
using Epochs = NamedType<int, EpochParamter>;
struct PatienceParamter {};
using Patience = NamedType<int, PatienceParamter>;
struct BatchSizeParamter {};
using BatchSize = NamedType<int, BatchSizeParamter>;
struct PadParameter {};
using Pad = NamedType<int, PadParameter>;
struct StrideParameter {};
using Stride = NamedType<int, StrideParameter>;
struct FiltersParameter {};
using Filters = NamedType<int, FiltersParameter>;
struct ChannelsParameter {};
using Channels = NamedType<int, ChannelsParameter>;
struct WindowParameter {};
using Window = NamedType<int, WindowParameter>;
struct FilterShapeParameter {};
using FilterShape = NamedPair<int, FilterShapeParameter>;
struct ImageShapeParameter {};
using ImageShape = NamedPair<int, ImageShapeParameter>;

// void print_Matrix_to_stdout(const Eigen::MatrixXd& val, std::string loc) {
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

// CLEAR THE NAMESPACE MESSS
// namespace CUDA_CHECKS{
#define MY_CHECK(call)                                             \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess) {                                \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
        }                                                          \
    }

#define CHECK_CUBLAS(call)                                                   \
    {                                                                        \
        cublasStatus_t err;                                                  \
        if ((err = (call)) != CUBLAS_STATUS_SUCCESS) {                       \
            fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__, \
                    __LINE__);                                               \
            exit(1);                                                         \
        }                                                                    \
    }

#define CHECK_CURAND(call)                                                   \
    {                                                                        \
        curandStatus_t err;                                                  \
        if ((err = (call)) != CURAND_STATUS_SUCCESS) {                       \
            fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__, \
                    __LINE__);                                               \
            exit(1);                                                         \
        }                                                                    \
    }

//#define CHECK_CUDNN(call)                                                   \
    //{                                                                       \
        //cudnnStatus_t err;                                                  \
        //if ((err = (call)) != CUDNN_STATUS_SUCCESS) {                       \
            //fprintf(stderr, "Got CUDNN error %d at %s:%d\n", err, __FILE__, \
                    //__LINE__);                                              \
            //exit(1);                                                        \
        //}                                                                   \
    //}
#define CHECK_CUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }
#endif

