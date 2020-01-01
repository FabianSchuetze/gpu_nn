cmake_minimum_required(VERSION 3.5)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Produce compile_commands.json

project(GPU_NN LANGUAGES CUDA CXX)
set(LIBRARY_TARGET_NAME ${PROJECT_NAME})

set(CUDA_HOST_COMPILER nvcc) # shouldn't fail anymore
#find_package(CUDA 9.0 REQUIRED)
enable_language(CUDA)
ADD_LIBRARY(cuda_obj SHARED src/cuda_math.cu) # works for me now

set(${LIBRARY_TARGET_NAME}_SRC
    src/layer/dense.cpp
    src/loss/loss.cpp
    src/math.cpp
    src/storage.cpp
    src/layer/layer.cpp
    src/layer/softmax.cpp
    src/layer/relu.cpp
    src/layer/dropout.cpp
    src/loss/cross_entropy.cpp
    src/layer/input.cpp
    src/layer/lstm.cpp
    src/network.cpp
    src/train.cpp
    src/gradient_descent/gradient_descent.cpp
    src/gradient_descent/sgd.cpp
    src/gradient_descent/rmsprop.cpp
    src/trainArgs.cpp
    src/layer/convolution.cpp
    src/layer/pooling.cpp
    src/layer/im2col_layer.cpp
    src/utils/standard_normalization.cpp
    src/utils/zca_scaler.cpp
    src/utils/global_contrast_normalization.cpp
    src/gradient_descent/momentum.cpp
    src/initalization/normal.cpp
    src/initalization/glorot.cpp
    src/initalization/lcn.cpp
    src/debug_info.cpp
    src/metrics/char_rnn.cpp
    src/metrics/missclassified.cpp
)
if (UNIX)
  add_library(${LIBRARY_TARGET_NAME} SHARED ${${LIBRARY_TARGET_NAME}_SRC})
endif(UNIX)

# moved the CXX flags to after the cuda_add_library call
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -lopenblas\
    -Wunused -Wunused-member-function -Wunused-function\
    -Wunneeded-member-function\
    -lcudnn -lcurand -L /usr/lib/cuda/lib64 -lcudart  -lcublas\
    -L /usr/lib/x86_64-linux-gnu -lblas\
    -Wall -std=c++17 -Wno-error=unused-command-line-argument")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")

add_subdirectory(test)
add_subdirectory(examples/cifar)
add_subdirectory(examples/rnn)
add_subdirectory(examples/mnist)

#add_subdirectory(third_party/pybind11)
#pybind11_add_module(NeuralNetwork python/bindings.cpp)

#target_link_libraries(NeuralNetwork PRIVATE GPU_NN ${CUDA_LIBRARIES})
#target_link_libraries(NeuralNetwork PRIVATE GPU_NN cuda_obj)
