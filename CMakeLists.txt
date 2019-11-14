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
    src/loss/cross_entropy.cpp
    src/layer/input.cpp
    src/network.cpp
    src/train.cpp
    src/gradient_descent/gradient_descent.cpp
    src/gradient_descent/sgd.cpp
)
if (UNIX)
  add_library(${LIBRARY_TARGET_NAME} SHARED ${${LIBRARY_TARGET_NAME}_SRC})
endif(UNIX)

# moved the CXX flags to after the cuda_add_library call
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -L /usr/lib/cuda/lib64 -lcudart  -lcublas -O3 -Wall -std=c++17")
add_executable(test_softmax test/softmax.cpp)
add_executable(test_dense test/dense.cpp)
add_executable(test_relu test/relu.cpp)
add_executable(test_cross_entropy test/cross_entropy.cpp)
add_executable(test_forward test/forward.cpp)
add_executable(test_backward test/backward.cpp)
add_executable(test_sgd test/sgd.cpp)
add_executable(test_mnist test/mnist.cpp)

target_link_libraries(test_softmax GPU_NN ${CUDA_LIBRARIES})
target_link_libraries(test_dense GPU_NN ${CUDA_LIBRARIES})
target_link_libraries(test_relu GPU_NN ${CUDA_LIBRARIES})
target_link_libraries(test_cross_entropy GPU_NN ${CUDA_LIBRARIES})
target_link_libraries(test_forward GPU_NN ${CUDA_LIBRARIES})
target_link_libraries(test_backward GPU_NN ${CUDA_LIBRARIES})
target_link_libraries(test_sgd GPU_NN ${CUDA_LIBRARIES})
target_link_libraries(test_mnist GPU_NN ${CUDA_LIBRARIES})
target_link_libraries(test_softmax GPU_NN cuda_obj)
target_link_libraries(test_dense GPU_NN cuda_obj)
target_link_libraries(test_relu GPU_NN cuda_obj)
target_link_libraries(test_cross_entropy GPU_NN cuda_obj)
target_link_libraries(test_forward GPU_NN cuda_obj)
target_link_libraries(test_backward GPU_NN cuda_obj)
target_link_libraries(test_sgd GPU_NN cuda_obj)
target_link_libraries(test_mnist GPU_NN cuda_obj)

enable_testing()
add_test(test_softmax test_softmax)
add_test(test_dense test_dense)
add_test(test_relu test_relu)
add_test(test_cross_entropy test_cross_entropy)
add_test(test_forward test_forward)
add_test(test_backward test_backward)
add_test(test_sgd test_sgd)
add_test(test_mnist test_mnist)
add_subdirectory(third_party/pybind11)
pybind11_add_module(NeuralNetwork python/bindings.cpp)

target_link_libraries(NeuralNetwork PRIVATE GPU_NN ${CUDA_LIBRARIES})
target_link_libraries(NeuralNetwork PRIVATE GPU_NN cuda_obj)
