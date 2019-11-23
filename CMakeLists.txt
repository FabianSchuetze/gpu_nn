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
    src/network.cpp
    src/train.cpp
    src/gradient_descent/gradient_descent.cpp
    src/gradient_descent/sgd.cpp
    src/trainArgs.cpp
    src/layer/convolution.cpp
)
if (UNIX)
  add_library(${LIBRARY_TARGET_NAME} SHARED ${${LIBRARY_TARGET_NAME}_SRC})
endif(UNIX)

# moved the CXX flags to after the cuda_add_library call
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -fPIC -lcudnn -lcurand -L /usr/lib/cuda/lib64 -lcudart  -lcublas -Wall -std=c++17")
set(${LIBRARY_TARGET_NAME}_TEST
    test/softmax.cpp
    test/dense.cpp
    test/cross_entropy.cpp
    test/forward.cpp
    test/backward.cpp
    test/sgd.cpp
    test/dropout.cpp
    test/mnist.cpp
)

enable_testing()
foreach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST} )
    string( REPLACE "test/" "" testname ${testsourcefile})
    #string( REPLACE ".cpp" "" testname ${testsourcefile} )
    add_executable( ${testname} ${testsourcefile} )
    target_link_libraries(${testname} GPU_NN ${CUDA_LIBRARIES})
    target_link_libraries(${testname} GPU_NN cuda_obj)
    add_test(${testname} ${testname})
endforeach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST})
#add_subdirectory(third_party/pybind11)
#pybind11_add_module(NeuralNetwork python/bindings.cpp)

#target_link_libraries(NeuralNetwork PRIVATE GPU_NN ${CUDA_LIBRARIES})
#target_link_libraries(NeuralNetwork PRIVATE GPU_NN cuda_obj)
