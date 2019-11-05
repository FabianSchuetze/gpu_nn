cmake_minimum_required(VERSION 3.5)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Produce compile_commands.json


project(GPU_NN)

#include_directories(include)
#include_directories(include/cuda)

# removed a space and added C++11 functionality (note the double '--' for NVCC)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode
    #arch=compute_35,code=sm_35;--std=c++11 -lcublas)
##set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lcublas)
set(CUDA_HOST_COMPILER nvcc) # shouldn't fail anymore

find_package(CUDA REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS})

CUDA_ADD_LIBRARY(cuda_obj STATIC src/cuda_math.cu) # works for me now

# moved the CXX flags to after the cuda_add_library call
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -L /usr/lib/cuda/lib64 -lcudart -lcublas")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC  -lcublas -O3 -Wall -std=c++17")
add_executable(main src/layer/dense.cpp src/math.cpp src/storage.cpp 
    src/layer/softmax.cpp test/forward.cpp) # only used one c++ class for this test

target_link_libraries(main ${CUDA_LIBRARIES})
target_link_libraries(main cuda_obj) # changed ${cuda_obj} to cuda_obj

install(TARGETS main DESTINATION ../bin)
