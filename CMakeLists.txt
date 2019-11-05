cmake_minimum_required(VERSION 3.5)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Produce compile_commands.json

project(GPU_NN)
set(LIBRARY_TARGET_NAME ${PROJECT_NAME})

set(CUDA_HOST_COMPILER nvcc) # shouldn't fail anymore
find_package(CUDA REQUIRED)
CUDA_ADD_LIBRARY(cuda_obj STATIC src/cuda_math.cu) # works for me now

set(${LIBRARY_TARGET_NAME}_SRC
    src/layer/dense.cpp
    src/math.cpp
    src/storage.cpp
    src/layer/softmax.cpp
)
if (UNIX)
  add_library(${LIBRARY_TARGET_NAME} SHARED ${${LIBRARY_TARGET_NAME}_SRC})
endif(UNIX)

# moved the CXX flags to after the cuda_add_library call
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -L /usr/lib/cuda/lib64 -lcudart  -lcublas -O3 -Wall -std=c++17")
add_executable(main test/softmax.cpp) # only used one c++ class for this test
add_executable(main2 test/forward.cpp) # only used one c++ class for this test

target_link_libraries(main GPU_NN ${CUDA_LIBRARIES})
target_link_libraries(main2 GPU_NN ${CUDA_LIBRARIES})
target_link_libraries(main GPU_NN cuda_obj) # changed ${cuda_obj} to cuda_obj
target_link_libraries(main2 GPU_NN cuda_obj) # changed ${cuda_obj} to cuda_obj

install(TARGETS main DESTINATION ../bin)
