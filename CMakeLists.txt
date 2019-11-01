#cmake_minimum_required (VERSION 3.5)

#project (gpu_NN VERSION 0.1)
#set(LIBRARY_TARGET_NAME ${PROJECT_NAME})
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Produce compile_commands.json
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS " -std=c++17 -O3 -march=native -Wall -Wextra")

#all: program

#program: mult.o
	#clang++ -o program -fPIC -L /usr/lib/cuda/lib64 -lcudart -lcublas test/copy_data.cpp src/storage.cpp src/math.cpp  mult.o -O3

#mult.o:
	#nvcc -c -o mult.o src/cuda_math.cu #-c: creates an object file of sum.cu, link then

#clean: rm -f *.o program

cmake_minimum_required(VERSION 3.5)

project(pupil_tracker)

#include_directories(include)
#include_directories(include/cuda)

# removed a space and added C++11 functionality (note the double '--' for NVCC)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode
    #arch=compute_35,code=sm_35;--std=c++11 -lcublas)
##set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lcublas)
set(CUDA_HOST_COMPILER nvcc) # shouldn't fail anymore

# didn't test with Boost
# find_package(Boost COMPONENTS system filesystem REQUIRED)
# include_directories(${Boost_INCLUDE_DIR})

find_package(CUDA REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS})

CUDA_ADD_LIBRARY(cuda_obj STATIC src/cuda_math.cu) # works for me now

# moved the CXX flags to after the cuda_add_library call
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -L /usr/lib/cuda/lib64 -lcudart -lcublas")
add_executable(main src/math.cpp src/storage.cpp test/copy_data.cpp) # only used one c++ class for this test

# target_link_libraries(main ${Boost_LIBRARIES})
target_link_libraries(main ${CUDA_LIBRARIES})
# target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main cuda_obj) # changed ${cuda_obj} to cuda_obj

install(TARGETS main DESTINATION ../bin)
