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
    src/gradient_descent/adagrad.cpp
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
set(${LIBRARY_TARGET_NAME}_TEST
    #test/normalization.cpp
    test/mnist_lenet.cpp
    test/mnist.cpp
    test/lstm.cpp
    #test/inits.cpp
    #test/im2col_layer.cpp
    #test/convolution.cpp
    #test/softmax.cpp
    #test/dense.cpp
    #test/cross_entropy.cpp
    #test/forward.cpp
    #test/backward.cpp
    #test/sgd.cpp
    #test/dropout.cpp
    #test/pooling.cpp
    #test/momentum.cpp
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

set(${LIBRARY_TARGET_NAME}_TEST
    test/cifar/cifar10.cpp
    test/cifar/create_data.cpp
)
foreach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST} )
    string( REPLACE "test/cifar/" "" testname ${testsourcefile})
    #string( REPLACE ".cpp" "" testname ${testsourcefile} )
    add_executable( ${testname} ${testsourcefile} )
    target_link_libraries(${testname} GPU_NN ${CUDA_LIBRARIES})
    target_link_libraries(${testname} GPU_NN cuda_obj)
    add_test(${testname} ${testname})
endforeach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST})

set(${LIBRARY_TARGET_NAME}_TEST2
    test/rnn/rnn.cpp
)
foreach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST2} )
    string( REPLACE "test/rnn/" "" testname ${testsourcefile})
    #string( REPLACE ".cpp" "" testname ${testsourcefile} )
    add_executable( ${testname} ${testsourcefile} )
    target_link_libraries(${testname} GPU_NN ${CUDA_LIBRARIES})
    target_link_libraries(${testname} GPU_NN cuda_obj)
    add_test(${testname} ${testname})
endforeach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST2})
#add_subdirectory(third_party/pybind11)
#pybind11_add_module(NeuralNetwork python/bindings.cpp)

#target_link_libraries(NeuralNetwork PRIVATE GPU_NN ${CUDA_LIBRARIES})
#target_link_libraries(NeuralNetwork PRIVATE GPU_NN cuda_obj)
