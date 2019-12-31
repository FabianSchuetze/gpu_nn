
set(${LIBRARY_TARGET_NAME}_TEST
    #normalization.cpp
    #mnist_lenet.cpp
    #mnist.cpp
    #lstm.cpp
    #inits.cpp
    #im2col_layer.cpp
    #convolution.cpp
    #softmax.cpp
    dense.cpp
    cross_entropy.cpp
    forward.cpp
    backward.cpp
    sgd.cpp
    dropout.cpp
    pooling.cpp
    momentum.cpp
)

enable_testing()
foreach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST} )
    #string( REPLACE "test/" "" testname ${testsourcefile})
    string( REPLACE ".cpp" "" testname ${testsourcefile} )
    add_executable( ${testname} ${testsourcefile} )
    target_link_libraries(${testname} GPU_NN ${CUDA_LIBRARIES})
    target_link_libraries(${testname} GPU_NN cuda_obj)
    add_test(${testname} ${testname})
endforeach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST})
