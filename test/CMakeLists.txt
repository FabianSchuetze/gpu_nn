
set(${LIBRARY_TARGET_NAME}_TEST
    #test/normalization.cpp
    #test/mnist_lenet.cpp
    #test/mnist.cpp
    #test/lstm.cpp
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
