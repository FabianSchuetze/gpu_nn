
set(${LIBRARY_TARGET_NAME}_TEST
    #normalization.cpp
    lstm.cpp
    inits.cpp
    convolution.cpp
    softmax.cpp
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
    string( REPLACE ".cpp" "" testname ${testsourcefile} )
    add_executable( ${testname} ${testsourcefile} )
    target_link_libraries(${testname} GPU_NN ${CUDA_LIBRARIES})
    target_link_libraries(${testname} GPU_NN cuda_obj)
    add_test(${testname} ${testname})
endforeach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST})
