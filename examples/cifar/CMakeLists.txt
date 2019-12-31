set(${LIBRARY_TARGET_NAME}_TEST
    cifar10.cpp
    create_data.cpp
)
foreach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST} )
    #string( REPLACE "test/cifar/" "" testname ${testsourcefile})
    string( REPLACE ".cpp" "" testname ${testsourcefile} )
    add_executable( ${testname} ${testsourcefile} )
    target_link_libraries(${testname} GPU_NN ${CUDA_LIBRARIES})
    target_link_libraries(${testname} GPU_NN cuda_obj)
    add_test(${testname} ${testname})
endforeach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST})
