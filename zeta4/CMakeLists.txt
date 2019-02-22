add_executable(zeta4 zeta4.cpp)
target_link_libraries(zeta4 zeta)
# Add MPI stuff
target_link_libraries(zeta4 ${MPI_CXX_LIBRARIES})
target_compile_options(zeta4 PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
target_compile_definitions(zeta4 PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
set_target_properties(zeta4 PROPERTIES LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
#Add OpenMP stuff
target_link_libraries(zeta4 ${OpenMP_CXX_LIBRARIES})
target_compile_options(zeta4 PRIVATE ${OpenMP_CXX_COMPILE_OPTIONS})
target_compile_definitions(zeta4 PRIVATE ${OpenMP_CXX_COMPILE_DEFINITIONS})