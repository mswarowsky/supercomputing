

add_executable(zeta1 zeta1.cpp)
target_link_libraries(zeta1 zeta)
target_link_libraries(zeta1 ${MPI_CXX_LIBRARIES})
target_compile_options(zeta1 PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
target_compile_definitions(zeta1 PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
set_target_properties(zeta1 PROPERTIES LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
