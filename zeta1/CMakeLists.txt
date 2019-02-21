

add_executable(zeta1 zeta1.cpp)
target_link_libraries(zeta1 zeta)
target_link_libraries(zeta1 ${MPI_CXX_LIBRARIES})