# real MPI use


add_executable(zeta2 zeta2.cpp)
target_link_libraries(zeta2 zeta)
target_link_libraries(zeta2 ${MPI_CXX_LIBRARIES})