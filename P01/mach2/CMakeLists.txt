add_executable(mach2 mach2.cpp)
target_link_libraries(mach2 mach)
target_link_libraries(mach2 ${MPI_CXX_LIBRARIES})
target_compile_options(mach2 PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
target_compile_definitions(mach2 PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
set_target_properties(mach2 PROPERTIES LINK_FLAGS ${MPI_CXX_LINK_FLAGS})