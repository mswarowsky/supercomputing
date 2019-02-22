

add_executable(mach1 mach1.cpp)
target_link_libraries(mach1 mach)
target_link_libraries(mach1 ${MPI_CXX_LIBRARIES})
target_compile_options(mach1 PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
target_compile_definitions(mach1 PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
set_target_properties(mach1 PROPERTIES LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
