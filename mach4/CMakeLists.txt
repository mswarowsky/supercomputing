add_executable(mach4 mach4.cpp)
target_link_libraries(mach4 mach)
# Add MPI stuff
target_link_libraries(mach4 ${MPI_CXX_LIBRARIES})
target_compile_options(mach4 PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
target_compile_definitions(mach4 PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
set_target_properties(mach4 PROPERTIES LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
#Add OpenMP stuff
target_link_libraries(mach4 ${OpenMP_CXX_LIBRARIES})
target_compile_options(mach4 PRIVATE ${OpenMP_CXX_COMPILE_OPTIONS})
target_compile_definitions(mach4 PRIVATE ${OpenMP_CXX_COMPILE_DEFINITIONS})