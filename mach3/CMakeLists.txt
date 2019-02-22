add_executable(mach3 mach3.cpp)
target_link_libraries(mach3 mach)

#Add OpenMP stuff
target_link_libraries(mach3 ${OpenMP_CXX_LIBRARIES})
target_compile_options(mach3 PRIVATE ${OpenMP_CXX_COMPILE_OPTIONS})
target_compile_definitions(zeta3 PRIVATE ${OpenMP_CXX_COMPILE_DEFINITIONS})