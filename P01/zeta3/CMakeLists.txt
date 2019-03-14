add_executable(zeta3 zeta3.cpp)
target_link_libraries(zeta3 zeta)

#Add OpenMP stuff
target_link_libraries(zeta3 ${OpenMP_CXX_LIBRARIES})
target_compile_options(zeta3 PRIVATE ${OpenMP_CXX_COMPILE_OPTIONS})
target_compile_definitions(zeta3 PRIVATE ${OpenMP_CXX_COMPILE_DEFINITIONS})