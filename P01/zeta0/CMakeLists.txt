# the zeta0 part


add_executable(zeta0 zeta0.cpp)
target_link_libraries(zeta0 zeta)

add_executable(zeta0UTest zeta0UTest.cpp)
target_link_libraries(zeta0UTest zeta)

add_executable(zeta0VTest zeta0VTest.cpp)
target_link_libraries(zeta0VTest zeta)


#Sanitizerer
#add_sanitize_thread(zeta0)
#add_sanitize_memory(zeta0)
#add_sanitize_address(zeta0)