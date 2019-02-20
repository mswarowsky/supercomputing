# the zeta0 part


add_executable(zeta0 zeta0.cpp)
target_link_libraries(zeta0 zeta)

add_executable(zeta0Test zeta0Test.cpp)
target_link_libraries(zeta0Test zeta)


#Sanitizerer
#add_sanitize_thread(zeta0)
#add_sanitize_memory(zeta0)
#add_sanitize_address(zeta0)