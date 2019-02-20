# The mach0 part

add_executable(mach0 mach0.cpp)
target_link_libraries(mach0 mach)

add_executable(mach0Test mach0Test.cpp)
target_link_libraries(mach0Test mach)