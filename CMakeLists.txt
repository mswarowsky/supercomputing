cmake_minimum_required(VERSION 3.13)
project(Project_1)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP)
#find_package(MPI)

#zeta libary
include_directories(zeta)
add_subdirectory(zeta)

#mach libary
include_directories(mach)
add_subdirectory(mach)

#all of zeta0
add_subdirectory(zeta0)

#all of mach0
add_subdirectory(mach0)

