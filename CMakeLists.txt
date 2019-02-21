cmake_minimum_required(VERSION 3.13)
project(Project_1)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP)
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_DIRS})



#zeta libary
include_directories(zeta)
add_subdirectory(zeta)

#mach libary
include_directories(mach)
add_subdirectory(mach)

#all the zeta programs
add_subdirectory(zeta0)
add_subdirectory(zeta1)
add_subdirectory(zeta2)

#all of mach0
add_subdirectory(mach0)

