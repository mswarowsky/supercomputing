cmake_minimum_required(VERSION 2.8)
project(Project_1)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP)
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_COMPILE_OPTIONS}")

message("link: " ${MPI_CXX_LINK_FLAGS})
message("compile options: " ${MPI_CXX_COMPILE_OPTIONS})
message("compiler: " ${MPI_CXX__COMPILER})

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

