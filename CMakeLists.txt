cmake_minimum_required(VERSION 2.8)
project(Project_1)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP REQUIRED)
include_directories(${})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_LINK_FLAGS}")

# try to set c++ 11 manually
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++14")

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
add_subdirectory(zeta3)
add_subdirectory(zeta4)

#all the machin programs
add_subdirectory(mach0)
add_subdirectory(mach1)
add_subdirectory(mach2)
add_subdirectory(mach3)
add_subdirectory(mach4)
