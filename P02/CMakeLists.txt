cmake_minimum_required(VERSION 2.8)
project(Project_2)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP REQUIRED)
include_directories(${})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_LINK_FLAGS}")

if(OpenMP_FOUND)
    message("Using OpenMP")
endif(OpenMP_FOUND)

# try to set c++ 11 manually
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11")

