cmake_minimum_required(VERSION 2.6)
project(poisson)

enable_language(C)
enable_language(CXX)
enable_language(Fortran)


set(CMAKE_CXX_STANDARD 14)

# CMake Options to enable/disable MPI and OpenMP
option(ENABLE_OPENMP "Enable OpenMP support?" ON)
option(ENABLE_MPI    "Enable MPI support?" ON)

if(ENABLE_MPI)
    find_package(MPI)
endif()

if(MPI_FOUND)
    message("Using MPI")
    add_definitions(-DHAVE_MPI=1)
    set(INCLUDES ${INCLUDES} ${MPI_INCLUDE_PATH})
    include_directories(${MPI_CXX_INCLUDE_DIRS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_LINK_FLAGS}")
    set(DEPLIBS_C ${DEPLIBS_C} ${MPI_C_LIBRARIES})
    set(DEPLIBS_CXX ${DEBLIBS_CXX} ${MPI_CXX_LIBARIES})
endif()

if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        message("Using OpenMP")
        add_definitions(-DHAVE_OPENMP=1)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(DEPLIBS_CXX ${DEBLIBS_CXX} ${OpenMP_CXX_LIBRARIES})
    endif()
endif()


find_package(Boost REQUIRED)
if(BOOST_FOUND)
    set(INCLUDES ${INCLUDES} ${Boost_INCLUDE_DIRS})
    set(DEPLIBS_CXX ${DEBLIBS_CXX} ${Boost_CXX_LIBRARY})

endif()


# try to set c++ 11 manually
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11")


# Send the include directories to the compiler
include_directories(${INCLUDES})

# Add a library with common sources
add_library(common fst.f)
set_target_properties(common PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES Fortran)



# The C based poisson application
add_executable(poisson-c poisson.c)

# Link to libraries
target_link_libraries(poisson-c common ${DEPLIBS_C})

# We want to link this as a C program
set_target_properties(poisson-c PROPERTIES LINKER_LANGUAGE C)


# The C++ based poission application
add_executable(poisson poisson.cpp)


#Add MPI and openMP stuff
target_link_libraries(poisson common ${DEPLIBS_CXX})
target_compile_options(poisson PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
target_compile_definitions(poisson PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
set_target_properties(poisson PROPERTIES LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
target_compile_options(poisson PRIVATE ${OpenMP_CXX_COMPILE_OPTIONS})
target_compile_definitions(poisson PRIVATE ${OpenMP_CXX_COMPILE_DEFINITIONS})


set_target_properties(poisson PROPERTIES LINKER_LANGUAGE CXX)