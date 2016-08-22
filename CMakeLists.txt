cmake_minimum_required(VERSION 3.5.1)
project(splittercell)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2")
include_directories(include)

option(BUILD_TESTS "Build the test executable" OFF)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

set(SOURCE_FILES src/graph.cpp include/graph.h src/distribution.cpp include/distribution.h)
add_library(splittercell SHARED ${SOURCE_FILES})