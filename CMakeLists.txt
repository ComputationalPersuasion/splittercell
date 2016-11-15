cmake_minimum_required(VERSION 3.6)

find_package(Boost REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2 -pedantic -Wall")

include_directories(include ${Boost_INCLUDE_DIRS})

add_library(splittercell src/distribution.cpp src/flock.cpp)
