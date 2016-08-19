cmake_minimum_required(VERSION 3.5)
project(jointdistribution)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

set(SOURCE_FILES main2.cpp)
add_executable(jointdistribution ${SOURCE_FILES})