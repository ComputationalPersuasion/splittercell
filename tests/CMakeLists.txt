cmake_minimum_required(VERSION 3.6)
project(splittercelltests)

add_subdirectory(googletest)

add_executable(sctests graph_tests.cpp)
target_link_libraries(sctests gtest gtest_main splittercell)