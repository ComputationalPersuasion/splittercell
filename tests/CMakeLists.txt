cmake_minimum_required(VERSION 3.6)

add_subdirectory(googletest)

include_directories(${splittercell_SOURCE_DIR}/include)

add_executable(sctests distribution_test.cpp)
target_link_libraries(sctests splittercell)
target_link_libraries(sctests gtest gtest_main)
