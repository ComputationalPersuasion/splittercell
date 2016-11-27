cmake_minimum_required(VERSION 3.6)

find_package(Boost REQUIRED)

add_subdirectory(googletest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -pedantic -Wall")

include_directories(../include ${Boost_INCLUDE_DIRS})
link_directories(../)

add_executable(sctests distribution_test.cpp)
target_link_libraries(sctests gtest gmock gmock_main splittercell)
