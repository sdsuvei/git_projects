cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(statistics_tests)

find_package(PCL 1.3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (statistics_tests statistics_tests.cpp)
target_link_libraries (statistics_tests ${PCL_LIBRARIES})


