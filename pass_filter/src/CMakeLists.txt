cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pass_filter)

find_package(PCL 1.3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pass_filter pass_filter.cpp)
target_link_libraries (pass_filter ${PCL_LIBRARIES})


