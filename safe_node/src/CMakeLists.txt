cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(safe_node)

find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (safe_node safe_node.cpp)
target_link_libraries (safe_node ${OpenCV_LIBS} ${PCL_LIBRARIES})


