cmake_minimum_required (VERSION 2.6)

project(stereo_match)

find_package(OpenCV REQUIRED)
find_package( PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(stereo_match stereo_match.cpp)
target_link_libraries(stereo_match ${OpenCV_LIBS} ${PCL_LIBRARIES} ${COVIS_LIBRARIES})
