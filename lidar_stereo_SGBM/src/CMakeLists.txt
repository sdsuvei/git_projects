cmake_minimum_required (VERSION 2.6)

project(lidar_stereo_SGBM)

find_package(OpenCV REQUIRED)
find_package( PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(lidar_stereo_SGBM lidar_stereo_SGBM.cpp)
target_link_libraries(lidar_stereo_SGBM ${OpenCV_LIBS} ${PCL_LIBRARIES} ${COVIS_LIBRARIES})
