cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set( CMAKE_CXX_STANDARD 14)
project(pcl_visualizer_viewports)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pcl_visualizer_demo pcl_visualizer_demo.cpp)
target_link_libraries (pcl_visualizer_demo ${PCL_LIBRARIES})
