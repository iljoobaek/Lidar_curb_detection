cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set( CMAKE_CXX_STANDARD 14)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
project(pcl_visualizer_point_cloud_annotation)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pointcloud_annotation gt_annotator.cpp)
target_link_libraries (pointcloud_annotation ${PCL_LIBRARIES})
