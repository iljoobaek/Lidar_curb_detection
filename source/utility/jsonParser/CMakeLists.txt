cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set( CMAKE_CXX_STANDARD 14)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(EXECUTABLE jsonParser)
project(jsonParser)

add_executable (${EXECUTABLE} main.cpp)


set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(nlohmann_json)
target_link_libraries( ${EXECUTABLE} nlohmann_json::nlohmann_json)

find_package(PCL 1.2 REQUIRED)

# Find Package Eigen3
find_package( Eigen3 3.3 REQUIRED NO_MODULE )
target_link_libraries( ${EXECUTABLE} Eigen3::Eigen rt)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

target_link_libraries (${EXECUTABLE} ${PCL_LIBRARIES})
target_link_libraries(${EXECUTABLE} stdc++fs)
