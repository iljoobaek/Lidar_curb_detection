cmake_minimum_required( VERSION 3.6 )

# Require C++11
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# Set O3 optimization flag
set( CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -lpthread -lrt ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ltbb")

# Create Project
project( Curb_detection )
set(EXECUTABLE_CLIENT Curb_detection)
set(EXECUTABLE_CLIENT_VIEWER Curb_detection_client_viewer)
set(EXECUTABLE_MANAGER Curb_detection_manager)
set(EXECUTABLE_MEM_RELEASE Release_memory)

# Create a detection log

# Include ros directory
include_directories(/opt/ros/kinetic/include)
link_directories(/opt/ros/kinetic/lib)

# Include vscan directory
include_directories(vscan/include)

set (SOURCES_CLIENT vscan/FastVirtualScan/fastvirtualscan.cpp boundary_detection.cpp Fusion.cpp data_reader.h sensor_config.h viewer.h VelodyneCapture.h main.cpp)
add_executable( ${EXECUTABLE_CLIENT} ${SOURCES_CLIENT} )

set (SOURCES_MANAGER vscan/FastVirtualScan/fastvirtualscan.cpp boundary_detection.cpp Fusion.cpp data_reader.h sensor_config.h viewer.h VelodyneCapture.h main_manager.cpp)
add_executable( ${EXECUTABLE_MANAGER} ${SOURCES_MANAGER} )

set (SOURCES_CLIENT_VIEWER vscan/FastVirtualScan/fastvirtualscan.cpp boundary_detection.cpp Fusion.cpp data_reader.h sensor_config.h viewer.h VelodyneCapture.h main_client_viewer.cpp)
add_executable( ${EXECUTABLE_CLIENT_VIEWER} ${SOURCES_CLIENT_VIEWER} )

# Find Package Python2
find_package(PythonInterp 2.7 REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)
set(PROJECT_LIBRARIES ${PYTHON_LIBRARIES} )
set(PROJECT_INCLUDES_PYTHON ${PYTHON_INCLUDE_DIRS} )
MESSAGE(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
MESSAGE(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
include_directories(${PROJECT_INCLUDES_PYTHON})
target_link_libraries(${EXECUTABLE_CLIENT} ${PROJECT_LIBRARIES})
target_link_libraries(${EXECUTABLE_MANAGER} ${PROJECT_LIBRARIES})
target_link_libraries(${EXECUTABLE_CLIENT_VIEWER} ${PROJECT_LIBRARIES})

# filesystem
target_link_libraries(${EXECUTABLE_CLIENT}  stdc++fs )
target_link_libraries(${EXECUTABLE_MANAGER}  stdc++fs )
target_link_libraries(${EXECUTABLE_CLIENT_VIEWER}  stdc++fs )

# *** Find the rosbag.so
find_package(rosbag REQUIRED)
target_link_libraries( ${EXECUTABLE_CLIENT} ${rosbag_LIBRARIES} ) 
target_link_libraries( ${EXECUTABLE_MANAGER} ${rosbag_LIBRARIES} ) 
target_link_libraries( ${EXECUTABLE_CLIENT_VIEWER} ${rosbag_LIBRARIES} ) 

# Find Package Qt
find_package(Qt5Core REQUIRED)
include_directories( ${Qt5Core_INCLUDE_DIRS} )
target_link_libraries( ${EXECUTABLE_CLIENT} Qt5::Core )
target_link_libraries( ${EXECUTABLE_MANAGER} Qt5::Core )
target_link_libraries( ${EXECUTABLE_CLIENT_VIEWER} Qt5::Core )

# Find Package Eigen3
find_package( Eigen3 3.3 REQUIRED NO_MODULE )
target_link_libraries( ${EXECUTABLE_CLIENT} Eigen3::Eigen rt)
target_link_libraries( ${EXECUTABLE_MANAGER} Eigen3::Eigen rt)
target_link_libraries( ${EXECUTABLE_CLIENT_VIEWER} Eigen3::Eigen rt)

# ltbb
# find_package(TBB REQUIRED) 
# target_link_libraries( ${EXECUTABLE} TBB)


# OpenMP
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    MESSAGE(STATUS "Found OpenMP")
    MESSAGE(STATUS "${CMAKE_CXX_FLAGS}")
ENDIF()

# Find Package Thread
find_package( Threads REQUIRED )

# Find Package Boost
set( BOOST_ROOT "C:/Program Files/boost" )
set( Boost_USE_STATIC_LIBS ON ) # Static Link Libraries ( libboost_* )
set( Boost_USE_MULTITHREADED ON ) # Multi Thread Libraries ( *-mt-* )
set( Boost_USE_STATIC_RUNTIME OFF ) # Static Runtime Libraries ( *-s* )
find_package( Boost COMPONENTS system date_time regex )

# If Direct Capture from Sensors, VelodyneCapture are required Boost.Asio.
# Please Define HAVE_BOOST in Preprocessor.
set( HAVE_BOOST )
if( Boost_FOUND )
  set( HAVE_BOOST "-DHAVE_BOOST" )
endif()

# Find Package PCAP
set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH} )
set( PCAP_DIR "C:/Program Files/WpdPack" )
find_package( PCAP )

# Disable sleep while reading a pcap
# Is is useful if you have to parse the result of a pcap faster than real
# time.

OPTION( HAVE_FAST_PCAP "Avoid sleeping whilst reading PCAP files " OFF )
if( HAVE_FAST_PCAP )
  add_definitions( -DHAVE_FAST_PCAP )
endif()

# If Capture from PCAP Files, VelodyneCapture are required PCAP.
# Please Define HAVE_PCAP in Preprocessor.
set( HAVE_PCAP )
if( PCAP_FOUND )
  set( HAVE_PCAP "-DHAVE_PCAP" )
endif()

# Use GPS timestamps insead of Unix epoch
# This is specially useful while reading PCAP files.
OPTION( HAVE_GPSTIME "Use GPS timestamps instead of Unix " OFF )
if( HAVE_GPSTIME )
  add_definitions( -DHAVE_GPSTIME )
endif()

# Use millimeters instead of the default, centimeters.
OPTION( USE_MILLIMETERS "Use millimeters instead of centimeters " OFF )
if( USE_MILLIMETERS )
  add_definitions( -DUSE_MILLIMETERS )
endif()

# Push data packets as soon as possible, if ndef the drivers push the 
# whole frame
OPTION( PUSH_SINGLE_PACKETS  "Push data packets to the stack instead of frames" OFF )
if( PUSH_SINGLE_PACKETS )
  add_definitions( -DPUSH_SINGLE_PACKETS )
endif()

# Don't push null returns to the lases queue
OPTION( NO_EMPTY_RETURNS "Discard empty returns before pushing them to the laser queue" OFF )
if( NO_EMPTY_RETURNS )
  add_definitions( -DNO_EMPTY_RETURNS )
endif()

# Find Package OpenCV
# set( OpenCV_DIR "/opt/opencv/build" )
set( OpenCV_STATIC OFF )
find_package( OpenCV REQUIRED )
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}" )
MESSAGE(STATUS "OpenCV_LIB_DIRS = ${OpenCV_LIB_DIRS} " )
MESSAGE(STATUS "OpenCV_LIBS = ${OpenCV_LIBS} " )


# Required Viz Module
if( OpenCV_FOUND )
  if( NOT "opencv_viz" IN_LIST OpenCV_LIBS )
    message( FATAL_ERROR "not found opencv_viz module." )
  endif()
endif()

if( OpenCV_STATIC AND MSVC )
  foreach( flag_var
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO )
    if( ${flag_var} MATCHES "/MD" )
      string( REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}" )
    endif()
  endforeach()
endif()

# Set Properties
if( OpenCV_FOUND AND ( PCAP_FOUND OR Boost_FOUND ) )
  # Additional Include Directories
  include_directories( ${Boost_INCLUDE_DIRS} )
  include_directories( ${PCAP_INCLUDE_DIRS} )
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Add Definitions
  add_definitions( ${Boost_LIB_DIAGNOSTIC_DEFINITIONS} )
  add_definitions( ${HAVE_BOOST} )
  add_definitions( ${HAVE_PCAP} )

  # Additional Library Directories
  link_directories( ${Boost_LIBRARY_DIRS} )
  link_directories( ${OpenCV_LIB_DIR} )

  # Additional Dependencies
  target_link_libraries( ${EXECUTABLE_CLIENT} ${CMAKE_THREAD_LIBS_INIT} )
  target_link_libraries( ${EXECUTABLE_CLIENT} ${Boost_LIBRARIES} )
  target_link_libraries( ${EXECUTABLE_CLIENT} ${PCAP_LIBRARIES} )
  target_link_libraries( ${EXECUTABLE_CLIENT} ${OpenCV_LIBS} )

  # Additional Dependencies
  target_link_libraries( ${EXECUTABLE_MANAGER} ${CMAKE_THREAD_LIBS_INIT} )
  target_link_libraries( ${EXECUTABLE_MANAGER} ${Boost_LIBRARIES} )
  target_link_libraries( ${EXECUTABLE_MANAGER} ${PCAP_LIBRARIES} )
  target_link_libraries( ${EXECUTABLE_MANAGER} ${OpenCV_LIBS} )

  # Additional Dependencies
  target_link_libraries( ${EXECUTABLE_CLIENT_VIEWER} ${CMAKE_THREAD_LIBS_INIT} )
  target_link_libraries( ${EXECUTABLE_CLIENT_VIEWER} ${Boost_LIBRARIES} )
  target_link_libraries( ${EXECUTABLE_CLIENT_VIEWER} ${PCAP_LIBRARIES} )
  target_link_libraries( ${EXECUTABLE_CLIENT_VIEWER} ${OpenCV_LIBS} )

else()
  message( WARNING "VelodyneCapture need at least either Boost or PCAP." )
endif()
