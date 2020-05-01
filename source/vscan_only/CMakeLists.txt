cmake_minimum_required( VERSION 3.6 )

# Require C++11
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# Set O3 optimization flag
set( CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

# Create Project
project( virtualscan )
set(EXECUTABLE virtualscan)

# Include ros directory
include_directories(/opt/ros/kinetic/include)
link_directories(/opt/ros/kinetic/lib)

# Include vscan directory
include_directories(vscan/include)

set (SOURCES vscan/FastVirtualScan/fastvirtualscan.cpp Fusion.cpp data_reader.h sensor_config.h viewer.h VelodyneCapture.h main.cpp timers.cpp)
add_executable( ${EXECUTABLE} ${SOURCES} )

# Find Package Python2
find_package(PythonInterp 2.7 REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)
set(PROJECT_LIBRARIES ${PYTHON_LIBRARIES} )
set(PROJECT_INCLUDES_PYTHON ${PYTHON_INCLUDE_DIRS} )
MESSAGE(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
MESSAGE(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
include_directories(${PROJECT_INCLUDES_PYTHON})
target_link_libraries(${EXECUTABLE} ${PROJECT_LIBRARIES})

# filesystem
target_link_libraries(${EXECUTABLE}  stdc++fs )

# *** Find the rosbag.so
#find_package(rosbag REQUIRED)
#target_link_libraries( ${EXECUTABLE} ${rosbag_LIBRARIES} ) 

# Find Package Qt
find_package(Qt5Core REQUIRED)
include_directories( ${Qt5Core_INCLUDE_DIRS} )
target_link_libraries( ${EXECUTABLE} Qt5::Core )

# Find Package Eigen3
find_package( Eigen3 3.3 REQUIRED NO_MODULE )
target_link_libraries( ${EXECUTABLE} Eigen3::Eigen rt)

# OpenMP
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} ")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ")
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
set( OpenCV_DIR "/home/tarang/Lidar_Project_Fall_2019_Tarang/Lidar_curb_detection/opencv_source/opencv-3.4.9/build" )
# set( OpenCV_STATIC OFF )
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


set(Boost_INCLUDE_DIRS "/usr/include/boost")
set(Boost_LIBRARIES "/usr/lib/aarch64-linux-gnu")
#set(Boost_LIBRARIES "/usr/local/lib")
find_package(Boost 1.45.0 COMPONENTS python)


find_package(Boost COMPONENTS filesystem date_time python system REQUIRED)
include_directories (${Boost_INCLUDE_DIRS})
target_link_libraries (${EXECUTABLE} boost_python ${Boost_LIBRARIES}
)
if(Boost_FOUND )
  message( "found boost" )
else()
  message( "Boost not found" )
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
  target_link_libraries( ${EXECUTABLE} ${CMAKE_THREAD_LIBS_INIT} )
  target_link_libraries( ${EXECUTABLE} ${Boost_LIBRARIES} )
  target_link_libraries( ${EXECUTABLE} ${PCAP_LIBRARIES} )
  target_link_libraries( ${EXECUTABLE} ${OpenCV_LIBS} )
else()
  message( WARNING "VelodyneCapture need at least either Boost or PCAP." )
endif()
