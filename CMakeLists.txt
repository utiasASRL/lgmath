################################################################################
### Header
################################################################################
cmake_minimum_required(VERSION 2.8)
project(lgmath)

# build tool
find_package(catkin_optional REQUIRED)
co_build_type() # cache the build type

################################################################################
### Compiler Setup
################################################################################

# TODO: -std=c++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -std=c++0x -O3 -march=native -mno-avx")

################################################################################
### Configure Dependency Headers and Libraries
################################################################################

# Add the search path for FindPackage.cmake files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# find Eigen
find_package(Eigen3 3.2.2 REQUIRED)
message(STATUS "FOUND: Eigen3 version ${EIGEN3_VERSION} found in ${EIGEN3_INCLUDE_DIR}")
include_directories(${EIGEN3_INCLUDE_DIR})

################################################################################
### Enable find_package()
################################################################################

# export the package (for package config)
co_export(
  VERSION 0.1.0
  INCLUDE_DIRS include
  LIBRARIES lgmath
  DEPENDS Eigen3)

################################################################################
### Specify Build Targets
################################################################################

##### Main Library 

# Find the header and source files
file(GLOB_RECURSE HEADER_FILES "include/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp")

# build and link the library
add_library(lgmath SHARED ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(lgmath ${EXTERNAL_LIBS})
set_target_properties(lgmath PROPERTIES VERSION "${PROJECT_VERSION}" SOVERSION 1)

################################################################################
### Set install targets
################################################################################

# install the exported items
co_install()

################################################################################
### Samples and Tests
################################################################################

# add 'Catch' test directory
set(TESTS_ON TRUE CACHE BOOL "Build unit tests (uses Catch)")
if(${TESTS_ON})
  enable_testing(true)  # Enables cmake unit-testing.
  add_subdirectory(tests)
endif()

# add benchmarks directory
set(BENCHMARK_ON TRUE CACHE BOOL "Build benchmark executables")
if(${BENCHMARK_ON})
  add_subdirectory(benchmarks)
endif()

find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
else(DOXYGEN_FOUND)
  message("Doxygen not found.")
  add_custom_target(doc
    COMMENT "Doxygen was not found on this system so this target does not exist, please install it and re-run CMake." VERBATIM)
endif(DOXYGEN_FOUND)
