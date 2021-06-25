cmake_minimum_required(VERSION 3.16)
project(lgmath)

option(USE_AMENT "Use ament_cmake to build lgmath for ROS2." OFF)

# Compiler setup
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-march=native -O3 -Wall -pedantic)

## cmake flow (default)
if (NOT USE_AMENT)

set(PROJECT_VERSION 1.1.0)

find_package(Eigen3 REQUIRED)

# Build library
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp")
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Install
install(
  DIRECTORY include/
  DESTINATION include
)

install(TARGETS ${PROJECT_NAME}
  DESTINATION lib
  EXPORT export_${PROJECT_NAME}
)

install(EXPORT export_${PROJECT_NAME}
  FILE export_${PROJECT_NAME}.cmake
  DESTINATION lib/cmake/${PROJECT_NAME}
)

# Export
include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "lib/cmake/example"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
# generate the version file for the config file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION "${Tutorial_VERSION_MAJOR}.${Tutorial_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
)
# install the configuration file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION lib/cmake/${PROJECT_NAME}
)
# export to build directory as well so no need to install in order to use find_package
export(EXPORT export_${PROJECT_NAME}
  FILE "${CMAKE_CURRENT_BINARY_DIR}/export_${PROJECT_NAME}.cmake"
)

## ROS2 ament_cmake flow
else()

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

file(GLOB_RECURSE SOURCE src/*.cpp)
add_library(${PROJECT_NAME} ${SOURCE})
ament_target_dependencies(${PROJECT_NAME} Eigen3)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  
  # Unit-tests
  ament_add_gtest(so3_tests tests/SO3Tests.cpp)
  target_link_libraries(so3_tests ${PROJECT_NAME})
  ament_add_gtest(se3_tests tests/SE3Tests.cpp)
  target_link_libraries(se3_tests ${PROJECT_NAME})
  ament_add_gtest(covariance_tests tests/CovarianceTests.cpp)
  target_link_libraries(covariance_tests ${PROJECT_NAME})
  ament_add_gtest(rotation_tests tests/RotationTests.cpp)
  target_link_libraries(rotation_tests ${PROJECT_NAME})
  ament_add_gtest(transform_tests tests/TransformTests.cpp)
  target_link_libraries(transform_tests ${PROJECT_NAME})
  ament_add_gtest(transform_with_covariance_tests tests/TransformWithCovarianceTests.cpp)
  target_link_libraries(transform_with_covariance_tests ${PROJECT_NAME})

  # Benchmarks (yuchen: temporarily disabled unless we have performance issue later on)
  # ament_add_gtest(so3_benchmarks benchmarks/SO3SpeedTest.cpp)
  # target_link_libraries(so3_benchmarks ${PROJECT_NAME})
  # ament_add_gtest(se3_benchmarks benchmarks/SE3SpeedTest.cpp)
  # target_link_libraries(se3_benchmarks ${PROJECT_NAME})
  # ament_add_gtest(rotation_benchmarks benchmarks/RotationSpeedTest.cpp)
  # target_link_libraries(rotation_benchmarks ${PROJECT_NAME})
  # ament_add_gtest(transform_benchmarks benchmarks/TransformSpeedTest.cpp)
  # target_link_libraries(transform_benchmarks ${PROJECT_NAME})
  # ament_add_gtest(transform_with_covariance_benchmarks benchmarks/TransformWithCovarianceSpeedTest.cpp)
  # target_link_libraries(transform_with_covariance_benchmarks ${PROJECT_NAME})

  # Linting
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies() # Lint based on linter test_depend in package.xml
endif()

endif()

# Documentation
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

if (USE_AMENT)
ament_package()
endif()
