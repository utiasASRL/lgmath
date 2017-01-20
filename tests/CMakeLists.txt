################################################################################
### Header
################################################################################

cmake_minimum_required(VERSION 2.8)
if(VERSION VERSION_GREATER 3.0)
  cmake_policy(SET CMP0048 OLD)       # Supresses warnings in newer versions of CMake
endif(VERSION VERSION_GREATER 3.0)

project(catch_tests)

################################################################################
### Misc
################################################################################

# Find catch (the unit-test header-only library)
co_find(catch)

# Disable compiler warnings that Catch throws
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-variadic-macros")
elseif(MSVC)
endif()


################################################################################
### Specify Build Targets
################################################################################

# Build test executable
add_executable(lgmath_unit_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SO3Tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SE3Tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TransformTests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TransformWithCovarianceTests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/RotationTests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CovarianceTests.cpp
)
target_link_libraries(lgmath_unit_tests lgmath ${DEPEND_LIBS})

# Add cmake unit tests
add_test(NAME lgmath_unit_tests COMMAND lgmath_unit_tests)


