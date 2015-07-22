################################################################################
### Header
################################################################################
cmake_minimum_required(VERSION 2.8)
project(catch_tests)

################################################################################
### Misc
################################################################################
# Expose required variable (CATCH_INCLUDE_DIR) to parent scope
set(CATCH_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty"
        CACHE INTERNAL "Path to include folder for Catch")

# Disable compiler warnings that Catch throws
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-variadic-macros")
elseif(MSVC)
endif()


################################################################################
### Specify Build Targets
################################################################################
include_directories(${CATCH_INCLUDE_DIR})

# Build test executable
add_executable(lgmath_unit_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SO3Tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SE3Tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TransformTests.cpp
)
target_link_libraries(lgmath_unit_tests lgmath ${DEPEND_LIBS})

# Add cmake unit tests
add_test(NAME lgmath_unit_tests COMMAND lgmath_unit_tests)


