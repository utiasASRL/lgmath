
# speed_test script
add_executable(SO3SpeedTest ${CMAKE_CURRENT_SOURCE_DIR}/SO3SpeedTest.cpp)
target_link_libraries(SO3SpeedTest lgmath ${DEPEND_LIBS})

# speed_test script
add_executable(SE3SpeedTest ${CMAKE_CURRENT_SOURCE_DIR}/SE3SpeedTest.cpp)
target_link_libraries(SE3SpeedTest lgmath ${DEPEND_LIBS})

# speed_test script
add_executable(RotationSpeedTest ${CMAKE_CURRENT_SOURCE_DIR}/RotationSpeedTest.cpp)
target_link_libraries(RotationSpeedTest lgmath ${DEPEND_LIBS})

# speed_test script
add_executable(TransformSpeedTest ${CMAKE_CURRENT_SOURCE_DIR}/TransformSpeedTest.cpp)
target_link_libraries(TransformSpeedTest lgmath ${DEPEND_LIBS})

