add_executable(
  predictive-unit-test
  
  predictive-unit-test.cpp
)

target_link_libraries(
    predictive-unit-test

    predictive-unit
    ${CMAKE_THREAD_LIBS_INIT}

    ${POCO_LIBS}
    yaml-cpp
)
