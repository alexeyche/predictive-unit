add_executable(
  predictive-unit-test
  
  test-pu.cpp
)

target_link_libraries(
    predictive-unit-test
    predictive-unit
    ${CMAKE_THREAD_LIBS_INIT}
    yaml-cpp
)
