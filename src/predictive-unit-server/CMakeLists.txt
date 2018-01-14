add_executable(
  predictive-unit-server
  
  predictive-unit-server.cpp
)

target_link_libraries(
    predictive-unit-server
    predictive-unit
    ${CMAKE_THREAD_LIBS_INIT}
    yaml-cpp
)
