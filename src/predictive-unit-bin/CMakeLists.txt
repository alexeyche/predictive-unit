
add_executable(
	predictive-unit-bin 

 	predictive-unit.cpp
 	defaults.cpp
 	dispatcher.cpp
 	client.cpp
 	protocol.cpp
)

set_target_properties(
	predictive-unit-bin  PROPERTIES OUTPUT_NAME predictive-unit
)

target_link_libraries(
    predictive-unit-bin

    predictive-unit
    ${POCO_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    yaml-cpp
)
