SET(SOURCES
 	
 	matrix.proto
  	layer-config.proto
 	messages.proto
 	hostmap.proto
)


set(FULL_PATH_SOURCES)

foreach(file ${SOURCES})
  set(FULL_PATH_SOURCES ${FULL_PATH_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${FULL_PATH_SOURCES})
PROTOBUF_GENERATE_PYTHON(PROTO_PY ${FULL_PATH_SOURCES})

add_custom_target(generate_proto ALL DEPENDS ${PROTO_PY})

ADD_LIBRARY(
	predictive-unit-pb SHARED 
	${PROTO_SRCS} 
	${PROTO_HDRS}
)


SET_TARGET_PROPERTIES(
	predictive-unit-pb PROPERTIES COMPILE_FLAGS "-fPIC"
)

TARGET_LINK_LIBRARIES(
	predictive-unit-pb

    ${PROTOBUF_LIBRARY}
)
