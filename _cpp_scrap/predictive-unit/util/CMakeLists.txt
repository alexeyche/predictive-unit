set(SOURCES
	tuple.cpp
    argument.cpp
    optional.cpp
    maybe.cpp
    string.cpp
    proto-struct.cpp
    matrix.cpp
    protobuf.cpp
    ring-matrix-buffer.cpp
    rwqueue.cpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(PU_SRCS ${PU_SRCS} ${DIR_SRCS} PARENT_SCOPE)
