
set(SOURCES
  log.cpp
  error.cpp
  model.cpp
  optional.cpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()


set(PU_SRCS ${PU_SRCS} ${DIR_SRCS})

set(DIRS
	protos
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()


add_library(
  predictive-unit
  SHARED
  
  ${PU_SRCS}
)


target_link_libraries(
	predictive-unit
	
	predictive-unit-pb
)
