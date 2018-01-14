# Update the module path to include any extra CMAKE modules we might ship.
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(predictive-unit)

# Set the minimum required version of cmake
CMAKE_MINIMUM_REQUIRED(VERSION 3.5.1)

# Enable warnings and make them errors
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

# Magic to set GCC-specific compile flags (to turn on optimisation).
set(GCC_FLAGS "-std=c++14 -Wall -O3 -g -msse3 -mavx -march=native")
add_definitions( -DSSE3 )
add_definitions( -DAVX )

if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_FLAGS}")
endif(CMAKE_COMPILER_IS_GNUCC)
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_FLAGS}")
endif(CMAKE_COMPILER_IS_GNUCXX)

# On x86_64 we need to compile with -fPIC
if(UNIX AND NOT WIN32)
  find_program(CMAKE_UNAME uname /bin /usr/bin /usr/local/bin )
  if(CMAKE_UNAME)
    exec_program(uname ARGS -m OUTPUT_VARIABLE CMAKE_SYSTEM_PROCESSOR)
    set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR} 
    	CACHE INTERNAL "processor type (i386 and x86_64)")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      add_definitions(-fPIC)
    endif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  endif(CMAKE_UNAME)
endif(UNIX AND NOT WIN32)

include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(/usr/local/include/eigen3)
include_directories(/usr/local/include/yaml-cpp)
include_directories(/usr/include/spdlog)

include(cmake/FindProtobuf.cmake)

find_package(Threads)

set(protobuf_source_dir /contrib/grpc/third_party/protobuf)
include(/contrib/grpc/third_party/protobuf/cmake/libprotobuf.cmake)

include_directories(${PROTOBUF_INCLUDE_DIRS})

add_subdirectory(predictive-unit)
add_subdirectory(predictive-unit-server)
add_subdirectory(predictive-unit-test)
