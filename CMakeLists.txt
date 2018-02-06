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

if(DEBUG)
  add_definitions(-DDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fsanitize=address -g -O0")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -fsanitize=address -g -O0")
elseif(PROFILE)
  add_definitions(-DNDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pg")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -pg")
else()
  add_definitions(-DNDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -std=c++14 -g -O3")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math -std=gnu99 -g -O3")
endif(DEBUG)


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

find_package(Threads REQUIRED)
find_package(Protobuf REQUIRED)

find_package(Poco REQUIRED Net Data) 

set(POCO_LIBS 
  /usr/local/lib/libPocoNet.so
  /usr/local/lib/libPocoFoundation.so
  /usr/local/lib/libPocoUtil.so
)

include_directories(${PROTOBUF_INCLUDE_DIRS})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(predictive-unit)
add_subdirectory(predictive-unit-bin)
add_subdirectory(predictive-unit-test)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/pu/cli.py ${CMAKE_BINARY_DIR}/bin/cli.py COPYONLY)
