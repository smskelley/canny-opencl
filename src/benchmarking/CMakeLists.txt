SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

# Handle OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

# Handle OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Autotimer
add_library (autotimer autotimer.cpp)
target_include_directories (autotimer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Benchmark suite
add_executable (benchmark-suite benchmark-suite.cpp benchmark.cpp)
target_link_libraries (benchmark-suite imageprocessors autotimer ${OpenCL_LIBRARY} ${OpenCV_LIBS} -lopencv_videoio )

