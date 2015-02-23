SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

# Autotimer
add_library (autotimer autotimer.cpp)
target_include_directories (autotimer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Benchmark suite
add_executable (benchmark-suite benchmark-suite.cpp benchmark.cpp)
target_link_libraries (benchmark-suite imageprocessors autotimer "-lopencv_core" "-lopencv_imgproc" "-lopencv_highgui" "-framework OpenCL")

