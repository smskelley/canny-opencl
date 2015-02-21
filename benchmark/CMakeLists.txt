SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
include_directories ("${PROJECT_SOURCE_DIR}/include/")
#add_library (benchmark benchmark.cpp)
add_library (autotimer autotimer.cpp)
add_executable (benchmark-suite benchmark-suite.cpp benchmark.cpp)
target_link_libraries (benchmark-suite image-processors autotimer "-lopencv_core" "-lopencv_imgproc" "-lopencv_highgui" "-framework OpenCL")

target_include_directories (autotimer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
