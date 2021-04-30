SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

# Handle OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

# Handle OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Live capture
add_executable (live-capture live-capture.cpp)
target_include_directories (live-capture PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries (live-capture imageprocessors autotimer "-lopencv_core" "-lopencv_imgproc" "-lopencv_highgui" ${OpenCL_LIBRARY} ${OpenCV_LIBS} )
target_link_libraries (live-capture imageprocessors autotimer ${OpenCL_LIBRARY} ${OpenCV_LIBS} -lopencv_videoio )
