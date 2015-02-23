SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
include_directories ("${PROJECT_SOURCE_DIR}/src/include/")

# Image Processors
add_library (imageprocessors cvimageprocessor.cpp imageprocessor.cpp openclimageprocessor.cpp serialimageprocessor.cpp)
target_include_directories (imageprocessors PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
