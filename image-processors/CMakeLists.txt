SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
include_directories ("${PROJECT_SOURCE_DIR}/include/")
add_library (image-processors cvimageprocessor.cpp imageprocessor.cpp openclimageprocessor.cpp serialimageprocessor.cpp)

target_include_directories (image-processors PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
