SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

# find all source files
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} imageprocessors_SRC)

# Image Processors
add_library (imageprocessors ${imageprocessors_SRC})
target_include_directories (imageprocessors PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
