cmake_minimum_required (VERSION 2.8.11)

project(canny-opencl)

subdirs(imageprocessors)
subdirs(live-capture)
subdirs(benchmarking)

# copy the kernels and images to the binary directory
file(COPY "${PROJECT_SOURCE_DIR}/kernels" DESTINATION "${PROJECT_SOURCE_DIR}/bin")
file(COPY "${PROJECT_SOURCE_DIR}/images" DESTINATION "${PROJECT_SOURCE_DIR}/bin")

# executables should go in /bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
