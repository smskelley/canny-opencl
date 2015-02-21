cmake_minimum_required (VERSION 2.8.11)

project(canny-opencl)

subdirs(image-processors)
subdirs(live-capture)
subdirs(benchmark)

# copy the kernels and images to the binary directory
file(COPY "${PROJECT_SOURCE_DIR}/kernels" DESTINATION "${PROJECT_SOURCE_DIR}/bin")
file(COPY "${PROJECT_SOURCE_DIR}/images" DESTINATION "${PROJECT_SOURCE_DIR}/bin")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
