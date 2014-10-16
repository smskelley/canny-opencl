# This Makefile is not portable and should be made portable in the future.
# It exists right now to serve as a starting point.

default: canny-opencl

canny-opencl: canny-opencl.cpp autotimer.o imageprocessor.o
	g++ -std=c++11 -o canny-opencl canny-opencl.cpp  autotimer.o imageprocessor.o \
		-lopencv_core -lopencv_imgproc -framework OpenCL

autotimer.o:
	g++ -std=c++11 -c autotimer.cpp

imageprocessor.o:
	g++ -std=c++11 -c imageprocessor.cpp

clean:
	rm *.o canny-opencl
