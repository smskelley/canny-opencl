# This Makefile is not portable and should be made portable in the future.
# It exists right now to serve as a starting point.

CV_PATH=/usr/local/
SRCDIR=src
VPATH=$(SRCDIR)
LDFLAGS=-L$(CV_PATH)lib/ 
CPPFLAGS=-I$(CV_PATH)include/ -std=c++11 
LDLIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -framework OpenCL

default: canny-opencl benchmark

canny-opencl: canny-opencl.cpp autotimer.o imageprocessor.o

benchmark: benchmark.cpp autotimer.o imageprocessor.o

autotimer.o: autotimer.h autotimer.cpp

imageprocessor.o: imageprocessor.h imageprocessor.cpp

clean:
	rm *.o canny-opencl benchmark
