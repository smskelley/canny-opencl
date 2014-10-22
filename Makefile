# This Makefile is not portable and should be made portable in the future.
# It exists right now to serve as a starting point.

CV_PATH=/usr/local/
default: canny-opencl benchmark

canny-opencl: canny-opencl.cpp autotimer.o imageprocessor.o
	g++ -std=c++11 -o canny-opencl canny-opencl.cpp  autotimer.o imageprocessor.o \
		-lopencv_core -lopencv_imgproc -lopencv_highgui -framework OpenCL \
		-I$(CV_PATH)include/ -L$(CV_PATH)lib/ 

benchmark: benchmark.cpp autotimer.o imageprocessor.o
	g++ -std=c++11 -o benchmark benchmark.cpp  autotimer.o imageprocessor.o \
		-lopencv_core -lopencv_imgproc -lopencv_highgui -framework OpenCL \
		-I$(CV_PATH)include/ -L$(CV_PATH)lib/ 

autotimer.o: autotimer.h autotimer.cpp
	g++ -std=c++11 -c autotimer.cpp

imageprocessor.o: imageprocessor.h imageprocessor.cpp
	g++ -std=c++11 -c imageprocessor.cpp -I$(CV_PATH)include/

clean:
	rm *.o canny-opencl benchmark
