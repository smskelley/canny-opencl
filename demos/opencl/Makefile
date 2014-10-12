CV_PATH=/usr/local/Cellar/opencv/2.4.9/

default:
	g++ -o hello_opencl hello_opencl.cpp -I$(CV_PATH)include/ -L$(CV_PATH)lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -framework OpenCL
