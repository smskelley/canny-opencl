//
//  hello_opencl.cpp
//  opencl-demo
//
//  Created by Sean Kelley on 10/11/14.
//  Copyright (c) 2014 Sean Kelley. All rights reserved.
//
//
//  main.cpp
//  OpenCL Test
//
//  Created by Sean Kelley on 10/5/14.
//  Copyright (c) 2014 Sean Kelley. All rights reserved.
//

#define __CL_ENABLE_EXCEPTIONS

#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cl.hpp"

using namespace std;

int main () {
    
    // OpenCL Objects
    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    vector<cl::Kernel> kernels;
    
    // OpenCV Objects
    cv::VideoCapture stream1(0);
    
    try {
        // OpenCV Initialization
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        cl::Context context(devices);
        cl::CommandQueue queue(context, devices[0]);
        
        // create and load the program
        ifstream cl_file("hello_opencl.cl");
        if (!cl_file.good())
            cerr << "Couldn't open hello_opencl.cl" << endl;
        string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
        cl::Program::Sources source(1, make_pair(cl_string.c_str(), cl_string.length() + 1));
        cl::Program program(context, source);
        program.build(devices);
        cl::Kernel kernel(program, "hello_opencl");
        
        // things to be performed every frame
        cv::Mat inFrame, grayFrame, edgeFrame;
        while (true) {
            stream1.read(inFrame);
            cv::cvtColor(inFrame, grayFrame, cv::COLOR_BGR2GRAY);
            
            size_t framePixels = grayFrame.rows * grayFrame.cols;
            size_t frameBytes = framePixels * grayFrame.elemSize();
            
            cl::Buffer frame_CL(context,
                                CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                frameBytes,
                                grayFrame.data);
            
            kernel.setArg(0, frame_CL);
            
            // execute kernel
            queue.enqueueNDRangeKernel(kernel,
                                       cl::NullRange,
                                       cl::NDRange(framePixels),
                                       cl::NDRange(1, 1),
                                       NULL);
            // copy the buffer back
            queue.enqueueReadBuffer(frame_CL, CL_TRUE, 0, frameBytes, grayFrame.data);
            
            // wait for completion
            queue.finish();
            imshow("hello_opencl", grayFrame);
            if (cv::waitKey(30) >= 0)
                break;
        }
    } catch (cl::Error e) {
        cout << endl << "Error: " << e.what() << " : " << e.err() << endl;
    }
    
    return 0;
    
}