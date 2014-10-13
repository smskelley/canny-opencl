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
#include <cassert>
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
        ifstream cl_file("convolution_kernel.cl");
        if (!cl_file.good())
            cerr << "Couldn't open hello_opencl.cl" << endl;
        string cl_string(istreambuf_iterator<char>(cl_file),
                         (istreambuf_iterator<char>()));
        cl::Program::Sources source(1, make_pair(cl_string.c_str(),
                                                 cl_string.length() + 1));
        cl::Program program(context, source);
        program.build(devices);
        cl::Kernel kernel(program, "convolution_kernel");
        
        // things to be performed every frame
        cv::Mat inFrame, grayFrame;
        cv::Mat edgeFrame(cv::Size(1280,720), CV_8UC1);
        while (true) {
            stream1.read(inFrame);
            cv::cvtColor(inFrame, grayFrame, cv::COLOR_BGR2GRAY);
            
            size_t framePixels = grayFrame.rows * grayFrame.cols;
            size_t frameBytes = framePixels * grayFrame.elemSize();
            
            // copy over the input frame.
            cl::Buffer frame_CL(context,
                                CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                frameBytes,
                                grayFrame.data);
            
            // There's probably a better way to do this; we don't actually need
            // to copy over the output buffer. But this does guarentee there's
            // space to put the output.
            cl::Buffer edge_CL(context,
                                CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                frameBytes,
                                edgeFrame.data);
            
            // kernel args: (__global uchar *data, __global uchar *out,
            //               size_t rows, size_t cols)
            kernel.setArg(0, frame_CL);
            kernel.setArg(1, edge_CL);
            kernel.setArg(2, grayFrame.rows);
            kernel.setArg(3, grayFrame.cols);
            
            // execute kernel
            queue.enqueueNDRangeKernel(kernel,
                                       cl::NullRange,
                                       cl::NDRange(grayFrame.rows,
                                                   grayFrame.cols),
                                       cl::NDRange(1, 1),
                                       NULL);
            // copy the buffer back
            queue.enqueueReadBuffer(edge_CL, CL_TRUE, 0, frameBytes,
                                    edgeFrame.data);
            
            // wait for completion
            queue.finish();
            imshow("hello_opencl", edgeFrame);
            if (cv::waitKey(30) >= 0)
                break;
        }
    } catch (cl::Error e) {
        cout << endl << "Error: " << e.what() << " : " << e.err() << endl;
    }
    
    return 0;
    
}
