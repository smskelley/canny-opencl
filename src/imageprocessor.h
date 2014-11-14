#pragma once
#define __CL_ENABLE_EXCEPTIONS

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cl.hpp"

class ImageProcessor
{
    // OpenCV Objects
    cv::Mat input;
    cv::Mat output;

    cv::Mat image_matrix;    
    cv::Mat gaussian_matrix;
    cv::Mat sobel_matrix;
    cv::Mat theta_matrix;
    cv::Mat nonMaxSuppression_matrix;
    cv::Mat hysteresisThresholding_matrix; 

    // OpenCL Objects
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Device selectedDevice;
    cl::Context context;
    cl::CommandQueue queue;
    
    cl::Kernel gaussian;
    cl::Kernel sobel;
    cl::Kernel nonMaxSuppression;
    cl::Kernel hysteresisThresholding;

    // Create a buffer to hold the direction angle theta
    cl::Buffer image_buffer;
    cl::Buffer gaussian_buffer;
    cl::Buffer sobel_buffer;
    cl::Buffer theta_buffer;
    cl::Buffer nonMaxSuppression_buffer;
    cl::Buffer hysteresisThresholding_buffer;

    // Note that existing code assumes only two buffers exist
    cl::Buffer buffers[2];
    
    // Keeps track of the next buffer to use as a destination. This should
    // not be accessed directly, instead look at using nextBuff/prevBuff.
    size_t bufferIndex = 0;

    // Private Methods
    // nextBuff returns a reference to the next buffer that should be modified.
    cl::Buffer& nextBuff() { return buffers[bufferIndex]; }
    // prevBuff returns a reference to the previous buffer that was modified.
    cl::Buffer& prevBuff() { return buffers[bufferIndex ^ 1]; }
    // Advance the buffer. Note there's only two, so right now it just swaps
    // to the other buffer right now.
    void advanceBuff() { bufferIndex ^= 1; }

    // returns the "desirable" device. If a discrete GPU is detected, then it
    // will be preferred over integrated graphics. If devices is empty, then the
    // this will throw std::out_of_range.
    cl::Device & findBestDevice();

    // Given a filename (without its path) load and return the kernel.
    cl::Kernel loadKernel(std::string filename, std::string kernel_name);
  public:
    ImageProcessor(bool UseGPU = true);

    // outputs basic information about the device in use.
    void DeviceInfo();

    // Note that input matrices are assumed to be 8 bit 1 channel grayscale.
    void LoadImage(cv::Mat &input);

    // Wait for all other operations to complete and then return the cv::Mat
    // corresponding to the output of previously enqueued operations.
    cv::Mat GetOutput();

    cv::Mat GetGaussian();
    cv::Mat GetSobel();
    cv::Mat GetTheta();
    cv::Mat GetNonMaxSuppression();
    cv::Mat GetHysteresisThresholding();
    
    // Blocking call which finishes all commands in queue. Useful for
    // benchmarking purposes, so that you can time a subset of an operation.
    void FinishJobs();

    // These operations will enqueue the appropriate kernel. They are
    // non-blocking.
    void Gaussian();
    void Sobel();
    void NonMaxSuppression();
    void HysteresisThresholding();
    void Canny();
};
