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
    // OpenCL Objects
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Context context;
    cl::CommandQueue queue;
    
    cl::Kernel gaussian;
    cl::Kernel sobel;
    cl::Kernel nonMaxSuppression;
    cl::Kernel hysteresisThresholding;

    cl::Buffer buffers[2];

    // Keeps track of the next buffer to use as a destination
    int nextBuffer = 0;

    // OpenCV Objects
    cv::Mat input;
    cv::Mat output;

    // Private Methods
    cl::Kernel loadKernel(std::string filename, std::string kernel_name);

  public:
    ImageProcessor();

    // Note that input matrices are assumed to be 8 bit 1 channel grayscale.
    ImageProcessor(cv::Mat &input);
    void LoadImage(cv::Mat &input);
    cv::Mat GetOutput();

    void Gaussian();
    void Sobel();
    void NonMaxSuppression();
    void HysteresisThresholding();
    void Canny();
};
