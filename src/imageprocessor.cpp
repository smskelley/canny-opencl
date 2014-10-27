#include <cassert>
#include <cmath>
#include "imageprocessor.h"

using namespace std;

/***  Private Methods  ********************************************************/

cl::Kernel ImageProcessor::loadKernel(string filename, string kernel_name)
{
    ifstream cl_file("kernels/" + filename);
    if (!cl_file.good())
        cerr << "Couldn't open " << filename<< endl;
    string cl_string(istreambuf_iterator<char>(cl_file),
            (istreambuf_iterator<char>()));
    cl::Program::Sources source(1, make_pair(cl_string.c_str(),
                cl_string.length() + 1));
    cl::Program program(context, source);
    try
    {
        program.build(devices);
    }
    catch (cl::Error e)
    {
        // If there's a build error, print out the build log to see what
        // exactly the problem was.
        cerr << "Build Status:\t"
             << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0])
             << endl
             << "Build Options:\t"
             << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0])
             << endl
             << "Build Log:\t "
             << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
             << endl;
    }

    return cl::Kernel(program, kernel_name.c_str());
}


// outputs basic information about the device in use.
void ImageProcessor::deviceInfo()
{
    cout << "Device info:" << endl
         << "Name: " << devices[0].getInfo<CL_DEVICE_NAME>() << endl;

}

/***  Public Methods **********************************************************/

ImageProcessor::ImageProcessor(bool UseGPU)
{
    try
    {
        // OpenCL Initialization
        cl::Platform::get(&platforms);

        if (UseGPU)
            platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        else
            platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);

        context = cl::Context(devices);
        queue = cl::CommandQueue(context, devices[0]);

        // output device information
        deviceInfo();
        
        // create and load the kernels
        gaussian = loadKernel("gaussian_kernel.cl", "gaussian_kernel");

        // currently using the edge kernel as the sobel kernel
        sobel = loadKernel("sobel_kernel.cl", "sobel_kernel");
    }
    catch (cl::Error e)
    {
        cerr << endl << "Error: " << e.what() << " : " << e.err() << endl;
    }
}

// load the 8bit 1channel grayscale Mat
void ImageProcessor::LoadImage(cv::Mat &input)
{
    this->input = input;
    output = cv::Mat(input.rows, input.cols, CV_8UC1);

    nextBuff() = cl::Buffer(context,
                            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                            input.rows * input.cols * input.elemSize(),
                            input.data);
    prevBuff() = cl::Buffer(context,
                            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                            input.rows * input.cols * input.elemSize(),
                            output.data);
                            
    // Initialize the theta buffer
    theta =  cl::Buffer(context,
                        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                        input.rows * input.cols * input.elemSize(),
                        output.data);                       
    advanceBuff();
}

// copies the buffer back and returns it. this is a blocking call.
cv::Mat ImageProcessor::GetOutput()
{
    // copy the buffer back
    queue.enqueueReadBuffer(prevBuff(), CL_TRUE, 0,
                            input.rows * input.cols * input.elemSize(),
                            output.data);

    queue.finish();
    assert(output.rows == input.rows && output.cols == input.cols);
    return output;
}

void ImageProcessor::FinishJobs()
{
    queue.finish();
}

// enqueues the gaussian kernel
void ImageProcessor::Gaussian()
{
    gaussian.setArg(0, prevBuff());
    gaussian.setArg(1, nextBuff());
    gaussian.setArg(2, input.rows);
    gaussian.setArg(3, input.cols);

    queue.enqueueNDRangeKernel(gaussian,
                               cl::NullRange,
                               cl::NDRange(input.rows,
                                           input.cols),
                               cl::NDRange(1, 1),
                               NULL);

    advanceBuff();
}

// enqueues the sobel kernel
void ImageProcessor::Sobel()
{
    sobel.setArg(0, prevBuff());
    sobel.setArg(1, nextBuff());
    sobel.setArg(2, theta);
    sobel.setArg(3, input.rows);
    sobel.setArg(4, input.cols);

    queue.enqueueNDRangeKernel(sobel,
                               cl::NullRange,
                               cl::NDRange(input.rows,
                                           input.cols),
                               cl::NDRange(1, 1),
                               NULL);

    advanceBuff();
}

// enqueues the nonMaxSuppression kernel
void ImageProcessor::NonMaxSuppression()
{
}

// enqueues the hysteresisThresholding kernel
void ImageProcessor::HysteresisThresholding()
{
}

// enqueues all of the canny stages
void ImageProcessor::Canny()
{
    Gaussian();
    Sobel();
    NonMaxSuppression();
    HysteresisThresholding();
}
