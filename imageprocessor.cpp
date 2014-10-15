#include "imageprocessor.h"

using namespace std;

/***  Private Methods  ********************************************************/

cl::Kernel ImageProcessor::loadKernel(string filename, string kernel_name)
{
    ifstream cl_file(filename);
    if (!cl_file.good())
        cerr << "Couldn't open " << filename<< endl;
    string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
    cl::Program::Sources source(1, make_pair(cl_string.c_str(), cl_string.length() + 1));
    cl::Program program(context, source);
    program.build(devices);
    return cl::Kernel(program, kernel_name.c_str());
}

/***  Public Methods **********************************************************/

ImageProcessor::ImageProcessor()
{
    try
    {
        // OpenCL Initialization
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        context = cl::Context(devices);
        queue = cl::CommandQueue(context, devices[0]);
        
        // create and load the kernels
        gaussian = loadKernel("convolution_kernel.cl", "guassian");
        sobel = loadKernel("convolution_kernel.cl", "sobel");
    }
    catch (cl::Error e)
    {
        cerr << endl << "Error: " << e.what() << " : " << e.err() << endl;
    }
}

// load the 8bit 1channel grayscale Mat and do opencl initialization
ImageProcessor::ImageProcessor(cv::Mat &input)
{
}

// load the 8bit 1channel grayscale Mat
void ImageProcessor::LoadImage(cv::Mat &input)
{
}

// copies the buffer back and returns it. this is a blocking call.
cv::Mat ImageProcessor::GetOutput()
{
    // Note, the following assumes there are only two buffers.
    int previousBuffer = nextBuffer ^ 1;

    // copy the buffer back
    queue.enqueueReadBuffer(buffers[previousBuffer], CL_TRUE, 0,
                            input.rows * input.cols * input.elemSize(),
                            output.data);

    queue.finish();
    return output;
}

// enqueues the gaussian kernel
void ImageProcessor::Gaussian()
{
}

// enqueues the sobel kernel
void ImageProcessor::Sobel()
{
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
}
