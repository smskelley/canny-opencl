#include <cassert>
#include <cmath>
#include <regex>
#include <stdexcept>
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
             << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(selectedDevice)
             << endl
             << "Build Options:\t"
             << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(selectedDevice)
             << endl
             << "Build Log:\t "
             << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selectedDevice)
             << endl;
    }

    return cl::Kernel(program, kernel_name.c_str());
}

// Returns (hopefully) the discrete GPU in devices. If none are found, then the
// first GPU is returned.
cl::Device &ImageProcessor::findBestDevice()
{
    if (devices.size() == 0)
        throw std::out_of_range("No devices in devices vector.");
    
    // look for nvidia, amd, or ati. This may yield a false positive for
    // integrated amd GPUs, but it's better than the current solution.
    std::regex valid_device("(NVIDIA|AMD|ATI)", std::regex_constants::icase);
    for (auto &d : devices)
    {
        if (std::regex_search(d.getInfo<CL_DEVICE_VENDOR>(), valid_device))
            return d;
    }

    return devices[0];
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

        selectedDevice = findBestDevice();
        context = cl::Context(devices);
        queue = cl::CommandQueue(context, selectedDevice);
        
        // create and load the kernels
        gaussian = loadKernel("gaussian_kernel.cl", "gaussian_kernel");

        // currently using the edge kernel as the sobel kernel
        sobel = loadKernel("sobel_kernel.cl", "sobel_kernel");

        nonMaxSuppression = loadKernel("non_max_supp_kernel.cl", "non_max_supp_kernel");
    }
    catch (cl::Error e)
    {
        cerr << endl << "Error: " << e.what() << " : " << e.err() << endl;
    }
}

// outputs basic information about the device in use.
void ImageProcessor::DeviceInfo()
{
    cout << "Device info:" << endl
         << "Name: " << selectedDevice.getInfo<CL_DEVICE_NAME>() << endl
         << "Vendor: " << selectedDevice.getInfo<CL_DEVICE_VENDOR>() << endl;

}

// load the 8bit 1channel grayscale Mat
void ImageProcessor::LoadImage(cv::Mat &image)
{
    //this->input = input;
    image_matrix = image;
    output = cv::Mat(image_matrix.rows, image_matrix.cols, CV_8UC1);

    // Initialize matrices created by Jared
    gaussian_matrix = cv::Mat(image_matrix.rows, image_matrix.cols, CV_8UC1);
    sobel_matrix = cv::Mat(image_matrix.rows, image_matrix.cols, CV_8UC1);
    theta_matrix = cv::Mat(image_matrix.rows, image_matrix.cols, CV_8UC1);
    nonMaxSuppression_matrix = cv::Mat(image_matrix.rows, image_matrix.cols, CV_8UC1);
    hysteresisThresholding_matrix = cv::Mat(image_matrix.rows, image_matrix.cols, CV_8UC1);


    nextBuff() = cl::Buffer(context,
                            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                            input.rows * input.cols * input.elemSize(),
                            input.data);
    prevBuff() = cl::Buffer(context,
                            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                            input.rows * input.cols * input.elemSize(),
                            output.data);
                            
    // Initialize buffers created by Jared
    gaussian_buffer =  cl::Buffer(context,
                        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                        image_matrix.rows * image_matrix.cols * image_matrix.elemSize(),
                        gaussian_matrix.data);                       
    sobel_buffer =  cl::Buffer(context,
                        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                        image_matrix.rows * image_matrix.cols * image_matrix.elemSize(),
                        sobel_matrix.data);                       
    theta_buffer =  cl::Buffer(context,
                        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                        input.rows * input.cols * input.elemSize(),
                        theta_matrix.data);                       
    nonMaxSuppression_buffer =  cl::Buffer(context,
                                CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                input.rows * input.cols * input.elemSize(),
                                nonMaxSuppression_matrix.data);                       
    hysteresisThresholding_buffer =  cl::Buffer(context,
                                     CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                     input.rows * input.cols * input.elemSize(),
                                     hysteresisThresholding_matrix.data);                       
    //advanceBuff();
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

cv::Mat ImageProcessor::GetGaussian()
{
    // copy the gaussian buffer back
    queue.enqueueReadBuffer(gaussian_buffer, CL_TRUE, 0,
                            input.rows * input.cols * input.elemSize(),
                            gaussian_matrix.data);
    queue.finish();
    return gaussian_matrix;
}

cv::Mat ImageProcessor::GetSobel()
{
    // copy the sobel buffer back
    queue.enqueueReadBuffer(sobel_buffer, CL_TRUE, 0,
                            input.rows * input.cols * input.elemSize(),
                            sobel_matrix.data);
    queue.finish();
    return sobel_matrix;
}

cv::Mat ImageProcessor::GetTheta()
{
    // copy the theta buffer back
    queue.enqueueReadBuffer(theta_buffer, CL_TRUE, 0,
                            input.rows * input.cols * input.elemSize(),
                            theta_matrix.data);
    queue.finish();
    return theta_matrix;
}

cv::Mat ImageProcessor::GetNonMaxSuppression()
{
    // copy the NonMaxSuppression buffer back
    queue.enqueueReadBuffer(nonMaxSuppression_buffer, CL_TRUE, 0,
                            input.rows * input.cols * input.elemSize(),
                            nonMaxSuppression_matrix.data);
    queue.finish();
    return theta_matrix;
}

cv::Mat ImageProcessor::GetHysteresisThresholding()
{
    // copy the hysteresisThresholding buffer back
    queue.enqueueReadBuffer(hysteresisThresholding_buffer, CL_TRUE, 0,
                            input.rows * input.cols * input.elemSize(),
                            hysteresisThresholding_matrix.data);
    queue.finish();
    return hysteresisThresholding_matrix;
}

void ImageProcessor::FinishJobs()
{
    queue.finish();
}

// enqueues the gaussian kernel
void ImageProcessor::Gaussian()
{
    gaussian.setArg(0, input);
    gaussian.setArg(1, gaussian_buffer);
    gaussian.setArg(2, input.rows);
    gaussian.setArg(3, input.cols);

    queue.enqueueNDRangeKernel(gaussian,
                               cl::NullRange,
                               cl::NDRange(input.rows,
                                           input.cols),
                               cl::NDRange(1, 1),
                               NULL);

    //advanceBuff();
}

// enqueues the sobel kernel
void ImageProcessor::Sobel()
{
    sobel.setArg(0, gaussian_buffer);
    sobel.setArg(1, sobel_buffer);
    sobel.setArg(2, theta_buffer);
    sobel.setArg(3, input.rows);
    sobel.setArg(4, input.cols);

    queue.enqueueNDRangeKernel(sobel,
                               cl::NullRange,
                               cl::NDRange(input.rows,
                                           input.cols),
                               cl::NDRange(1, 1),
                               NULL);

    //advanceBuff();
}

// enqueues the nonMaxSuppression kernel
void ImageProcessor::NonMaxSuppression()
{
    nonMaxSuppression.setArg(0, sobel_buffer);
    nonMaxSuppression.setArg(1, nonMaxSuppression_buffer);
    nonMaxSuppression.setArg(2, theta_buffer);
    nonMaxSuppression.setArg(3, input.rows);
    nonMaxSuppression.setArg(4, input.cols);

    queue.enqueueNDRangeKernel(nonMaxSuppression,
                               cl::NullRange,
                               cl::NDRange(input.rows - 2,
                                           input.cols - 2),
                               cl::NDRange(1, 1),
                               NULL);

    //advanceBuff();
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
