#include <cassert>
#include <cmath>
#include <regex>
#include "imageprocessor.h"
#include "openclimageprocessor.h"

using namespace std;

namespace ImageProcessors {

/***  Private Methods  ********************************************************/
// Return the relative path to the cpu or gpu  kernel given a filename
// e.g. KernelPath("mykernel.cl", true); // returns: kernels/gpu/mykerne.cl
string OpenclImageProcessor::KernelPath(std::string filename, bool use_gpu) {
  // this is not platform independent and should be rewritten if that becomes
  // a requirement.
  string path = "kernels/";

  if (use_gpu)
    path += "gpu/";
  else
    path += "cpu/";

  path += filename;
  return path;
}
// Loads the file as a new OpenCL program, builds it on the device, and then
// creates an OpenCL kernel with the result.
cl::Kernel OpenclImageProcessor::LoadKernel(string filename, string kernel_name,
        bool use_gpu) {
  ifstream cl_file(KernelPath(filename, use_gpu));
  if (!cl_file.good())
    cerr << "Couldn't open " << KernelPath(filename, use_gpu) << endl;

  // Read the kernel file and create a cl::Program with it.
  string cl_string(istreambuf_iterator<char>(cl_file),
                   (istreambuf_iterator<char>()));
  cl::Program::Sources source(
      1, make_pair(cl_string.c_str(), cl_string.length() + 1));
  cl::Program program(context_, source);
  try {
    program.build(devices_);
  } catch (cl::Error e) {
    // If there's a build error, print out the build log to see what
    // exactly the problem was.
    cerr << "Build Status:\t"
         << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(selected_device_)
         << endl << "Build Options:\t"
         << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(selected_device_)
         << endl << "Build Log:\t "
         << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device_) << endl;
  }

  return cl::Kernel(program, kernel_name.c_str());
}

// Returns (hopefully) the discrete GPU in devices. If none are found, then the
// first GPU is returned.
cl::Device &OpenclImageProcessor::GetBestDevice() {
  if (devices_.size() == 0)
    throw std::out_of_range("No devices in devices vector.");

  // look for nvidia, amd, or ati. This may yield a false positive for
  // integrated amd GPUs, but it's better than the current solution.
  std::regex valid_device("(NVIDIA|AMD|ATI)", std::regex_constants::icase);
  for (auto &d : devices_) {
    if (std::regex_search(d.getInfo<CL_DEVICE_VENDOR>(), valid_device))
      return d;
  }

  return devices_[0];
}

/***  Public Methods **********************************************************/

// Perform all OpenCL setup steps required to have an operational image
// processor
OpenclImageProcessor::OpenclImageProcessor(bool use_gpu) {
  // Determine the NDRange size for each group. This should be made more
  // general in the future, but it works on most hardware for now.
  if (use_gpu)
    workgroup_size_ = 16;
  else
    workgroup_size_ = 1;

  // Initialize OpenCL
  try {
    cl::Platform::get(&platforms_);

    if (use_gpu)
      platforms_[0].getDevices(CL_DEVICE_TYPE_GPU, &devices_);
    else
      platforms_[0].getDevices(CL_DEVICE_TYPE_CPU, &devices_);

    selected_device_ = GetBestDevice();
    context_ = cl::Context(devices_);
    queue_ = cl::CommandQueue(context_, selected_device_);

    // create and load the kernels
    gaussian_ = LoadKernel("gaussian_kernel.cl", "gaussian_kernel", use_gpu);
    sobel_ = LoadKernel("sobel_kernel.cl", "sobel_kernel", use_gpu);
    non_max_suppression_ =
           LoadKernel("non_max_supp_kernel.cl", "non_max_supp_kernel", use_gpu);
    hysteresis_thresholding_ =
           LoadKernel("hyst_kernel.cl", "hyst_kernel", use_gpu);
  } catch (cl::Error e) {
    cerr << endl << "Error: " << e.what() << " : " << e.err() << endl;
  }
}

// outputs basic information about the device in use.
void OpenclImageProcessor::DeviceInfo() {
  cout << "Device info:" << endl
       << "Name: " << selected_device_.getInfo<CL_DEVICE_NAME>() << endl
       << "Vendor: " << selected_device_.getInfo<CL_DEVICE_VENDOR>() << endl;
}

// load the 8bit 1channel grayscale Mat and crop if necessary to fit within
// the limitations of our group size, since no partially full workgroups will
// be used.
void OpenclImageProcessor::LoadImage(cv::Mat &image_input) {
  // We want the rows and columns to be an integer multiple of groupSize *after*
  // 2 is subtracted from them, since all of our kernels do not run edge
  // pixels. The following math yields the following results with groupSize=16
  // (using small integers for obviousness):
  // input size   desired size
  // 31           18
  // 32           18
  // 33           18
  // 34           34
  // 35           34
  int rows = ((image_input.rows - 2) / workgroup_size_) * workgroup_size_ + 2;
  int cols = ((image_input.cols - 2) / workgroup_size_) * workgroup_size_ + 2;

  // Use these new row/cols to create a rectangle which will serve as our crop
  cv::Rect croppedArea(0, 0, cols, rows);

  // Crop the image and clone it. If it's not cloned, then the layout of the
  // data won't change, so our kernels wouldn't be writing to the correct
  // location. This could be a place of likely inefficiency. It might be
  // better to move towards not actually cropping the image, instead doing
  // more work in the kernel.
  this->input_ = image_input(croppedArea).clone();

  output_ = cv::Mat(input_.rows, input_.cols, CV_8UC1);
    NextBuff() = cl::Buffer(
          context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
      input_.rows * input_.cols * input_.elemSize(), input_.data);
    PrevBuff() = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                          input_.rows * input_.cols * input_.elemSize());

  // Initialize the theta buffer
  theta_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                     input_.rows * input_.cols * input_.elemSize());
    AdvanceBuff();
}

// copies the buffer back from the device and returns it. This will block.
cv::Mat OpenclImageProcessor::output() {
  // copy the buffer back
  queue_.enqueueReadBuffer(PrevBuff(), CL_TRUE, 0,
                          input_.rows * input_.cols * input_.elemSize(),
                          output_.data);

  queue_.finish();
  assert(output_.rows == input_.rows && output_.cols == input_.cols);
  return output_;
}

// Block until all jobs finish.
void OpenclImageProcessor::FinishJobs() {
  queue_.finish();
}

// enqueues the gaussian kernel
void OpenclImageProcessor::Gaussian() {
  try {
    gaussian_.setArg(0, PrevBuff());
    gaussian_.setArg(1, NextBuff());
    gaussian_.setArg(2, input_.rows);
    gaussian_.setArg(3, input_.cols);

    // 1,1 offset and -2 to to dimensions so that we don't run on edge pixels.
    queue_.enqueueNDRangeKernel(gaussian_, cl::NDRange(1, 1),
                                cl::NDRange(input_.rows - 2, input_.cols - 2),
                                cl::NDRange(workgroup_size_, workgroup_size_),
                                NULL);

  } catch (cl::Error e) {
    cerr << endl << "Error: " << e.what() << " : " << e.err() << endl;
  }
    AdvanceBuff();
}

// enqueues the sobel kernel
void OpenclImageProcessor::Sobel() {
  sobel_.setArg(0, PrevBuff());
  sobel_.setArg(1, NextBuff());
  sobel_.setArg(2, theta_);
  sobel_.setArg(3, input_.rows);
  sobel_.setArg(4, input_.cols);

  // 1,1 offset and -2 to to dimensions so that we don't run on edge pixels.
  queue_.enqueueNDRangeKernel(sobel_, cl::NDRange(1, 1),
                              cl::NDRange(input_.rows - 2, input_.cols - 2),
                              cl::NDRange(workgroup_size_, workgroup_size_),
                              NULL);

    AdvanceBuff();
}

// enqueues the nonMaxSuppression kernel
void OpenclImageProcessor::NonMaxSuppression() {
  non_max_suppression_.setArg(0, PrevBuff());
  non_max_suppression_.setArg(1, NextBuff());
  non_max_suppression_.setArg(2, theta_);
  non_max_suppression_.setArg(3, input_.rows);
  non_max_suppression_.setArg(4, input_.cols);

  // 1,1 offset and -2 to to dimensions so that we don't run on edge pixels.
  queue_.enqueueNDRangeKernel(non_max_suppression_, cl::NDRange(1, 1),
                              cl::NDRange(input_.rows - 2, input_.cols - 2),
                              cl::NDRange(workgroup_size_, workgroup_size_),
                              NULL);

    AdvanceBuff();
}

// enqueues the hysteresisThresholding kernel
void OpenclImageProcessor::HysteresisThresholding() {
  hysteresis_thresholding_.setArg(0, PrevBuff());
  hysteresis_thresholding_.setArg(1, NextBuff());
  hysteresis_thresholding_.setArg(2, input_.rows);
  hysteresis_thresholding_.setArg(3, input_.cols);

  // 1,1 offset and -2 to to dimensions so that we don't run on edge pixels.
  queue_.enqueueNDRangeKernel(hysteresis_thresholding_, cl::NDRange(1, 1),
                              cl::NDRange(input_.rows - 2, input_.cols - 2),
                              cl::NDRange(workgroup_size_, workgroup_size_),
                              NULL);

    AdvanceBuff();
}

// enqueues all of the canny stages
void OpenclImageProcessor::Canny() {
  Gaussian();
  Sobel();
  NonMaxSuppression();
  HysteresisThresholding();
}
}
