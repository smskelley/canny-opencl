#pragma once
#define __CL_ENABLE_EXCEPTIONS

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cl.hpp"
#include "imageprocessor.h"

namespace ImageProcessors {

// Implements the Canny Edge Detection algorithm in OpenCL. Can be run on either
// the CPU or GPU. If run on the GPU, it will try to select a discrete GPU over
// an integrated GPU.
class OpenclImageProcessor : public ImageProcessor {
  // OpenCL Objects
  std::vector<cl::Platform> platforms_;
  std::vector<cl::Device> devices_;
  cl::Device selected_device_;
  cl::Context context_;
  cl::CommandQueue queue_;

  // OpenCL kernels
  cl::Kernel gaussian_;
  cl::Kernel sobel_;
  cl::Kernel non_max_suppression_;
  cl::Kernel hysteresis_thresholding_;

  // Create a buffer to hold the direction angle theta
  cl::Buffer theta_;

  // Note that existing code assumes only two buffers exist
  cl::Buffer buffers_[2];

  // Keeps track of the next buffer to use as a destination. This should
  // not be accessed directly, instead look at using nextBuff/prevBuff.
  size_t buffer_index_ = 0;

  // Determines the NDRange workgroup size.
  int workgroup_size_ = 1;

  // Private Methods

  // nextBuff returns a reference to the next buffer that should be modified.
  inline cl::Buffer& NextBuff() { return buffers_[buffer_index_]; }

  // prevBuff returns a reference to the previous buffer that was modified.
  inline cl::Buffer& PrevBuff() { return buffers_[buffer_index_ ^ 1]; }

  // Advance the buffer. Note there's only two, so right now it just swaps
  // to the other buffer.
  inline void AdvanceBuff() { buffer_index_ ^= 1; }

  // returns the "desirable" device. If a discrete GPU is detected, then it
  // will be preferred over integrated graphics. If devices is empty, then the
  // this will throw std::out_of_range.
  cl::Device& GetBestDevice();

  // Return the relative path to the cpu or gpu  kernel given a filename
  // e.g. KernelPath("mykernel.cl", true); // returns: kernels/gpu/mykerne.cl
  std::string KernelPath(std::string filename, bool use_gpu);

  // Given a filename (without its path) load and return the kernel.
  cl::Kernel LoadKernel(std::string filename, std::string kernel_name,
                        bool use_gpu);

 public:
  // If use_gpu is true, this image processor will attempt to use the GPU
  OpenclImageProcessor(bool use_gpu = true);

  // outputs basic information about the device in use.
  void DeviceInfo();

  // Loads image. Note that input is assumed to be 8 bit 1 channel grayscale.
  void LoadImage(cv::Mat& input);

  // Block until all other operations are complete and then return the cv::Mat
  // corresponding to the output of previously enqueued operations.
  cv::Mat output();

  // Blocking call which finishes all commands in queue. Useful for
  // benchmarking purposes, so that you can time a subset of an operation.
  void FinishJobs();

  // Enqueue the Gaussian blur operation. This is a non-blocking call.
  void Gaussian();

  // Enqueue the Sobel Filtering operation. This is a non-blocking call.
  void Sobel();

  // Enqueue the Non-maximum suppression operation. This is a non-blocking call.
  void NonMaxSuppression();

  // Enqueue the Hysteresis Thresholding operation. This is a non-blocking call.
  void HysteresisThresholding();

  // Enqueue all stages. This is a non-blocking call.
  void Canny();
};

}
