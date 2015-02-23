#pragma once

#include <opencv2/highgui/highgui.hpp>

namespace ImageProcessors {

// Abstract Base Class for image processors which implement the canny edge
// detection algorithm.
class ImageProcessor {
 protected:
  // OpenCV Objects.
  // Input matrix contains the input image
  cv::Mat input_;
  // Output matrix will contain the output image.
  cv::Mat output_;

 public:
  ImageProcessor() {}

  // Note that input matrices are assumed to be 8 bit 1 channel grayscale.
  virtual void LoadImage(cv::Mat &input);

  // Gets the output image.
  virtual cv::Mat output();

  // For parallel implementations, this is a blocking call which finishes all
  // operations, acting as a barrier. This is useful for benchmarking
  // purposes, so that you can time a subset of an operation.
  // For serial versions, this will do nothing.
  virtual void FinishJobs() {}

  // Execute Gaussian Blur
  virtual void Gaussian() = 0;

  // Execute Sobel Filtering
  virtual void Sobel() = 0;

  // Execute Non-Maximum Supression
  virtual void NonMaxSuppression() = 0;

  // Execute Hysteresis Thresholding
  virtual void HysteresisThresholding() = 0;

  // Execute all stages as quickly as possible
  virtual void Canny();
};
}
