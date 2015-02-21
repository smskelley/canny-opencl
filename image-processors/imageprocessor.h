#pragma once

#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

class ImageProcessor {
 protected:
  // OpenCV Objects
  cv::Mat input_;
  cv::Mat output_;

 public:
  ImageProcessor() {}

  // Note that input matrices are assumed to be 8 bit 1 channel grayscale.
  virtual void LoadImage(cv::Mat &input);

  // Wait for all other operations to complete and then return the cv::Mat
  // corresponding to the output of previously enqueued operations.
  virtual cv::Mat output();

  // For parallel implementations, this is a blocking call which finishes all
  // operations, acting as a barrier. This is useful for benchmarking
  // purposes, so that you can time a subset of an operation.
  // For serial versions, this will do nothing.
  virtual void FinishJobs() {}

  // These operations will enqueue the appropriate kernel. They are
  // non-blocking.
  virtual void Gaussian() = 0;
  virtual void Sobel() = 0;
  virtual void NonMaxSuppression() = 0;
  virtual void HysteresisThresholding() = 0;
  virtual void Canny();
};
