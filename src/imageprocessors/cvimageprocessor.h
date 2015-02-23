#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include "imageprocessor.h"

namespace ImageProcessors {

// OpenCV implementation of the canny edge detection algorithm. Note that
// OpenCV does not support running each stage in isolation, so those methods
// do nothing.
class CvImageProcessor : public ImageProcessor {
 public:
  CvImageProcessor();

  // Run OpenCV's canny edge detection algorithm
  virtual void Canny();

  // For the OpenCV Image Processor, these are all blank, since we cannot
  // run each stage in isolation.
  virtual void Gaussian() {}
  virtual void Sobel() {}
  virtual void NonMaxSuppression() {}
  virtual void HysteresisThresholding() {}
};
}
