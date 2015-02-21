#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include "imageprocessor.h"
//#include <opencv2/imgproc/imgproc.hpp>

class CvImageProcessor : public ImageProcessor {
 public:
  CvImageProcessor();

  virtual void Canny();

  // For the OpenCV Image Processor, these are all blank
  virtual void Gaussian() {}
  virtual void Sobel() {}
  virtual void NonMaxSuppression() {}
  virtual void HysteresisThresholding() {}
};
