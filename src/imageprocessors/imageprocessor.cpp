#include <cassert>
#include <regex>
#include "imageprocessor.h"

namespace ImageProcessors {

/***  Public Methods **********************************************************/

// load the 8bit 1channel grayscale Mat and setup the output matrix
void ImageProcessor::LoadImage(cv::Mat &input) {
  this->input_ = input;
  output_ = cv::Mat(input.rows, input.cols, CV_8UC1);
}

// Returns the output image which should always be the same size as the input.
cv::Mat ImageProcessor::output() {
  assert(output_.rows == input_.rows && output_.cols == input_.cols);
  return output_;
}

// Executes all of the canny stages
void ImageProcessor::Canny() {
  Gaussian();
  Sobel();
  NonMaxSuppression();
  HysteresisThresholding();
}

}
