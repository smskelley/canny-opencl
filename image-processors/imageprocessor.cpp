#include <cassert>
#include <cmath>
#include <regex>
#include <stdexcept>
#include "imageprocessor.h"

using namespace std;
/***  Public Methods **********************************************************/
// load the 8bit 1channel grayscale Mat
void ImageProcessor::LoadImage(cv::Mat &input) {
  this->input = input;
  output = cv::Mat(input.rows, input.cols, CV_8UC1);
}

// copies the buffer back and returns it. this is a blocking call.
cv::Mat ImageProcessor::GetOutput() {
  assert(output.rows == input.rows && output.cols == input.cols);
  return output;
}

// Executes all of the canny stages
void ImageProcessor::Canny() {
  Gaussian();
  Sobel();
  NonMaxSuppression();
  HysteresisThresholding();
}
