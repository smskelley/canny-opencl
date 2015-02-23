#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cvimageprocessor.h"

using namespace std;

namespace ImageProcessors {

// no setup is needed.
CvImageProcessor::CvImageProcessor() {}

// Runs OpenCV's canny edge detection with 40, 80 being low, high thresholds
void CvImageProcessor::Canny() {
  cv::Canny(input_, output_, 40, 80);
}

}
