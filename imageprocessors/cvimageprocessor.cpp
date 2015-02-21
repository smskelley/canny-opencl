#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cvimageprocessor.h"

using namespace std;

namespace ImageProcessors {
CvImageProcessor::CvImageProcessor() {}

void CvImageProcessor::Canny() { cv::Canny(input_, output_, 40, 80); }
}
