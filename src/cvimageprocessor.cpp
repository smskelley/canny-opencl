#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iostream>
#include "cvimageprocessor.h"

using namespace std;

CvImageProcessor::CvImageProcessor()
{

}

void CvImageProcessor::Canny()
{
    cv::Canny(input, output, 40, 80);
}
