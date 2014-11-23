#include <opencv2/highgui/highgui.hpp>
#include "imageprocessor.h"
//#include <opencv2/imgproc/imgproc.hpp>

class SerialImageProcessor : public ImageProcessor
{
  public:
    SerialImageProcessor();

    // These methods are blocking calls which will perform what their name
    // implies
    virtual void Gaussian();
    virtual void Sobel();
    virtual void NonMaxSuppression();
    virtual void HysteresisThresholding();
};
