#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include "imageprocessor.h"
//#include <opencv2/imgproc/imgproc.hpp>

class SerialImageProcessor : public ImageProcessor
{
  protected:
    cv::Mat theta;
    cv::Mat buffers[2];
    unsigned int nextIndex = 0;

    // Private Methods
    // nextBuff returns a reference to the next buffer that should be modified.
    cv::Mat& nextBuff() { return buffers[nextIndex]; }
    // prevBuff returns a reference to the previous buffer that was modified.
    cv::Mat& prevBuff() { return buffers[nextIndex ^ 1]; }
    // Advance the buffer. Note there's only two, so right now it just swaps
    // to the other buffer right now.
    void advanceBuff() { nextIndex ^= 1; }

  public:
    SerialImageProcessor();

    virtual void LoadImage(cv::Mat &input);
    virtual cv::Mat GetOutput();

    // These methods are blocking calls which will perform what their name
    // implies. There are two sets, the first act upon internal buffers,
    // the second only act upon buffers passed in.
    virtual void Gaussian();
    virtual void Sobel();
    virtual void NonMaxSuppression();
    virtual void HysteresisThresholding();

    virtual void Gaussian(cv::Mat data, cv::Mat out);
    virtual void Sobel(cv::Mat data, cv::Mat out, cv::Mat theta);
    virtual void NonMaxSuppression(cv::Mat data, cv::Mat out, cv::Mat theta);
    virtual void HysteresisThresholding(cv::Mat data, cv::Mat out);
};
