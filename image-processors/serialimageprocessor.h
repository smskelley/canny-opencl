#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include "imageprocessor.h"

class SerialImageProcessor : public ImageProcessor {
 protected:
  cv::Mat theta_;
  cv::Mat buffers_[2];
  unsigned int next_index_ = 0;

  // Private Methods
  // nextBuff returns a reference to the next buffer that should be modified.
  cv::Mat& NextBuff() { return buffers_[next_index_]; }
  // prevBuff returns a reference to the previous buffer that was modified.
  cv::Mat& PrevBuff() { return buffers_[next_index_ ^ 1]; }
  // Advance the buffer. Note there's only two, so right now it just swaps
  // to the other buffer right now.
  void AdvanceBuff() { next_index_ ^= 1; }

 public:
  SerialImageProcessor();

  virtual void LoadImage(cv::Mat& input);
  virtual cv::Mat output();

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
