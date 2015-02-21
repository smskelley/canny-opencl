#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include "serialimageprocessor.h"

namespace ImageProcessors {

using std::min;
using std::max;

const float gaus[3][3] = {
    {0.0625, 0.125, 0.0625}, {0.1250, 0.125, 0.1250}, {0.0625, 0.125, 0.0625}};

const int sobx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

const int soby[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

// Contructor
SerialImageProcessor::SerialImageProcessor() {}

void SerialImageProcessor::LoadImage(cv::Mat &input) {
  ImageProcessor::LoadImage(input);
  NextBuff() = input;
  PrevBuff() = cv::Mat(input.rows, input.cols, CV_8UC1);
  theta_ = cv::Mat(input.rows, input.cols, CV_8UC1);
  AdvanceBuff();
}

cv::Mat SerialImageProcessor::output() { return PrevBuff(); }

// These methods are blocking calls which will perform what their name
// implies
void SerialImageProcessor::Gaussian() {
  Gaussian(PrevBuff(), NextBuff());
  AdvanceBuff();
}

void SerialImageProcessor::Gaussian(cv::Mat data, cv::Mat out) {
  size_t rows = data.rows;
  size_t cols = data.cols;

  // iterate over the rows of the photo matrix
  for (int row = 1; row < rows - 1; row++) {
    // iterate over the columns of the photo matrix
    for (int col = 1; col < cols - 1; col++) {
      int sum = 0;
      size_t pos = row * cols + col;

      // convolve the photo with the gaussian filter
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          sum += gaus[i][j] * data.at<uchar>(row + i - 1, col + j - 1);
        }
      }
      out.data[pos] = min(255, max(0, sum));
    }
  }
}

void SerialImageProcessor::Sobel() {
  Sobel(PrevBuff(), NextBuff(), theta_);
  AdvanceBuff();
}

void SerialImageProcessor::Sobel(cv::Mat data, cv::Mat out, cv::Mat theta) {
  // collect sums separately. we're storing them into floats because that
  // is what hypot and atan2 will expect.
  const float PI = 3.14159265;
  size_t rows = data.rows;
  size_t cols = data.cols;

  // iterate over the rows of the photo matrix
  for (int row = 1; row < rows - 1; row++) {
    // iterate over the columns of the photo matrix
    for (int col = 1; col < cols - 1; col++) {
      float sumx = 0, sumy = 0, angle = 0;
      size_t pos = row * cols + col;

      // find x and y derivatives
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          sumx += sobx[i][j] * data.at<uchar>(row + i - 1, col + j - 1);
          sumy += soby[i][j] * data.at<uchar>(row + i - 1, col + j - 1);
        }
      }

      // The output is now the square root of their squares, but they are
      // constrained to 0 <= value <= 255. Note that hypot is a built in
      // function defined as: hypot(x,y) = sqrt(x*x, y*y).
      out.data[pos] = min(255, max(0, (int)hypot(sumx, sumy)));

      // Compute the direction angle theta in radians
      // atan2 has a range of (-PI, PI) degrees
      angle = atan2(sumy, sumx);

      // If the angle is negative,
      // shift the range to (0, 2PI) by adding 2PI to the angle,
      // then perform modulo operation of 2PI
      if (angle < 0) {
        angle = fmod((angle + 2 * PI), (2 * PI));
      }

      // Round the angle to one of four possibilities: 0, 45, 90, 135
      // degrees then store it in the theta buffer at the proper position
      if (angle <= PI / 8) {
        theta.data[pos] = 0;
      } else if (angle <= 3 * PI / 8) {
        theta.data[pos] = 45;
      } else if (angle <= 5 * PI / 8) {
        theta.data[pos] = 90;
      } else if (angle <= 7 * PI / 8) {
        theta.data[pos] = 135;
      } else if (angle <= 9 * PI / 8) {
        theta.data[pos] = 0;
      } else if (angle <= 11 * PI / 8) {
        theta.data[pos] = 45;
      } else if (angle <= 13 * PI / 8) {
        theta.data[pos] = 90;
      } else if (angle <= 15 * PI / 8) {
        theta.data[pos] = 135;
      } else  // (angle <= 16*PI/8)
      {
        theta.data[pos] = 0;
      }
    }
  }
}

void SerialImageProcessor::NonMaxSuppression() {
  NonMaxSuppression(PrevBuff(), NextBuff(), theta_);
  AdvanceBuff();
}

void SerialImageProcessor::NonMaxSuppression(cv::Mat data, cv::Mat out,
                                             cv::Mat theta) {
  size_t rows = data.rows;
  size_t cols = data.cols;

  // iterate over the rows of the photo matrix
  for (int row = 1; row < rows - 1; row++) {
    // iterate over the columns of the photo matrix
    for (int col = 1; col < cols - 1; col++) {
      // These variables are used to address the matrices more easily
      const size_t POS = row * cols + col;
      const size_t N = (row - 1) * cols + col;
      const size_t NE = (row - 1) * cols + (col + 1);
      const size_t E = row * cols + (col + 1);
      const size_t SE = (row + 1) * cols + (col + 1);
      const size_t S = (row + 1) * cols + col;
      const size_t SW = (row + 1) * cols + (col - 1);
      const size_t W = row * cols + (col - 1);
      const size_t NW = (row - 1) * cols + (col - 1);

      switch (theta.data[POS]) {
        // A gradient angle of 0 degrees = an edge that is North/South
        // Check neighbors to the East and West
        case 0:
          // supress me if my neighbor has larger magnitude
          if (data.data[POS] <= data.data[E] ||
              data.data[POS] <= data.data[W]) {
            out.data[POS] = 0;
          }
          // otherwise, copy my value to the output buffer
          else {
            out.data[POS] = data.data[POS];
          }
          break;

        // A gradient angle of 45 degrees = an edge that is NW/SE
        // Check neighbors to the NE and SW
        case 45:
          // supress me if my neighbor has larger magnitude
          if (data.data[POS] <= data.data[NE] ||
              data.data[POS] <= data.data[SW]) {
            out.data[POS] = 0;
          }
          // otherwise, copy my value to the output buffer
          else {
            out.data[POS] = data.data[POS];
          }
          break;

        // A gradient angle of 90 degrees = an edge that is E/W
        // Check neighbors to the North and South.
        case 90:
          // supress me if my neighbor has larger magnitude
          if (data.data[POS] <= data.data[N] ||
              data.data[POS] <= data.data[S]) {
            out.data[POS] = 0;
          }
          // otherwise, copy my value to the output buffer
          else {
            out.data[POS] = data.data[POS];
          }
          break;

        // A gradient angle of 135 degrees = an edge that is NE/SW
        // Check neighbors to the NW and SE
        case 135:
          // supress me if my neighbor has larger magnitude
          if (data.data[POS] <= data.data[NW] ||
              data.data[POS] <= data.data[SE]) {
            out.data[POS] = 0;
          }
          // otherwise, copy my value to the output buffer
          else {
            out.data[POS] = data.data[POS];
          }
          break;

        default:
          out.data[POS] = data.data[POS];
          break;
      }
    }
  }
}

void SerialImageProcessor::HysteresisThresholding() {
  HysteresisThresholding(PrevBuff(), NextBuff());

  AdvanceBuff();
}

void SerialImageProcessor::HysteresisThresholding(cv::Mat data, cv::Mat out) {
  float tHigh = 70;
  float tLow = 30;
  float median = (tLow + tHigh) / 2;
  const float EDGE = 255;
  const float NOEDGE = 0;

  size_t rows = data.rows;
  size_t cols = data.cols;

  // iterate over the rows of the photo matrix
  for (int row = 1; row < rows - 1; row++) {
    // iterate over the columns of the photo matrix
    for (int col = 1; col < cols - 1; col++) {
      if (data.at<uchar>(row, col) >= tHigh) {
        out.at<uchar>(row, col) = EDGE;
      } else if (data.at<uchar>(row, col) <= tLow) {
        out.at<uchar>(row, col) = NOEDGE;
      } else if (data.at<uchar>(row, col) > median) {
        out.at<uchar>(row, col) = EDGE;
      } else {
        out.at<uchar>(row, col) = NOEDGE;
      }
    }
  }
}

}
