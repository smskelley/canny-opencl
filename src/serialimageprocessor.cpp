#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <iostream>
#include "serialimageprocessor.h"

using namespace std;

SerialImageProcessor::SerialImageProcessor()
{

}

const float gaus[3][3] = { {0.0625, 0.125, 0.0625},
                           {0.1250, 0.125, 0.1250},
                           {0.0625, 0.125, 0.0625} };

const int sobx[3][3] = { {-1, 0, 1},
                         {-2, 0, 2},
                         {-1, 0, 1} };

const int soby[3][3] = { {-1,-2,-1},
                         { 0, 0, 0},
                         { 1, 2, 1} };

void SerialImageProcessor::LoadImage(cv::Mat &input)
{
    ImageProcessor::LoadImage(input);
    nextBuff() = input;
    prevBuff() = cv::Mat(input.rows, input.cols, CV_8UC1);
    theta = cv::Mat(input.rows, input.cols, CV_8UC1);
    advanceBuff();
}

cv::Mat SerialImageProcessor::GetOutput()
{
    return prevBuff();
}


// These methods are blocking calls which will perform what their name
// implies
void SerialImageProcessor::Gaussian()
{
    Gaussian(prevBuff(), nextBuff());
    advanceBuff();
}

void SerialImageProcessor::Gaussian(cv::Mat data, cv::Mat out)
{
    size_t rows = data.rows;
    size_t cols = data.cols;

    // iterate over the rows of the photo matrix
    for (int row = 1; row < rows - 1; row++)
    {
        // iterate over the columns of the photo matrix
        for (int col = 1; col < cols - 1; col++)
        {
            int sum = 0;

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    sum += gaus[i][j] *
                           data.at<uchar>(row + j - 1, col + i - 1);
                }
            }
            out.at<uchar>(row, col) = min(255,max(0,sum));
        }
    }
}

void SerialImageProcessor::Sobel()
{
    Sobel(prevBuff(), nextBuff(), theta);
    advanceBuff();
}

void SerialImageProcessor::Sobel(cv::Mat data, cv::Mat out, cv::Mat theta)
{
    // collect sums separately. we're storing them into floats because that
    // is what hypot and atan2 will expect.
    const float PI = 3.14159265;
    size_t rows = data.rows;
    size_t cols = data.cols;

    
    // iterate over the rows of the photo matrix
    for (int row = 1; row < rows - 1; row++)
    {
        // iterate over the columns of the photo matrix
        for (int col = 1; col < cols - 1; col++)
        {
            float sumx = 0, sumy = 0, angle = 0;
            // find x and y derivatives
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    sumx += sobx[i][j] *
                            data.at<uchar>(row + j - 1, col + i - 1);
                    sumy += soby[i][j] *
                            data.at<uchar>(row + j - 1, col + i - 1);
                }
            }
            
            // The output is now the square root of their squares, but they are
            // constrained to 0 <= value <= 255. Note that hypot is a built in
            // function defined as: hypot(x,y) = sqrt(x*x, y*y).
            out.at<uchar>(row,col) = min(255, max(0, (int)hypot(sumx,sumy) ));

            // Compute the direction angle theta in radians
            // atan2 has a range of (-PI, PI) degrees
            angle = atan2(sumy,sumx);
            
            // If the angle is negative,
            // shift the range to (0, 2PI) by adding 2PI to the angle,
            // then perform modulo operation of 2PI
            if (angle < 0)
            {
                angle = fmod((angle + 2*PI),(2*PI));
            }
            
            // Round the angle to one of four possibilities: 0, 45, 90, 135
            // degrees then store it in the theta buffer at the proper position
            if (angle <= PI/8)
            {
                theta.at<uchar>(row,col) = 0;
            }
            else if (angle <= 3*PI/8)
            {
                theta.at<uchar>(row,col) = 45;
            }
            else if (angle <= 5*PI/8)
            {
                theta.at<uchar>(row,col) = 90;
            }
            else if (angle <= 7*PI/8)
            {
                theta.at<uchar>(row,col) = 135;
            }
            else if (angle <= 9*PI/8)
            {
                theta.at<uchar>(row,col) = 0;
            }
            else if (angle <= 11*PI/8)
            {
                theta.at<uchar>(row,col) = 45;
            }
            else if (angle <= 13*PI/8)
            {
                theta.at<uchar>(row,col) = 90;
            }
            else if (angle <= 15*PI/8)
            {
                theta.at<uchar>(row,col) = 135;
            }
            else // (angle <= 16*PI/8)
            {
                theta.at<uchar>(row,col) = 0;
            }
        }
    }
}

void SerialImageProcessor::NonMaxSuppression()
{
    NonMaxSuppression(prevBuff(), nextBuff(), theta);
    advanceBuff();
}

void SerialImageProcessor::NonMaxSuppression(cv::Mat data,
                                             cv::Mat out,
                                             cv::Mat theta)
{
    size_t rows = data.rows;
    size_t cols = data.cols;
    
    // iterate over the rows of the photo matrix
    for (int row = 1; row < rows - 1; row++)
    {
        // iterate over the columns of the photo matrix
        for (int col = 1; col < cols - 1; col++)
        {
            
            // These variables are used to address the matrices more easily
            const unsigned char DATA_POS = data.at<uchar>(row,col);
            const unsigned char DATA_N = data.at<uchar>(row-1,col);
            const unsigned char DATA_NE = data.at<uchar>(row-1,col+1);
            const unsigned char DATA_E = data.at<uchar>(row,col+1);
            const unsigned char DATA_SE = data.at<uchar>(row+1,col+1);
            const unsigned char DATA_S = data.at<uchar>(row+1,col);
            const unsigned char DATA_SW = data.at<uchar>(row+1,col-1);
            const unsigned char DATA_W = data.at<uchar>(row,col-1);
            const unsigned char DATA_NW = data.at<uchar>(row-1,col-1);
            const unsigned char THETA_POS = theta.at<uchar>(row,col);
            
            switch (THETA_POS)
            {
                // A gradient angle of 0 degrees = an edge that is North/South
                // Check neighbors to the East and West
                case 0:
                    // supress me if my neighbor has larger magnitude
                    if (DATA_POS <= DATA_E || DATA_POS <= DATA_W)
                    {
                        out.at<uchar>(row,col) = 0;
                    }
                    // otherwise, copy my value to the output buffer
                    else
                    {
                        out.at<uchar>(row,col) = DATA_POS;
                    }
                    break;
                    
                // A gradient angle of 45 degrees = an edge that is NW/SE
                // Check neighbors to the NE and SW
                case 45:
                    // supress me if my neighbor has larger magnitude
                    if (DATA_POS <= DATA_NE || DATA_POS <= DATA_SW)
                    {
                        out.at<uchar>(row,col) = 0;
                    }
                    // otherwise, copy my value to the output buffer
                    else
                    {
                        out.at<uchar>(row,col) = DATA_POS;
                    }
                    break;
                    
                // A gradient angle of 90 degrees = an edge that is E/W
                // Check neighbors to the North and South.
                case 90:
                    // supress me if my neighbor has larger magnitude
                    if (DATA_POS <= DATA_N || DATA_POS <= DATA_S)
                    {
                        out.at<uchar>(row,col) = 0;
                    }
                    // otherwise, copy my value to the output buffer
                    else
                    {
                        out.at<uchar>(row,col) = DATA_POS;
                    }
                    break;
                    
                // A gradient angle of 135 degrees = an edge that is NE/SW
                // Check neighbors to the NW and SE
                case 135:
                    // supress me if my neighbor has larger magnitude
                    if (DATA_POS <= DATA_NW || DATA_POS <= DATA_SE)
                    {
                        out.at<uchar>(row,col) = 0;
                    }
                    // otherwise, copy my value to the output buffer
                    else
                    {
                        out.at<uchar>(row,col) = DATA_POS;
                    }
                    break;
                    
                defaut:
                    out.at<uchar>(row,col) = DATA_POS;
                    break;
            } 
        }
    }
}

void SerialImageProcessor::HysteresisThresholding()
{
<<<<<<< HEAD
    
=======
    HysteresisThresholding(prevBuff(), nextBuff());
    // commented out until HysteresisThresholding is implemented
    //advanceBuff();
}

void SerialImageProcessor::HysteresisThresholding(cv::Mat data, cv::Mat out)
{

>>>>>>> c2979cb56744ed3ccc32da5c897d2dd449650153
}

