#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace std;

int main()
{
    // 0 is first video capture device, generally the webcam
    cv::VideoCapture cam(0);

    // make sure we were able to open it
    if (!cam.isOpened())
        cout << "cannot open camera";

    // If we re-use frames, then we only need two. Perhaps it would be better
    // to have more so that we may give them better names.
    cv::Mat frameA, frameB;
    while (true)
    {
        cam.read(frameA);
        cv::cvtColor(frameA, frameB, cv::COLOR_BGR2GRAY);
        cv::Canny(frameB, frameA, 40, 80);
        cv::imshow("canny", frameA);
 
        // key 30 is escape
        if (cv::waitKey(30) >= 0)
            break;
    }

    return 0;
}
