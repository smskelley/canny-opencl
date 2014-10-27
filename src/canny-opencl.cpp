#include <iostream>
#include <cstring>
#include <opencv2/highgui/highgui.hpp>
#include "autotimer.h"
#include "imageprocessor.h"

using namespace std;

void onTimerFinish(double time)
{
    cout << "Took " << time << " milliseconds.\n";
}

int main(int argc, char *argv[])
{
    cv::VideoCapture webcam(0);
    cv::Mat inFrame, grayFrame;
    bool useGPU = true;
    if (argc > 1 && strcmp(argv[1], "cpu") == 0)
        useGPU = false;

    ImageProcessor processor(useGPU);
    while (true)
    {
        webcam.read(inFrame);
        cv::cvtColor(inFrame, grayFrame, cv::COLOR_BGR2GRAY);

        processor.LoadImage(grayFrame);

        processor.FinishJobs();
        {
            AutoTimer timer(onTimerFinish);
            processor.Canny();
            processor.FinishJobs();
        }

        imshow("canny", processor.GetOutput());
        if (cv::waitKey(30) >= 0)
            break;
    }

    return 0;
}
