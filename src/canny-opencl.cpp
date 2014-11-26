#include <iostream>
#include <memory>
#include <cstring>
#include <opencv2/highgui/highgui.hpp>
#include "autotimer.h"
#include "imageprocessor.h"
#include "openclimageprocessor.h"
#include "serialimageprocessor.h"

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
    bool useParallel = true;
    unique_ptr<ImageProcessor> processor(nullptr);

    // parse command line arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "cpu") == 0)
            useGPU = false;
        else if (strcmp(argv[i], "serial") == 0)
            useParallel = false;
    }

    // Create and load the appropriate image processor
    if (useParallel)
        processor.reset(new OpenclImageProcessor(useGPU));
    else
        processor.reset(new SerialImageProcessor());


    while (true)
    {
        webcam.read(inFrame);
        cv::cvtColor(inFrame, grayFrame, cv::COLOR_BGR2GRAY);

        processor->LoadImage(grayFrame);

        processor->FinishJobs();
        {
            AutoTimer timer(onTimerFinish);
            processor->Canny();
            processor->FinishJobs();
        }
 
        imshow("canny", processor->GetOutput());
        if (cv::waitKey(30) >= 0)
            break;
    }

    return 0;
}
