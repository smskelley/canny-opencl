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
    string path = "images/";
    string filename = "Great_tit.jpg";

    // if first param is 'cpu', then image processor should use the cpu.
    bool useGPU = true;
    if (argc > 1 && strcmp(argv[1], "cpu") == 0)
        useGPU = false;

    // Load the image
    ImageProcessor processor(useGPU);
    cv::Mat image = cv::imread(path + filename, CV_LOAD_IMAGE_GRAYSCALE);
    processor.LoadImage(image);

    // Make sure that image is loaded before beginning the benchmark.
    processor.FinishJobs();

    {
        AutoTimer timer(onTimerFinish);
        processor.Canny();
        processor.FinishJobs();
    }

    // This is optional. Note that this is saving out to a jpg.
    cv::imwrite(path + "canny_" + filename, processor.GetOutput());

    return 0;
}
