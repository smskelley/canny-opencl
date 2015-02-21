// This application runs the desired canny algorithm on webcam data,
// displaying it back to the user in near real time.
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "autotimer.h"
#include "imageprocessor.h"
#include "openclimageprocessor.h"
#include "serialimageprocessor.h"

using namespace std;

// Called when timer finishes, output the amount of time it took.
void onTimerFinish(double time) {
  cout << "Took " << time << " milliseconds.\n";
}

int main(int argc, char *argv[]) {
  cv::VideoCapture webcam(0);
  cv::Mat in_frame, gray_frame;
  bool use_gpu = true;
  bool use_parallel = true;
  unique_ptr<ImageProcessor> processor(nullptr);

  // parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "cpu") == 0)
      use_gpu = false;
    else if (strcmp(argv[i], "serial") == 0)
      use_parallel = false;
  }

  // Create and load the appropriate image processor
  if (use_parallel)
    processor.reset(new OpenclImageProcessor(use_gpu));
  else
    processor.reset(new SerialImageProcessor());

  // grab new frame, convert to grayscale, detect edges, then display the result.
  while (true) {
    webcam.read(in_frame);
    cv::cvtColor(in_frame, gray_frame, cv::COLOR_BGR2GRAY);

    processor->LoadImage(gray_frame);

    processor->FinishJobs();
    {
      AutoTimer timer(onTimerFinish);
      processor->Canny();
      processor->FinishJobs();
    }

    imshow("canny", processor->output());
    if (cv::waitKey(30) >= 0) break;
  }

  return 0;
}
