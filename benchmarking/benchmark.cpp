#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include "autotimer.h"
#include "benchmark.h"

using namespace std;

namespace Benchmarking {
///////// Public Methods ///////////////////////////////////////////////////////
void Benchmark::Run() {
  RunFullAlogirithm();
  RunComponents();
}

void Benchmark::OutputResults() {
  // I wish these could be merged via a templated lambda, but that doesn't
  // exist.
  auto outputDouble = [](string heading, double value) {
    cout << left << setw(10) << heading << value << endl;
  };

  auto outputString = [](string heading, string value) {
    cout << left << setw(10) << heading << value << endl;
  };

  outputString("Title:", title_);
  outputString("File:", input_.filename);
  outputDouble("Megapixels:", input_.MegaPixels());
  outputDouble("Average:", results_.average);
  outputDouble("StDev:", results_.standard_deviation);
  outputDouble("Kpx/ms:",
               (input_.height * input_.width) / (1000 * results_.average));

  for (int i = 0; i < results_.stage_times.size(); i++)
    outputDouble("Stage " + to_string(i), results_.stage_times[i]);
}

///////// Private Methods //////////////////////////////////////////////////////

// Runs the full algorithm 'iteration' times. Finds the average and standard
// deviation of its runtime.
void Benchmark::RunFullAlogirithm() {
  double total_duration = 0;

  // each iteration duration is squared and then added to squared_durations.
  double squared_durations = 0;

  // time it however many times
  for (int i = 0; i < iterations_; i++) {
    double duration = 0;
    processor_->LoadImage(image_);

    // Make sure that image is loaded before beginning the benchmark.
    processor_->FinishJobs();
    {
      AutoTimer timer;
      processor_->Canny();
      processor_->FinishJobs();
      duration = timer.Duration();
    }
    total_duration += duration;
    squared_durations += duration * duration;
  }

  // record the average
  results_.average = total_duration / iterations_;

  // record the standard deviation
  results_.standard_deviation =
      sqrt(squared_durations / iterations_ - results_.average * results_.average);

  // write the generated image. Optional, but helps us verify that everything
  // ran correctly.
  cv::imwrite(path_ + "canny_" + input_.filename, processor_->output());
}

// Times each stage separately.
void Benchmark::RunComponents() {
  processor_->LoadImage(image_);
  processor_->FinishJobs();

  // Stage 1: Gaussian Blur
  {
    AutoTimer timer;
    processor_->Gaussian();
    processor_->FinishJobs();
    results_.stage_times.push_back(timer.Duration());
  }

  // Stage 2: Sobel Filtering
  {
    AutoTimer timer;
    processor_->Sobel();
    processor_->FinishJobs();
    results_.stage_times.push_back(timer.Duration());
  }

  // Stage 3: Nonmaximum Suppression
  {
    AutoTimer timer;
    processor_->NonMaxSuppression();
    processor_->FinishJobs();
    results_.stage_times.push_back(timer.Duration());
  }

  // Stage 4: Hysteresis Thresholding
  {
    AutoTimer timer;
    processor_->HysteresisThresholding();
    processor_->FinishJobs();
    results_.stage_times.push_back(timer.Duration());
  }
}
}
