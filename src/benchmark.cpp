#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include "autotimer.h"
#include "benchmark.h"
#include "imageprocessor.h"

using namespace std;


///////// Public Methods ///////////////////////////////////////////////////////
void Benchmark::Run()
{
    runFullAlogirithm();
    runComponents();
}

void Benchmark::OutputResults()
{
    // I wish these could be merged via a templated lambda, but that doesn't
    // exist.
    auto outputDouble = [](string heading, double value) {
        cout << left << setw(10) << heading << value << endl;
    };

    auto outputString = [](string heading, string value) {
        cout << left << setw(10) << heading << value << endl;
    };

    outputString("Title:", title);
    outputString("File:", input.filename);
    outputDouble("Megapixels:", input.MegaPixels());
    outputDouble("Average:", results.average);
    outputDouble("StDev:", results.standard_deviation);
    outputDouble("Kpx/ms:", (input.height * input.width) /
                         (1000 * results.average));

    for (int i = 0; i < results.stage_times.size(); i++)
        outputDouble("Stage " + to_string(i), results.stage_times[i]);
}

///////// Private Methods //////////////////////////////////////////////////////

// Runs the full algorithm 'iteration' times. Finds the average and standard
// deviation of its runtime.
void Benchmark::runFullAlogirithm()
{
    double total_duration = 0;

    // each iteration duration is squared and then added to squared_durations.
    double squared_durations = 0;

    // time it however many times
    for (int i = 0; i < iterations; i++)
    {
        double duration = 0;
        processor->LoadImage(image);

        // Make sure that image is loaded before beginning the benchmark.
        processor->FinishJobs();
        {
            AutoTimer timer;
            processor->Canny();
            processor->FinishJobs();
            duration = timer.Duration();
        }
        total_duration += duration;
        squared_durations += duration * duration;
    }

    // record the average
    results.average = total_duration / iterations;

    // record the standard deviation
    results.standard_deviation = sqrt(squared_durations / iterations -
                                      results.average * results.average);

    // write the generated image. Optional, but helps us verify that everything
    // ran correctly.
    cv::imwrite(path + "canny_" + input.filename, processor->GetOutput());
}

// Times each stage separately.
void Benchmark::runComponents()
{
    processor->LoadImage(image);
    processor->FinishJobs();

    // Stage 1: Gaussian Blur
    {
        AutoTimer timer;
        processor->Gaussian();
        processor->FinishJobs();
        results.stage_times.push_back(timer.Duration());
    }

    // Stage 2: Sobel Filtering
    {
        AutoTimer timer;
        processor->Sobel();
        processor->FinishJobs();
        results.stage_times.push_back(timer.Duration());
    }

    // Stage 3: Nonmaximum Suppression
    {
        AutoTimer timer;
        processor->NonMaxSuppression();
        processor->FinishJobs();
        results.stage_times.push_back(timer.Duration());
    }

    // Stage 4: Hysteresis Thresholding
    {
        AutoTimer timer;
        processor->HysteresisThresholding();
        processor->FinishJobs();
        results.stage_times.push_back(timer.Duration());
    }
}
