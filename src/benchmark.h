#pragma once

#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include "autotimer.h"
#include "imageprocessor.h"

// Stores basic information about an input image
struct InputImage
{
    std::string filename;
    int height;
    int width;
    InputImage(std::string _filename, int _width, int _height) :
        filename(_filename),
        height(_height),
        width(_width) { }
    
    double MegaPixels() { return (height * width) / 1000000.0; }
};

// Holds all results collected. In general, these results are averaged over
// more than one iteration.
struct ResultSet
{
    double average;
    double standard_deviation;
    std::vector<double> stage_times;
};

// Provides the ability to benchmark our canny edge detection algorithm.
// This could probably stand to be abstracted so that it could work for the
// other types of benchmarks we need to perform (serial and opencv's gpu
// implementation). Probably turn this into an abstract base and derive from it.
class Benchmark
{
    std::shared_ptr<ImageProcessor> processor;
    cv::Mat image;
    InputImage input;
    ResultSet results;
    std::string path;
    std::string title;
    int iterations;

    void runFullAlogirithm();
    void runComponents();

public:
    Benchmark(std::string _title, 
              std::shared_ptr<ImageProcessor> _processor,
              std::string _path,
              InputImage _input,
              int _iterations) :
        processor(_processor),
        input(_input),
        path(_path),
        title(_title),
        iterations(_iterations)
    {
        image = cv::imread(path + input.filename,
                           CV_LOAD_IMAGE_GRAYSCALE);
    }

    void Run();
    void OutputResults();
    ResultSet Results() { return results; }
};
