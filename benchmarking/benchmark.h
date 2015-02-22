#pragma once

#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include "autotimer.h"
#include "imageprocessor.h"

namespace Benchmarking
{
// Stores basic information about an input image
struct InputImage {
  std::string filename;
  int height;
  int width;
  InputImage(std::string _filename, int _width, int _height)
      : filename(_filename), height(_height), width(_width) {}

  double MegaPixels() { return (height * width) / 1000000.0; }
};

// Holds all results collected for a single benchmark. In general, these results
// are averaged over more than one iteration.
struct ResultSet {
  double average;
  double standard_deviation;
  std::vector<double> stage_times;
};

// Provides the ability to benchmark canny edge detection algorithms that
// inherit from ImageProcessor. Each benchmark relates to one image processor
// being run on one image. The full algorithm is run multiple times to find
// the average and standard deviation of run times. It is then run on each stage
// in isolation.
class Benchmark {
  std::shared_ptr<ImageProcessors::ImageProcessor> processor_;
  cv::Mat image_;
  InputImage input_;
  ResultSet results_;
  std::string path_;
  std::string title_;
  int iterations_;

  void RunFullAlogirithm();
  void RunComponents();

 public:
  // Construct a new benchmark using one image and one image processor.
  // title: The name of the benchmark which will appear when results are output.
  // processor: An image processor to benchmark
  // path: The path to the image file
  // input: InputImage object relating to the image to run on
  // iterations: The number of times to perform the full algorithm.
  Benchmark(std::string title, std::shared_ptr<ImageProcessors::ImageProcessor> processor,
            std::string path, InputImage input, int iterations)
      : processor_(processor),
        input_(input),
        path_(path),
        title_(title),
        iterations_(iterations) {
    image_ = cv::imread(path_ + input_.filename, CV_LOAD_IMAGE_GRAYSCALE);
  }

  // Runs the full algorithm multiple times to find the average and standard
  // deviation of run times. It is then run on each stage in isolation.
  void Run();

  // Output the results to stdout
  void OutputResults();

  // return the set of results.
  ResultSet Results() { return results_; }
};
}

