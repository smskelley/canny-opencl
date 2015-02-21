// Benchmarks all of the image processors by timing them on test images.
// For each image, each image processor will execute canny multiple times
// to find the mean and standard deviation. Next, each stage is executed
// in isolation to determine the runtime of each stage.
#include <iostream>
#include <vector>
#include <memory>
#include "benchmark.h"
#include "openclimageprocessor.h"
#include "serialimageprocessor.h"
#include "cvimageprocessor.h"

using namespace std;
using namespace Benchmarking;
using namespace ImageProcessors;
const string IMG_PATH = "images/";

vector<Benchmark> create_benchmarks(vector<InputImage> input_images);

int main(int argc, char *argv[]) {
  // Images that will be used in the benchmarks as well as their resolution
  vector<InputImage> input_images {
          InputImage("lena.jpg", 100, 100),
          InputImage("Great_Tit.jpg", 2948, 2057),
          InputImage("hs-2004-07-a-full_jpg.jpg", 6200, 6200),
          InputImage("world.jpg", 24000, 12000)};

  // Each benchmark targets a different canny implementation.
  vector<Benchmark> benchmarks = create_benchmarks(input_images);

  // Run all benchmarks
  for (auto benchmark : benchmarks) {
    benchmark.Run();
    benchmark.OutputResults();
    cout << endl;
  }

  return 0;
}

// Given a set of input images, this will generate benchmark objects
// for each image using each image processor.
vector<Benchmark> create_benchmarks(vector<InputImage> input_images) {
  vector<Benchmark> benchmarks;

  // compile a list of all benchmarks to run.
  for (auto image : input_images) {
    // OpenCL GPU
    benchmarks.push_back(Benchmark("OpenCL GPU",
                                   make_shared<OpenclImageProcessor>(true),
                                   IMG_PATH, image, 3));
    // OpenCL CPU
    benchmarks.push_back(Benchmark("OpenCL CPU",
                                   make_shared<OpenclImageProcessor>(false),
                                   IMG_PATH, image, 3));

    // Benchmarks to compare our results against.
    // Serial
    benchmarks.push_back(Benchmark("Serial",
                                   make_shared<SerialImageProcessor>(),
                                   IMG_PATH, image, 3));

    // OpenCV
    benchmarks.push_back(Benchmark("OpenCV",
                                   make_shared<CvImageProcessor>(),
                                   IMG_PATH, image, 3));
  }

  return benchmarks;
}
