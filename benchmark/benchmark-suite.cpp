#include <iostream>
#include <string>
#include <vector>
#include "benchmark.h"
#include "imageprocessor.h"
#include "openclimageprocessor.h"
#include "serialimageprocessor.h"
#include "cvimageprocessor.h"

using namespace std;

const string IMG_PATH = "images/";

vector<Benchmark> create_benchmarks(vector<InputImage> input_images);

int main(int argc, char *argv[]) {
  // Images that will be used in the benchmarks as well as their resolution
  vector<InputImage> input_images {
      InputImage("world.jpg", 24000, 12000),
      InputImage("Great_Tit.jpg", 2948, 2057), InputImage("lena.jpg", 100, 100),
      InputImage("hs-2004-07-a-full_jpg.jpg", 6200, 6200)};

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

vector<Benchmark> create_benchmarks(vector<InputImage> input_images) {
  vector<Benchmark> benchmarks;

  // compile a list of all benchmarks to run.
  for (auto image : input_images) {
    // OpenCL GPU
    benchmarks.push_back(
        Benchmark("OpenCL GPU",
                  shared_ptr<ImageProcessor>(new OpenclImageProcessor(true)),
                  IMG_PATH, image, 3));
    // OpenCL CPU
    benchmarks.push_back(
        Benchmark("OpenCL CPU",
                  shared_ptr<ImageProcessor>(new OpenclImageProcessor(false)),
                  IMG_PATH, image, 3));

    // Benchmarks to compare our results against.
    // Serial
    benchmarks.push_back(Benchmark(
        "Serial", shared_ptr<ImageProcessor>(new SerialImageProcessor()),
        IMG_PATH, image, 3));

    // OpenCV
    benchmarks.push_back(
        Benchmark("OpenCV", shared_ptr<ImageProcessor>(new CvImageProcessor()),
                  IMG_PATH, image, 3));
  }

  return benchmarks;
}
