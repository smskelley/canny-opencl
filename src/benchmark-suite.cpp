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

void CvBenchmark(string filename);

int main(int argc, char *argv[])
{
    vector<Benchmark> benchmarks;
    vector<InputImage> input_images {
        InputImage("world.jpg", 24000, 12000),
        InputImage("Great_Tit.jpg", 2948, 2057),
        InputImage("lena.jpg", 100, 100),
        InputImage("hs-2004-07-a-full_jpg.jpg", 6200, 6200)
    };


    // compile a list of all benchmarks to run.
    for (auto image : input_images)
    {
        benchmarks.push_back(
                Benchmark("Serial",
                shared_ptr<ImageProcessor>(new SerialImageProcessor()),
                IMG_PATH, image, 3));

        benchmarks.push_back(
                Benchmark("OpenCV",
                shared_ptr<ImageProcessor>(new CvImageProcessor()),
                IMG_PATH, image, 3));

        // OpenCL GPU & CPU benchmarks
        benchmarks.push_back(
                Benchmark("OpenCL GPU",
                shared_ptr<ImageProcessor>(new OpenclImageProcessor(true)),
                IMG_PATH, image, 3));

// OpenCL CPU disabled in gpu optimized version since CPU only supports
// linear workgroup sizing.
//
//        benchmarks.push_back(
//            Benchmark("OpenCL CPU",
//            shared_ptr<ImageProcessor>(new OpenclImageProcessor(false)),
//            IMG_PATH, image, 3));

    }
    
    // Run all benchmarks
    for (auto benchmark : benchmarks)
    {
        benchmark.Run();
        benchmark.OutputResults();
        cout << endl;
    }

    return 0;
}
