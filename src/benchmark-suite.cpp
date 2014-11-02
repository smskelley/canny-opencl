#include <iostream>
#include <string>
#include "benchmark.h"

using namespace std;

const string IMG_PATH = "images/";

int main(int argc, char *argv[])
{
    // if first param is 'cpu', then image processor should use the cpu.
    bool useGPU = true;
    if (argc > 1 && strcmp(argv[1], "cpu") == 0)
        useGPU = false;

    vector<InputImage> input_images {
        InputImage("Great_Tit.jpg", 2948, 2057),
        InputImage("hs-2004-07-a-full_jpg.jpg", 6200, 6200)
    };


    for (auto image : input_images)
    {
        Benchmark benchmark(IMG_PATH, image, useGPU, 3);
        benchmark.Run();
        benchmark.OutputResults();
        cout << endl;
    }

    return 0;
}
