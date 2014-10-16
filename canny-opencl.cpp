#include <iostream>
#include "autotimer.h"
#include "imageprocessor.h"

using namespace std;

int main()
{
    ImageProcessor processor;
    AutoTimer timer([](double time){ cout << "I have died after being alive for " << time << " milliseconds.\n";});

    return 0;
}
