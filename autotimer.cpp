//Name: Sean Kelley
//Date: Sep 22 2014
//Source File: AutoTimer.cpp
//Description: Implementation of AutoTimer. Use RAII principles with a basic
//             timer.
//Class: B424
#include <chrono>
#include <functional>
#include "AutoTimer.h"

using namespace std;

AutoTimer::AutoTimer()
{
    start = chrono::steady_clock::now();
}

AutoTimer::AutoTimer(std::function<void(double)> call_back)
{
    this->call_back = call_back;
    start = chrono::steady_clock::now();
}

AutoTimer::~AutoTimer()
{
    if (call_back)
        call_back(Duration());
}

double AutoTimer::Duration()
{
    end = chrono::steady_clock::now();
    return (chrono::duration_cast< chrono::duration<int,micro> >(end-start)).count() / 1000.0;
}
