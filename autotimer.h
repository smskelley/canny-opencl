//Name: Sean Kelley
//Date: Sep 22 2014
//Source File: AutoTimer.h
//Description: Specification of AutoTimer. Use RAII principles with a basic
//             timer. All times are in milliseconds, timer begins on
//             instantiation. A call back can optionally be called which is
//             called at destruction.
//Class: B424
#include <functional>
#include <chrono>

class AutoTimer
{
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    std::function<void(double)> call_back;

  public:
    AutoTimer();
    AutoTimer(std::function<void(double)> call_back);
    ~AutoTimer();
    double Duration();
};
