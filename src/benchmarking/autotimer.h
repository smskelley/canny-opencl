#pragma once
// Name: Sean Kelley
// Date: Sep 22 2014
// Source File: AutoTimer.h
// Description: Specification of AutoTimer. Use RAII principles with a basic
//             timer. All times are in milliseconds, timer begins on
//             instantiation. A call back can optionally be called which is
//             called at destruction.
// Class: B424
#include <functional>
#include <chrono>

namespace Benchmarking {

class AutoTimer {
  std::chrono::steady_clock::time_point start_;
  std::chrono::steady_clock::time_point end_;
  std::function<void(double)> call_back_;

 public:
  // Create a simple timer that does not execute callback on destruction.
  // Starts timer immediately.
  AutoTimer();

  // Create a timer which will call the callback on destruction.
  // Starts timer immediately.
  AutoTimer(std::function<void(double)> call_back);
  ~AutoTimer();

  // Returns duration in milliseconds.
  double Duration();
};
}
