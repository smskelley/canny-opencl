// Name: Sean Kelley
// Date: Sep 22 2014
// Source File: AutoTimer.cpp
// Description: Implementation of AutoTimer. Calls callback function on
//              destruction in a RAII inspired fashion.
// Class: B424
#include <chrono>
#include <functional>
#include "autotimer.h"

using namespace std;

namespace Benchmarking {

// Starts timer on construction
AutoTimer::AutoTimer() { start_ = chrono::steady_clock::now(); }

// Stores callback and starts timer
AutoTimer::AutoTimer(std::function<void(double)> call_back) {
  this->call_back_ = call_back;
  start_ = chrono::steady_clock::now();
}

// Calls callback if one exists
AutoTimer::~AutoTimer() {
  if (call_back_) call_back_(Duration());
}

// Finds the difference between start and end with microsecond accuracy and
// converts the result ti milliseconds.
double AutoTimer::Duration() {
  end_ = chrono::steady_clock::now();
  return (chrono::duration_cast<chrono::duration<int, micro> >(end_ - start_))
             .count() / 1000.0;
}
}
