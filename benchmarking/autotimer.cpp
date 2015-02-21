// Name: Sean Kelley
// Date: Sep 22 2014
// Source File: AutoTimer.cpp
// Description: Implementation of AutoTimer. Use RAII principles with a basic
//             timer.
// Class: B424
#include <chrono>
#include <functional>
#include "autotimer.h"

using namespace std;

namespace Benchmarking {

AutoTimer::AutoTimer() { start_ = chrono::steady_clock::now(); }

AutoTimer::AutoTimer(std::function<void(double)> call_back) {
  this->call_back_ = call_back;
  start_ = chrono::steady_clock::now();
}

AutoTimer::~AutoTimer() {
  if (call_back_) call_back_(Duration());
}

double AutoTimer::Duration() {
  end_ = chrono::steady_clock::now();
  return (chrono::duration_cast<chrono::duration<int, micro> >(end_ - start_))
             .count() / 1000.0;
}
}
