#include <iostream>

#include <lgmath/CommonTools.hpp>
#include <lgmath/so3/Operations.hpp>
#include <lgmath/so3/Rotation.hpp>

#include "PrecisionTimer.hpp"

int main(int argc, char **argv) {

  // Init variables
  unsigned int N = 1000000;
  unsigned int L = 1000;
  unsigned int M = 10000;
  lgmath::common::Timer timer;
  ChronoTimer::HighResolutionTimer htimer;
  double time1, time2;
  double recorded;

  // Allocate test memory
  lgmath::so3::Rotation rotation;
  Eigen::Vector3d v3 = Eigen::Vector3d::Random();

  /////////////////////////////////////////////////////////////////////////////////////////////
  /// Rotation Testing
  /////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "Starting Rotation Tests" << std::endl;
  std::cout << "-----------------------------" << std::endl;
  std::cout << "Comparison timings are to get a ballpark estimate." << std::endl;
  std::cout << "Check that it is not an order of magnitude off; you may not be in release mode." << std::endl;
  std::cout << " " << std::endl;

  // test
  std::cout << "Test rotation vec2rot, over " << N << " iterations." << std::endl;
  htimer.reset();
  for (unsigned int i = 0; i < N; i++) {
    rotation = lgmath::so3::Rotation(v3);
  }
  time1 = htimer.nanoseconds();
  recorded = 0.091;
  std::cout << "your speed: " << time1/double(N) << "nsec per call." << std::endl;
  std::cout << "recorded:   " << 1000.0*recorded << "nsec per call, 2.4 GHz processor, July 2015" << std::endl;
  std::cout << " " << std::endl;


  // test
  std::cout << "Test lval vs rval assignment, over " << N << " iterations." << std::endl;
  lgmath::so3::Rotation tmp(rotation);
  lgmath::so3::Rotation tmp2(rotation);
  double build_time;
  build_time = time1 = time2 = 0;

  for (unsigned int j = 0; j < L; ++j) {
    htimer.reset();
    for (unsigned int i = 0; i < M; i++) {
      tmp = lgmath::so3::Rotation(rotation);
    }
    build_time += htimer.nanoseconds();

    htimer.reset();
    for (unsigned int i = 0; i < M; i++) {
      tmp = lgmath::so3::Rotation(rotation);
      tmp2 = tmp;
    }
    time1 += htimer.nanoseconds();

    htimer.reset();
    for (unsigned int i = 0; i < M; i++) {
      tmp = lgmath::so3::Rotation(rotation);
      tmp2 = std::move(tmp);
    }
    time2 += htimer.nanoseconds();
  }

  std::cout << "Lval assignment time: " << (time1-build_time)/double(M*L) << "nsec per call." << std::endl;
  std::cout << "Rval assignment time: " << (time2-build_time)/double(M*L) << "nsec per call." << std::endl;
  std::cout << "Difference: " << (time1-time2)/(time1-build_time)*100 << "%" << std::endl;
  std::cout << " " << std::endl;

  // test
  std::cout << "Test rotation rot2vec, over " << N << " iterations." << std::endl;
  htimer.reset();
  for (unsigned int i = 0; i < N; i++) {
    v3 = rotation.vec();
  }
  time1 = htimer.nanoseconds();
  recorded = 0.051;
  std::cout << "your speed: " << time1/double(N) << "nsec per call." << std::endl;
  std::cout << "recorded:   " << 1000.0*recorded << "nsec per call, 2.4 GHz processor, July 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  std::cout << "Test product over " << N << " iterations." << std::endl;
  htimer.reset();
  for (unsigned int i = 0; i < N; i++) {
    rotation = rotation*rotation;
  }
  time1 = htimer.nanoseconds();
  recorded = 0.027;
  std::cout << "your speed: " << time1/double(N) << "nsec per call." << std::endl;
  std::cout << "recorded:   " << 1000.0*recorded << "nsec per call, 2.4 GHz processor, July 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  std::cout << "Test product with inverse over " << N << " iterations." << std::endl;
  htimer.reset();
  for (unsigned int i = 0; i < N; i++) {
    rotation = rotation/rotation;
  }
  time1 = htimer.nanoseconds();
  recorded = 0.027;
  std::cout << "your speed: " << time1/double(N) << "nsec per call." << std::endl;
  std::cout << "recorded:   " << 1000.0*recorded << "nsec per call, 2.4 GHz processor, July 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  std::cout << "Test product with landmark over " << N << " iterations." << std::endl;
  htimer.reset();
  for (unsigned int i = 0; i < N; i++) {
    v3 = rotation*v3;
  }
  time1 = htimer.nanoseconds();
  recorded = 0.008;
  std::cout << "your speed: " << time1/double(N) << "nsec per call." << std::endl;
  std::cout << "recorded:   " << 1000.0*recorded << "nsec per call, 2.4 GHz processor, July 2015" << std::endl;
  std::cout << " " << std::endl;

  return 0;
}

