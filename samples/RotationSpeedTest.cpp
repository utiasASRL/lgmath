#include <iostream>

#include <lgmath/CommonTools.hpp>
#include <lgmath/so3/Operations.hpp>
#include <lgmath/so3/Rotation.hpp>

int main(int argc, char **argv) {

  // Init variables
  unsigned int N = 1000000;
  lgmath::common::Timer timer;
  double time1;
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
  timer.reset();
  for (unsigned int i = 0; i < N; i++) {
    rotation = lgmath::so3::Rotation(v3);
  }
  time1 = timer.milliseconds();
  recorded = 0.091;
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   " <<        recorded        << "usec per call, 2.4 GHz processor, July 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  std::cout << "Test rotation rot2vec, over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned int i = 0; i < N; i++) {
    v3 = rotation.vec();
  }
  time1 = timer.milliseconds();
  recorded = 0.051;
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   " <<        recorded        << "usec per call, 2.4 GHz processor, July 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  std::cout << "Test product over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned int i = 0; i < N; i++) {
    rotation = rotation*rotation;
  }
  time1 = timer.milliseconds();
  recorded = 0.021;
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   " <<        recorded        << "usec per call, 2.4 GHz processor, July 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  std::cout << "Test product with inverse over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned int i = 0; i < N; i++) {
    rotation = rotation/rotation;
  }
  time1 = timer.milliseconds();
  recorded = 0.022;
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   " <<        recorded        << "usec per call, 2.4 GHz processor, July 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  std::cout << "Test product with landmark over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned int i = 0; i < N; i++) {
    v3 = rotation*v3;
  }
  time1 = timer.milliseconds();
  recorded = 0.008;
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   " <<        recorded        << "usec per call, 2.4 GHz processor, July 2015" << std::endl;
  std::cout << " " << std::endl;

  return 0;
}

