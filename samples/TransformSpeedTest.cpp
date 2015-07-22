#include <iostream>

#include <glog/logging.h>

#include <lgmath/CommonTools.hpp>
#include <lgmath/se3/Operations.hpp>
#include <lgmath/se3/Transformation.hpp>


int main(int argc, char **argv) {

  // Init Logging
  google::InitGoogleLogging(argv[0]);

  // Init variables
  unsigned int N = 1000000;
  lgmath::common::Timer timer;
  double time1;
  double recorded;

  // Allocate test memory
  lgmath::se3::Transformation transform;
  Eigen::Matrix<double,4,1> v4 = Eigen::Matrix<double,4,1>::Random();
  Eigen::Matrix<double,6,1> v6 = Eigen::Matrix<double,6,1>::Random();

  /////////////////////////////////////////////////////////////////////////////////////////////
  /// Transformation Testing
  /////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "Starting Transformation Tests" << std::endl;
  std::cout << "-----------------------------" << std::endl;
  std::cout << "Comparison timings are to get a ballpark estimate." << std::endl;
  std::cout << "Check that it is not an order of magnitude off; you may not be in release mode." << std::endl;
  std::cout << " " << std::endl;

  // test
  std::cout << "Test transform vec2tran, over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned int i = 0; i < N; i++) {
    transform = lgmath::se3::Transformation(v6);
  }
  time1 = timer.milliseconds();
  recorded = 0.122;
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   " <<        recorded        << "usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  std::cout << "Test transform tran2vec, over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned int i = 0; i < N; i++) {
    v6 = transform.vec();
  }
  time1 = timer.milliseconds();
  recorded = 0.132;
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   " <<        recorded        << "usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  std::cout << "Test product over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned int i = 0; i < N; i++) {
    transform = transform*transform;
  }
  time1 = timer.milliseconds();
  recorded = 0.025;
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   " <<        recorded        << "usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  std::cout << "Test product with inverse over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned int i = 0; i < N; i++) {
    transform = transform/transform;
  }
  time1 = timer.milliseconds();
  recorded = 0.035;
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   " <<        recorded        << "usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  std::cout << "Test product with landmark over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned int i = 0; i < N; i++) {
    v4 = transform*v4;
  }
  time1 = timer.milliseconds();
  recorded = 0.015;
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   " <<        recorded        << "usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  return 0;
}

