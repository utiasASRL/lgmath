#include <iostream>

#include <glog/logging.h>

#include <lgmath/CommonTools.hpp>
#include <lgmath/SE3.hpp>
#include <lgmath/Transformation.hpp>

int main(int argc, char **argv) {

  // Init Logging
  google::InitGoogleLogging(argv[0]);

  // Init variables
  size_t N = 0;
  const size_t million = 1000000;
  lgmath::common::Timer timer;
  double time1;

  // Allocate test memory
  lgmath::se3::Transformation transform;
  Eigen::Matrix<double,4,1> v4 = Eigen::Matrix<double,4,1>::Random();
  Eigen::Matrix<double,6,1> v6 = Eigen::Matrix<double,6,1>::Random();

  /////////////////////////////////////////////////////////////////////////////////////////////
  /// Transformation Testing
  /////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "Starting Transformation Tests" << std::endl;
  std::cout << "Comparison timings are for a ballpark... check that it is not an order of magnitude off..." << std::endl;
  std::cout << " " << std::endl;

  // test
  N = 10*million;
  std::cout << "Test transform vec2tran, over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned i = 0; i < N; i++) {
    transform = lgmath::se3::Transformation(v6);
  }
  time1 = timer.milliseconds();
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   0.122usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  N = 10*million;
  std::cout << "Test transform tran2vec, over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned i = 0; i < N; i++) {
    v6 = transform.vec();
  }
  time1 = timer.milliseconds();
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   0.132usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  N = 50*million;
  std::cout << "Test product over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned i = 0; i < N; i++) {
    transform = transform*transform;
  }
  time1 = timer.milliseconds();
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   0.025usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  N = 50*million;
  std::cout << "Test product with inverse over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned i = 0; i < N; i++) {
    transform = transform/transform;
  }
  time1 = timer.milliseconds();
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   0.035usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  N = 50*million;
  std::cout << "Test product with landmark over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned i = 0; i < N; i++) {
    v4 = transform*v4;
  }
  time1 = timer.milliseconds();
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   0.015usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  return 0;
}

