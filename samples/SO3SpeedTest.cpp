#include <iostream>

#include <glog/logging.h>

#include <lgmath/CommonTools.hpp>
#include <lgmath/SO3.hpp>

int main(int argc, char **argv) {

  // Init Logging
  google::InitGoogleLogging(argv[0]);

  // Init variables
  size_t N = 0;
  const size_t million = 1000000;
  lgmath::common::Timer timer;
  double time1;

  // Allocate test memory
  Eigen::Matrix<double,3,3> m33;
  Eigen::Matrix<double,3,1> v3 = Eigen::Matrix<double,3,1>::Random();

  /////////////////////////////////////////////////////////////////////////////////////////////
  /// SO Testing
  /////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "Starting SO(3) Tests" << std::endl;
  std::cout << "Comparison timings are for a ballpark... check that it is not an order of magnitude off..." << std::endl;
  std::cout << " " << std::endl;

  // test
  N = 100*million;
  std::cout << "Test SO3 hat, over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned i = 0; i < N; i++) {
    m33 = lgmath::so3::hat(v3);
  }
  time1 = timer.milliseconds();
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   0.0059usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  N = 10*million;
  std::cout << "Test SO3 vec2rot, over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned i = 0; i < N; i++) {
    m33 = lgmath::so3::vec2rot(v3);
  }
  time1 = timer.milliseconds();
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   0.077usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  N = 10*million;
  std::cout << "Test SO3 rot2vec, over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned i = 0; i < N; i++) {
    v3 = lgmath::so3::rot2vec(m33);
  }
  time1 = timer.milliseconds();
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   0.053usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  N = 10*million;
  std::cout << "Test SO3 vec2jac, over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned i = 0; i < N; i++) {
    m33 = lgmath::so3::vec2jac(v3);
  }
  time1 = timer.milliseconds();
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   0.0907usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  // test
  N = 10*million;
  std::cout << "Test SO3 vec2jacinv, over " << N << " iterations." << std::endl;
  timer.reset();
  for (unsigned i = 0; i < N; i++) {
    m33 = lgmath::so3::vec2jacinv(v3);
  }
  time1 = timer.milliseconds();
  std::cout << "your speed: " << 1000.0*time1/double(N) << "usec per call." << std::endl;
  std::cout << "recorded:   0.0623usec per call, 2.4 GHz processor, March 2015" << std::endl;
  std::cout << " " << std::endl;

  return 0;
}
