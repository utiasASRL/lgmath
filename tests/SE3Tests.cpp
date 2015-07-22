//////////////////////////////////////////////////////////////////////////////////////////////
/// \file NaiveSE3Tests.cpp
/// \brief Unit tests for the naive implementation of the SE3 Lie Group math.
/// \details Unit tests for the various Lie Group functions will test both special cases,
///          and randomly generated cases.
///
/// \author Sean Anderson
//////////////////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <iostream>
#include <iomanip>
#include <ios>

#include <Eigen/Dense>
#include <lgmath/CommonMath.hpp>
#include <lgmath/SO3.hpp>
#include <lgmath/SE3.hpp>

#include "catch.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////
///
/// UNIT TESTS OF SE(3) MATH
///
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SE(3) hat function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Test 4x4 hat function.", "[lgmath]" ) {

  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double,6,1> > trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double,6,1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double,4,4> > trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,4,4> mat;
    mat <<               0.0,  -trueVecs.at(i)[5],   trueVecs.at(i)[4],  trueVecs.at(i)[0],
           trueVecs.at(i)[5],                 0.0,  -trueVecs.at(i)[3],  trueVecs.at(i)[1],
          -trueVecs.at(i)[4],   trueVecs.at(i)[3],                 0.0,  trueVecs.at(i)[2],
                         0.0,                 0.0,                 0.0,                0.0;
    trueMats.push_back(mat);
  }

  // Test the function
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,4,4> testMat = lgmath::se3::hat(trueVecs.at(i));
    INFO("true: " << trueMats.at(i));
    INFO("func: " << testMat);
    CHECK(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
  }

} // TEST_CASE

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SE(3) curlyhat function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Test curlyhat function.", "[lgmath]" ) {

  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double,6,1> > trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double,6,1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double,6,6> > trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,6,6> mat;
    mat <<               0.0,  -trueVecs.at(i)[5],   trueVecs.at(i)[4],                0.0,  -trueVecs.at(i)[2],   trueVecs.at(i)[1],
           trueVecs.at(i)[5],                 0.0,  -trueVecs.at(i)[3],  trueVecs.at(i)[2],                 0.0,  -trueVecs.at(i)[0],
          -trueVecs.at(i)[4],   trueVecs.at(i)[3],                 0.0, -trueVecs.at(i)[1],   trueVecs.at(i)[0],                 0.0,
                         0.0,                 0.0,                 0.0,                0.0,  -trueVecs.at(i)[5],   trueVecs.at(i)[4],
                         0.0,                 0.0,                 0.0,  trueVecs.at(i)[5],                 0.0,  -trueVecs.at(i)[3],
                         0.0,                 0.0,                 0.0, -trueVecs.at(i)[4],   trueVecs.at(i)[3],                 0.0;
    trueMats.push_back(mat);
  }

  // Test the function
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,6,6> testMat = lgmath::se3::curlyhat(trueVecs.at(i));
    INFO("true: " << trueMats.at(i));
    INFO("func: " << testMat);
    CHECK(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
  }

} // TEST_CASE

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of homogeneous point to 4x6 matrix function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Test point to 4x6 matrix function.", "[lgmath]" ) {

  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double,4,1> > trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double,4,1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double,4,6> > trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,4,6> mat;
    mat << trueVecs.at(i)[3],                 0.0,                 0.0,                0.0,   trueVecs.at(i)[2],  -trueVecs.at(i)[1],
                         0.0,   trueVecs.at(i)[3],                 0.0, -trueVecs.at(i)[2],                 0.0,   trueVecs.at(i)[0],
                         0.0,                 0.0,   trueVecs.at(i)[3],  trueVecs.at(i)[1],  -trueVecs.at(i)[0],                 0.0,
                         0.0,                 0.0,                 0.0,                0.0,                 0.0,                 0.0;
    trueMats.push_back(mat);
  }

  // Test the 3x1 function with scaling param
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,4,6> testMat = lgmath::se3::point2fs(trueVecs.at(i).head<3>(), trueVecs.at(i)[3]);
    INFO("true: " << trueMats.at(i));
    INFO("func: " << testMat);
    CHECK(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
  }

} // TEST_CASE

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of homogeneous point to 6x4 matrix function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Test point to 6x4 matrix function.", "[lgmath]" ) {

  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double,4,1> > trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double,4,1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double,6,4> > trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,6,4> mat;
    mat <<               0.0,                 0.0,                 0.0,  trueVecs.at(i)[0],
                         0.0,                 0.0,                 0.0,  trueVecs.at(i)[1],
                         0.0,                 0.0,                 0.0,  trueVecs.at(i)[2],
                         0.0,   trueVecs.at(i)[2],  -trueVecs.at(i)[1],                0.0,
          -trueVecs.at(i)[2],                 0.0,   trueVecs.at(i)[0],                0.0,
           trueVecs.at(i)[1],  -trueVecs.at(i)[0],                 0.0,                0.0;
    trueMats.push_back(mat);
  }

  // Test the 3x1 function with scaling param
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,6,4> testMat = lgmath::se3::point2sf(trueVecs.at(i).head<3>(), trueVecs.at(i)[3]);
    INFO("true: " << trueMats.at(i));
    INFO("func: " << testMat);
    CHECK(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
  }

} // TEST_CASE

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential functions: vec2tran and tran2vec
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Compare analytical and numeric vec2tran.", "[lgmath]" ) {

  // Add vectors to be tested
  std::vector<Eigen::Matrix<double,6,1> > trueVecs;
  Eigen::Matrix<double,6,1> temp;
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.5*lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.5*lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.5*lgmath::constants::PI; trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    trueVecs.push_back(Eigen::Matrix<double,6,1>::Random());
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Calc matrices
  std::vector<Eigen::Matrix<double,4,4> > analyticTrans;
  for (unsigned i = 0; i < numTests; i++) {
    analyticTrans.push_back(lgmath::se3::vec2tran(trueVecs.at(i)));
  }

  // Compare analytical and numeric result
  SECTION("analytic vs numeric vec2tran" ) {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double,4,4> numericTran = lgmath::se3::vec2tran(trueVecs.at(i), 20);
      INFO("ana: " << analyticTrans.at(i));
      INFO("num: " << numericTran);
      CHECK(lgmath::common::nearEqual(analyticTrans.at(i), numericTran, 1e-6));
    }
  }

  // Test rot2vec
  SECTION("tran2vec" ) {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double,6,1> testVec = lgmath::se3::tran2vec(analyticTrans.at(i));
      INFO("true: " << trueVecs.at(i));
      INFO("func: " << testVec);
      CHECK(lgmath::common::nearEqualLieAlg(trueVecs.at(i), testVec, 1e-6));
    }
  }

} // TEST_CASE

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential jacobians: vec2jac and vec2jacinv
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Compare analytical jacobians, inverses and numeric counterparts in SE(3).", "[lgmath]" ) {

  // Add vectors to be tested
  std::vector<Eigen::Matrix<double,6,1> > trueVecs;
  Eigen::Matrix<double,6,1> temp;
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.5*lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.5*lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.5*lgmath::constants::PI; trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    trueVecs.push_back(Eigen::Matrix<double,6,1>::Random());
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Calc analytical matrices
  std::vector<Eigen::Matrix<double,6,6> > analyticJacs;
  std::vector<Eigen::Matrix<double,6,6> > analyticJacInvs;
  for (unsigned i = 0; i < numTests; i++) {
    analyticJacs.push_back(lgmath::se3::vec2jac(trueVecs.at(i)));
    analyticJacInvs.push_back(lgmath::se3::vec2jacinv(trueVecs.at(i)));
  }

  // Compare inversed analytical and analytical inverse
  for (unsigned i = 0; i < numTests; i++) {
    INFO("ana: " << analyticJacs.at(i));
    INFO("num: " << analyticJacInvs.at(i));
    CHECK(lgmath::common::nearEqual(analyticJacs.at(i).inverse(), analyticJacInvs.at(i), 1e-6));
  }

  // Compare analytical and 'numerical' jacobian
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,6,6> numericJac = lgmath::se3::vec2jac(trueVecs.at(i), 20);
    INFO("ana: " << analyticJacs.at(i));
    INFO("num: " << numericJac);
    CHECK(lgmath::common::nearEqual(analyticJacs.at(i), numericJac, 1e-6));
  }

  // Compare analytical and 'numerical' jacobian inverses
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,6,6> numericJac = lgmath::se3::vec2jacinv(trueVecs.at(i), 20);
    INFO("ana: " << analyticJacInvs.at(i));
    INFO("num: " << numericJac);
    CHECK(lgmath::common::nearEqual(analyticJacInvs.at(i), numericJac, 1e-6));
  }
} // TEST_CASE

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of adjoint tranformation identity, Ad(T(v)) = I + curlyhat(v)*J(v)
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Test the identity Ad(T(v)) = I + curlyhat(v)*J(v).", "[lgmath]" ) {

  // Add vectors to be tested
  std::vector<Eigen::Matrix<double,6,1> > trueVecs;
  Eigen::Matrix<double,6,1> temp;
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, lgmath::constants::PI; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, -lgmath::constants::PI; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.5*lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.5*lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.5*lgmath::constants::PI; trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    trueVecs.push_back(Eigen::Matrix<double,6,1>::Random());
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Test Identity
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,6,6> lhs = lgmath::se3::tranAd(lgmath::se3::vec2tran(trueVecs.at(i)));
    Eigen::Matrix<double,6,6> rhs = Eigen::Matrix<double,6,6>::Identity() +
      lgmath::se3::curlyhat(trueVecs.at(i))*lgmath::se3::vec2jac(trueVecs.at(i));
    INFO("lhs: " << lhs);
    INFO("rhs: " << rhs);
    CHECK(lgmath::common::nearEqual(lhs, rhs, 1e-6));
  }
} // TEST_CASE

