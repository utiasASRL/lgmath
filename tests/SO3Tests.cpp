//////////////////////////////////////////////////////////////////////////////////////////////
/// \file NaiveSO3Tests.cpp
/// \brief Unit tests for the naive implementation of the SO3 Lie Group math.
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
#include <lgmath/so3/Operations.hpp>

#include "catch.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////
///
/// UNIT TESTS OF SO(3) MATH
///
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SO(3) hat function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Test 3x3 hat function.", "[lgmath]" ) {

  // Init
  std::vector<Eigen::Matrix<double,3,1> > trueVecs;
  std::vector<Eigen::Matrix<double,3,3> > trueMats;

  // Add vectors to be tested - we choose a few
  trueVecs.push_back(Eigen::Matrix<double,3,1>( 0.0,  0.0,  0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>( 1.0,  2.0,  3.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(-1.0, -2.0, -3.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>::Random());

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Setup truth matrices
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,3,3> mat;
    mat <<               0.0,  -trueVecs.at(i)[2],   trueVecs.at(i)[1],
           trueVecs.at(i)[2],                 0.0,  -trueVecs.at(i)[0],
          -trueVecs.at(i)[1],   trueVecs.at(i)[0],                 0.0;
    trueMats.push_back(mat);
  }

  // Test the function
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,3,3> testMat = lgmath::so3::hat(trueVecs.at(i));
    INFO("true: " << trueMats.at(i));
    INFO("func: " << testMat);
    CHECK(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
  }

  // Test identity,  hat(v)^T = -hat(v)
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,3,3> testMat = lgmath::so3::hat(trueVecs.at(i));
    CHECK(lgmath::common::nearEqual(testMat.transpose(), -testMat, 1e-6));
  }

} // TEST_CASE

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test special cases of exponential functions: vec2rot and rot2vec
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Special case tests of vec2rot.", "[lgmath]" ) {

  // Init
  std::vector<Eigen::Matrix<double,3,1> > trueVecs;
  std::vector<Eigen::Matrix<double,3,3> > trueMats;
  Eigen::Matrix<double,3,3> temp;

  // Identity
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.0, 0.0));
  trueMats.push_back(Eigen::Matrix<double,3,3>::Identity());

  // x-rot by PI
  trueVecs.push_back(Eigen::Matrix<double,3,1>(lgmath::constants::PI, 0.0, 0.0));
  temp <<  1.0,  0.0,  0.0,
           0.0, -1.0,  0.0,
           0.0,  0.0, -1.0;
  trueMats.push_back(temp);

  // y-rot by PI
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, lgmath::constants::PI, 0.0));
  temp << -1.0,  0.0,  0.0,
           0.0,  1.0,  0.0,
           0.0,  0.0, -1.0;
  trueMats.push_back(temp);

  // z-rot by PI
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.0, lgmath::constants::PI));
  temp << -1.0,  0.0,  0.0,
           0.0, -1.0,  0.0,
           0.0,  0.0,  1.0;
  trueMats.push_back(temp);

  // x-rot by -PI
  trueVecs.push_back(Eigen::Matrix<double,3,1>(-lgmath::constants::PI, 0.0, 0.0));
  temp <<  1.0,  0.0,  0.0,
           0.0, -1.0,  0.0,
           0.0,  0.0, -1.0;
  trueMats.push_back(temp);

  // y-rot by -PI
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, -lgmath::constants::PI, 0.0));
  temp << -1.0,  0.0,  0.0,
           0.0,  1.0,  0.0,
           0.0,  0.0, -1.0;
  trueMats.push_back(temp);

  // z-rot by -PI
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.0, -lgmath::constants::PI));
  temp << -1.0,  0.0,  0.0,
           0.0, -1.0,  0.0,
           0.0,  0.0,  1.0;
  trueMats.push_back(temp);

  // x-rot by PI/2
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.5*lgmath::constants::PI, 0.0, 0.0));
  temp <<  1.0,  0.0,  0.0,
           0.0,  0.0, -1.0,
           0.0,  1.0,  0.0;
  trueMats.push_back(temp);

  // y-rot by PI/2
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.5*lgmath::constants::PI, 0.0));
  temp <<  0.0,  0.0,  1.0,
           0.0,  1.0,  0.0,
          -1.0,  0.0,  0.0;
  trueMats.push_back(temp);

  // z-rot by PI/2
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.0, 0.5*lgmath::constants::PI));
  temp <<  0.0, -1.0,  0.0,
           1.0,  0.0,  0.0,
           0.0,  0.0,  1.0;
  trueMats.push_back(temp);

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Test vec2rot
  SECTION("vec2rot" ) {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double,3,3> testMat = lgmath::so3::vec2rot(trueVecs.at(i));
      INFO("true: " << trueMats.at(i));
      INFO("func: " << testMat);
      CHECK(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
    }
  }

  // Test rot2vec
  SECTION("rot2vec" ) {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double,3,1> testVec = lgmath::so3::rot2vec(trueMats.at(i));
      INFO("true: " << trueVecs.at(i));
      INFO("func: " << testVec);
      CHECK(lgmath::common::nearEqualAxisAngle(trueVecs.at(i), testVec, 1e-6));
    }
  }

} // TEST_CASE

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential functions: vec2rot and rot2vec
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Compare analytical and numeric vec2rot.", "[lgmath]" ) {

  // Add vectors to be tested
  std::vector<Eigen::Matrix<double,3,1> > trueVecs;
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.0, 0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(lgmath::constants::PI, 0.0, 0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, lgmath::constants::PI, 0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.0, lgmath::constants::PI));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(-lgmath::constants::PI, 0.0, 0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, -lgmath::constants::PI, 0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.0, -lgmath::constants::PI));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.5*lgmath::constants::PI, 0.0, 0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.5*lgmath::constants::PI, 0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.0, 0.5*lgmath::constants::PI));
  const unsigned numRand = 50;
  for (unsigned i = 0; i < numRand; i++) {
    trueVecs.push_back(Eigen::Matrix<double,3,1>::Random());
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Calc matrices
  std::vector<Eigen::Matrix<double,3,3> > analyticRots;
  for (unsigned i = 0; i < numTests; i++) {
    analyticRots.push_back(lgmath::so3::vec2rot(trueVecs.at(i)));
  }

  // Compare analytical and numeric result
  SECTION("analytic vs numeric vec2rot" ) {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double,3,3> numericRot = lgmath::so3::vec2rot(trueVecs.at(i), 20);
      INFO("ana: " << analyticRots.at(i));
      INFO("num: " << numericRot);
      CHECK(lgmath::common::nearEqual(analyticRots.at(i), numericRot, 1e-6));
    }
  }

  // Test identity, rot^T = rot^-1
  SECTION("analytic vs numeric vec2rot" ) {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double,3,3> lhs = analyticRots.at(i).transpose();
      Eigen::Matrix<double,3,3> rhs = analyticRots.at(i).inverse();
      INFO("lhs: " << lhs);
      INFO("rhs: " << rhs);
      CHECK(lgmath::common::nearEqual(lhs, rhs, 1e-6));
    }
  }

  // Test rot2vec
  SECTION("rot2vec" ) {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double,3,1> testVec = lgmath::so3::rot2vec(analyticRots.at(i));
      INFO("true: " << trueVecs.at(i));
      INFO("func: " << testVec);
      CHECK(lgmath::common::nearEqualAxisAngle(trueVecs.at(i), testVec, 1e-6));
    }
  }

} // TEST_CASE

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential jacobians: vec2jac and vec2jacinv
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Compare analytical jacobians, inverses and numeric counterparts.", "[lgmath]" ) {

  // Add vectors to be tested
  std::vector<Eigen::Matrix<double,3,1> > trueVecs;
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.0, 0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(lgmath::constants::PI, 0.0, 0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, lgmath::constants::PI, 0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.0, lgmath::constants::PI));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(-lgmath::constants::PI, 0.0, 0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, -lgmath::constants::PI, 0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.0, -lgmath::constants::PI));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.5*lgmath::constants::PI, 0.0, 0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.5*lgmath::constants::PI, 0.0));
  trueVecs.push_back(Eigen::Matrix<double,3,1>(0.0, 0.0, 0.5*lgmath::constants::PI));
  const unsigned numRand = 50;
  for (unsigned i = 0; i < numRand; i++) {
    trueVecs.push_back(Eigen::Matrix<double,3,1>::Random());
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Calc analytical matrices
  std::vector<Eigen::Matrix<double,3,3> > analyticJacs;
  std::vector<Eigen::Matrix<double,3,3> > analyticJacInvs;
  for (unsigned i = 0; i < numTests; i++) {
    analyticJacs.push_back(lgmath::so3::vec2jac(trueVecs.at(i)));
    analyticJacInvs.push_back(lgmath::so3::vec2jacinv(trueVecs.at(i)));
  }

  // Compare inversed analytical and analytical inverse
  for (unsigned i = 0; i < numTests; i++) {
    INFO("ana: " << analyticJacs.at(i));
    INFO("num: " << analyticJacInvs.at(i));
    CHECK(lgmath::common::nearEqual(analyticJacs.at(i).inverse(), analyticJacInvs.at(i), 1e-6));
  }

  // Compare analytical and 'numerical' jacobian
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,3,3> numericJac = lgmath::so3::vec2jac(trueVecs.at(i), 20);
    INFO("ana: " << analyticJacs.at(i));
    INFO("num: " << numericJac);
    CHECK(lgmath::common::nearEqual(analyticJacs.at(i), numericJac, 1e-6));
  }

  // Compare analytical and 'numerical' jacobian inverses
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,3,3> numericJac = lgmath::so3::vec2jacinv(trueVecs.at(i), 20);
    INFO("ana: " << analyticJacInvs.at(i));
    INFO("num: " << numericJac);
    CHECK(lgmath::common::nearEqual(analyticJacInvs.at(i), numericJac, 1e-6));
  }

  // Test identity, rot(v) = eye(3) + hat(v)*jac(v), through the 'alternate' vec2rot method
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double,3,3> lhs = lgmath::so3::vec2rot(trueVecs.at(i));
    Eigen::Matrix<double,3,3> rhs, jac;
    // the following vec2rot call uses the identity: rot(v) = eye(3) + hat(v)*jac(v)
    lgmath::so3::vec2rot(trueVecs.at(i), &rhs, &jac);
    INFO("lhs: " << lhs);
    INFO("rhs: " << rhs);
    CHECK(lgmath::common::nearEqual(lhs, rhs, 1e-6));
  }

} // TEST_CASE



