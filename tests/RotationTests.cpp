//////////////////////////////////////////////////////////////////////////////////////////////
/// \file RotationTests.cpp
/// \brief Unit tests for the implementation of the rotation matrix class.
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
#include <lgmath/so3/Rotation.hpp>

#include "catch.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////
///
/// UNIT TESTS OF ROTATION MATRIX
///
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of rotation constructors
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Rotation Constructors.", "[lgmath]" ) {

  // Generate random transform from most basic constructor
  Eigen::Matrix3d C_ba_rand = lgmath::so3::vec2rot(Eigen::Vector3d::Random());
  lgmath::so3::Rotation rand(C_ba_rand);

  // Rotation();
  SECTION("default" ) {
    lgmath::so3::Rotation cmatrix;
    Eigen::Matrix3d test = Eigen::Matrix3d::Identity();
    INFO("cmat: " << cmatrix.matrix());
    INFO("test: " << test);
    CHECK(lgmath::common::nearEqual(cmatrix.matrix(), test, 1e-6));
  }

  // Rotation(const Rotation& C);
  SECTION("copy constructor" ) {
    lgmath::so3::Rotation test(rand);
    INFO("cmat: " << rand.matrix());
    INFO("test: " << test.matrix());
    CHECK(lgmath::common::nearEqual(rand.matrix(), test.matrix(), 1e-6));
  }

  // Rotation(const Eigen::Matrix3d& C, bool reproj = true);
  SECTION("matrix constructor" ) {
    lgmath::so3::Rotation test = rand;
    INFO("cmat: " << rand.matrix());
    INFO("test: " << test.matrix());
    CHECK(lgmath::common::nearEqual(rand.matrix(), test.matrix(), 1e-6));

    // Test manual with no reprojection
    Eigen::Matrix3d notRotation = Eigen::Matrix3d::Random();
    lgmath::so3::Rotation test_bad(notRotation, false); // don't project
    INFO("cmat: " << test_bad.matrix());
    INFO("test: " << notRotation.matrix());
    CHECK(lgmath::common::nearEqual(test_bad.matrix(), notRotation.matrix(), 1e-6));
  }

  // Rotation& operator=(Rotation C);
  SECTION("assignment operator" ) {

    lgmath::so3::Rotation test(C_ba_rand);
    INFO("cmat: " << rand.matrix());
    INFO("test: " << test.matrix());
    CHECK(lgmath::common::nearEqual(rand.matrix(), test.matrix(), 1e-6));
  }

  // Rotation(const Eigen::Vector3d& vec, unsigned int numTerms = 0);
  SECTION("exponential map" ) {
    Eigen::Vector3d vec = Eigen::Vector3d::Random();
    Eigen::Matrix3d cmat = lgmath::so3::vec2rot(vec);
    lgmath::so3::Rotation testAnalytical(vec);
    lgmath::so3::Rotation testNumerical(vec,15);
    INFO("cmat: " << cmat);
    INFO("testAnalytical: " << testAnalytical.matrix());
    INFO("testNumerical: " << testNumerical.matrix());
    CHECK(lgmath::common::nearEqual(cmat, testAnalytical.matrix(), 1e-6));
    CHECK(lgmath::common::nearEqual(cmat, testNumerical.matrix(), 1e-6));
  }

  // Rotation(const Eigen::VectorXd& vec);
  SECTION("exponential map with VectorXd" ) {
    Eigen::VectorXd vec = Eigen::Vector3d::Random();
    Eigen::Matrix3d tmat = lgmath::so3::vec2rot(vec);
    lgmath::so3::Rotation test(vec);
    INFO("tmat: " << tmat);
    INFO("test: " << test.matrix());
    CHECK(lgmath::common::nearEqual(tmat, test.matrix(), 1e-6));

//    Eigen::VectorXd vec4 = Eigen::Matrix<double,4,1>::Random();
//    try {
//      lgmath::so3::Rotation testFailure(vec4);
//      CHECK(*this doesn't happen*);
//    } catch () {
//      CHECK(*this happens*);
//    }
  }

} // TEST_CASE

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test exponential map construction and logarithmic vec() method
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Test Rotation to/from SE(3) algebra.", "[lgmath]" ) {

  // Add vectors to be tested
  std::vector<Eigen::Vector3d> trueVecs;
  Eigen::Vector3d temp;
  temp << 0.0, 0.0, 0.0; trueVecs.push_back(temp);
  temp << lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, lgmath::constants::PI; trueVecs.push_back(temp);
  temp << -lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, -lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, -lgmath::constants::PI; trueVecs.push_back(temp);
  temp << 0.5*lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.5*lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.5*lgmath::constants::PI; trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    trueVecs.push_back(Eigen::Vector3d::Random());
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Calc rotation matrices
  std::vector<Eigen::Matrix3d > rotMatrices;
  for (unsigned i = 0; i < numTests; i++) {
    rotMatrices.push_back(lgmath::so3::vec2rot(trueVecs.at(i)));
  }

  // Calc rotations
  std::vector<lgmath::so3::Rotation > rotations;
  for (unsigned i = 0; i < numTests; i++) {
    rotations.push_back(lgmath::so3::Rotation(trueVecs.at(i)));
  }

  // Compare matrices
  SECTION("vec2rot") {
    for (unsigned i = 0; i < numTests; i++) {
      INFO("matr: " << rotMatrices.at(i));
      INFO("tran: " << rotations.at(i).matrix());
      CHECK(lgmath::common::nearEqual(rotMatrices.at(i), rotations.at(i).matrix(), 1e-6));
    }
  }

  // Test logarithmic map
  SECTION("rot2vec") {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Vector3d testVec = rotations.at(i).vec();
      INFO("true: " << trueVecs.at(i));
      INFO("func: " << testVec);
      CHECK(lgmath::common::nearEqualAxisAngle(trueVecs.at(i), testVec, 1e-6));
    }
  }

} // TEST_CASE

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test inverse and operatations
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Test Rotation inverse.", "[lgmath]" ) {

  // Add vectors to be tested
  std::vector<Eigen::Vector3d > trueVecs;
  Eigen::Vector3d temp;
  temp << 0.0, 0.0, 0.0; trueVecs.push_back(temp);
  temp << lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, lgmath::constants::PI; trueVecs.push_back(temp);
  temp << -lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, -lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, -lgmath::constants::PI; trueVecs.push_back(temp);
  temp << 0.5*lgmath::constants::PI, 0.0, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.5*lgmath::constants::PI, 0.0; trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.5*lgmath::constants::PI; trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    trueVecs.push_back(Eigen::Vector3d::Random());
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Add vectors to be tested - random
  std::vector<Eigen::Vector3d > landmarks;
  for (unsigned i = 0; i < numTests; i++) {
    landmarks.push_back(Eigen::Vector3d::Random());
  }

  // Calc rotation matrices
  std::vector<Eigen::Matrix3d > rotMatrices;
  for (unsigned i = 0; i < numTests; i++) {
    rotMatrices.push_back(lgmath::so3::vec2rot(trueVecs.at(i)));
  }

  // Calc rotations
  std::vector<lgmath::so3::Rotation > rotations;
  for (unsigned i = 0; i < numTests; i++) {
    rotations.push_back(lgmath::so3::Rotation(trueVecs.at(i)));
  }

  // Compare inverse to basic matrix inverse
  SECTION("compare inverse") {
    for (unsigned i = 0; i < numTests; i++) {
      INFO("matr: " << rotMatrices.at(i).inverse());
      INFO("tran: " << rotations.at(i).inverse().matrix());
      CHECK(lgmath::common::nearEqual(rotMatrices.at(i).inverse(), rotations.at(i).inverse().matrix(), 1e-6));
    }
  }

  // Test that product of inverse and self make identity
  SECTION("test product of inverse") {
    for (unsigned i = 0; i < numTests; i++) {
      INFO("C*Cinv: " << rotations.at(i).matrix()*rotations.at(i).inverse().matrix());
      CHECK(lgmath::common::nearEqual(rotations.at(i).matrix()*rotations.at(i).inverse().matrix(), Eigen::Matrix3d::Identity(), 1e-6));
    }
  }

  // Test self-product
  SECTION("test self product") {
    for (unsigned i = 0; i < numTests-1; i++) {
      lgmath::so3::Rotation test = rotations.at(i);
      test *= rotations.at(i+1);
      Eigen::Matrix3d matrix = rotMatrices.at(i)*rotMatrices.at(i+1);
      INFO("matr: " << matrix);
      INFO("tran: " << test.matrix());
      CHECK(lgmath::common::nearEqual(matrix, test.matrix(), 1e-6));
    }
  }

  // Test product
  SECTION("test product") {
    for (unsigned i = 0; i < numTests-1; i++) {
      lgmath::so3::Rotation test = rotations.at(i)*rotations.at(i+1);
      Eigen::Matrix3d matrix = rotMatrices.at(i)*rotMatrices.at(i+1);
      INFO("matr: " << matrix);
      INFO("tran: " << test.matrix());
      CHECK(lgmath::common::nearEqual(matrix, test.matrix(), 1e-6));
    }
  }

  // Test self product with inverse
  SECTION("test self product with inverse") {
    for (unsigned i = 0; i < numTests-1; i++) {
      lgmath::so3::Rotation test = rotations.at(i);
      test /= rotations.at(i+1);
      Eigen::Matrix3d matrix = rotMatrices.at(i) * rotMatrices.at(i+1).inverse();
      INFO("matr: " << matrix);
      INFO("tran: " << test.matrix());
      CHECK(lgmath::common::nearEqual(matrix, test.matrix(), 1e-6));
    }
  }

  // Test product with inverse
  SECTION("test product with inverse") {
    for (unsigned i = 0; i < numTests-1; i++) {
      lgmath::so3::Rotation test = rotations.at(i) / rotations.at(i+1);
      Eigen::Matrix3d matrix = rotMatrices.at(i) * rotMatrices.at(i+1).inverse();
      INFO("matr: " << matrix);
      INFO("tran: " << test.matrix());
      CHECK(lgmath::common::nearEqual(matrix, test.matrix(), 1e-6));
    }
  }

  // Test product with landmark
  SECTION("test product with landmark") {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Vector3d mat = rotMatrices.at(i)*landmarks.at(i);
      Eigen::Vector3d test = rotations.at(i)*landmarks.at(i);

      INFO("matr: " << mat);
      INFO("test: " << test);
      CHECK(lgmath::common::nearEqual(mat, test, 1e-6));
    }
  }

} // TEST_CASE

