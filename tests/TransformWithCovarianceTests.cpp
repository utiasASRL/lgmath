//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TransformTests.cpp
/// \brief Unit tests for the implementation of the transformation matrix class.
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
#include <lgmath/se3/Operations.hpp>
#include <lgmath/se3/TransformationWithCovariance.hpp>

#include "catch.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////
///
/// UNIT TESTS OF TRANSFORMATION MATRIX WITH COVARIANCE
///
/// NOTE: These tests are mainly comparitive against the base Transform, and assume that the
///       relevant methods in the base class have all passed testing.
///
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of transformation constructors
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("TransformationWithCovariance Constructors.", "[lgmath]" ) {

  // Generate random transform from most basic constructor
  Eigen::Matrix<double,3,3> C_ba = lgmath::so3::vec2rot(Eigen::Matrix<double,3,1>::Random());
  Eigen::Matrix<double,3,1> r_ba_ina = Eigen::Matrix<double,3,1>::Random();
  Eigen::Matrix<double,6,6> U = Eigen::Matrix<double,6,6>::Random();
  lgmath::se3::Transformation randBase(C_ba, r_ba_ina);
  lgmath::se3::TransformationWithCovariance rand(C_ba, r_ba_ina, U);

  // TransformationWithCovariance();
  SECTION("default" ) {
    lgmath::se3::TransformationWithCovariance tmatrix;
    Eigen::Matrix4d test = Eigen::Matrix4d::Identity();
    INFO("tmat: " << tmatrix.matrix());
    INFO("test: " << test);
    CHECK(lgmath::common::nearEqual(tmatrix.matrix(), test, 1e-6));

    // We should not be able to query the covariance
    INFO("Checking covarianceSet_ flag...");
    bool passed = false;
    Eigen::Matrix<double,6,6> testU;
    try {
      testU = tmatrix.cov();
    }
    catch (const std::logic_error& e) {
      passed = true;
    }
    CHECK(passed);
  }

  // TransformationWithCovariance(const TransformationWithCovariance& T);
  SECTION("copy constructor" ) {
    lgmath::se3::TransformationWithCovariance test(rand);
    INFO("tmat: " << rand.matrix());
    INFO("test: " << test.matrix());
    CHECK(lgmath::common::nearEqual(rand.matrix(), test.matrix(), 1e-6));

    INFO("tmat: " << rand.cov());
    INFO("test: " << test.cov());
    CHECK(lgmath::common::nearEqual(rand.cov(), test.cov(), 1e-6));
  }


  // TransformationWithCovariance(const Transformation& T);
  SECTION("copy constructor (from base)" ) {
    lgmath::se3::TransformationWithCovariance test(randBase);
    INFO("tmat: " << randBase.matrix());
    INFO("test: " << test.matrix());
    CHECK(lgmath::common::nearEqual(randBase.matrix(), test.matrix(), 1e-6));

    // We should NOT be able to query the covariance
    INFO("Checking covarianceSet_ flag...");
    bool passed = false;
    Eigen::Matrix<double,6,6> testU;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = true;
      testU = Eigen::Matrix<double,6,6>::Zero();
    }
    CHECK(passed);
    CHECK(lgmath::common::nearEqual(Eigen::Matrix<double,6,6>::Zero(), testU, 1e-6));
  }

  // TransformationWithCovariance(const Transformation& T, Eigen::Matrix6d& U);
  SECTION("copy constructor (from base, with covariance)" ) {
    lgmath::se3::TransformationWithCovariance test2(randBase, U);
    INFO("tmat: " << randBase.matrix());
    INFO("test: " << test2.matrix());
    CHECK(lgmath::common::nearEqual(randBase.matrix(), test2.matrix(), 1e-6));

    INFO("tmat: " << U);
    INFO("test: " << test2.cov());
    CHECK(lgmath::common::nearEqual(U, test2.cov(), 1e-6));
  }

  // Transformation(const Eigen::Matrix4d& T, bool reproj = true);
  SECTION("matrix constructor" ) {
    lgmath::se3::TransformationWithCovariance test(rand.matrix());
    INFO("tmat: " << rand.matrix());
    INFO("test: " << test.matrix());
    CHECK(lgmath::common::nearEqual(rand.matrix(), test.matrix(), 1e-6));

    // We should not be able to query the covariance
    INFO("Checking covarianceSet_ flag...");
    bool passed = false;
    Eigen::Matrix<double,6,6> testU;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = true;
    }
    CHECK(passed);

    // Test manual with no reprojection
    Eigen::Matrix3d notRotation = Eigen::Matrix3d::Random();
    Eigen::Matrix4d notTransform = Eigen::Matrix4d::Identity();
    notTransform.topLeftCorner<3,3>() = notRotation;
    notTransform.topRightCorner<3,1>() = -notRotation*r_ba_ina;
    lgmath::se3::TransformationWithCovariance test_bad(notTransform, false); // don't project
    INFO("cmat: " << test_bad.matrix());
    INFO("test: " << notTransform.matrix());
    CHECK(lgmath::common::nearEqual(test_bad.matrix(), notTransform.matrix(), 1e-6));

    // We should not be able to query the covariance
    INFO("Checking covarianceSet_ flag...");
    bool passed2 = false;
    try {
      testU = test_bad.cov();
    }
    catch (const std::logic_error &e) {
      passed2 = true;
    }
    CHECK(passed2);
  }

  // Transformation(const Eigen::Matrix4d& T, const Eigen::Matrix6d& U, bool reproj = true);
  SECTION("matrix constructor with covariance" ) {
    lgmath::se3::TransformationWithCovariance test(rand.matrix(), rand.cov());
    INFO("tmat: " << rand.matrix());
    INFO("test: " << test.matrix());
    CHECK(lgmath::common::nearEqual(rand.matrix(), test.matrix(), 1e-6));

    INFO("tmat: " << rand.cov());
    INFO("test: " << test.cov());
    CHECK(lgmath::common::nearEqual(rand.cov(), test.cov(), 1e-6));

    // Test manual with no reprojection
    Eigen::Matrix3d notRotation = Eigen::Matrix3d::Random();
    Eigen::Matrix4d notTransform = Eigen::Matrix4d::Identity();
    notTransform.topLeftCorner<3,3>() = notRotation;
    notTransform.topRightCorner<3,1>() = -notRotation*r_ba_ina;
    lgmath::se3::TransformationWithCovariance test_bad(notTransform, U, false); // don't project
    INFO("cmat: " << test_bad.matrix());
    INFO("test: " << notTransform.matrix());
    CHECK(lgmath::common::nearEqual(test_bad.matrix(), notTransform.matrix(), 1e-6));

    INFO("cmat: " << test_bad.cov());
    INFO("test: " << U);
    CHECK(lgmath::common::nearEqual(test_bad.cov(), U, 1e-6));
  }

  // TransformationWithCovariance& operator=(TransformationWithCovariance T);
  SECTION("assignment operator" ) {
    lgmath::se3::TransformationWithCovariance test = rand;
    INFO("tmat: " << rand.matrix());
    INFO("test: " << test.matrix());
    CHECK(lgmath::common::nearEqual(rand.matrix(), test.matrix(), 1e-6));

    INFO("tmat: " << rand.cov());
    INFO("test: " << test.cov());
    CHECK(lgmath::common::nearEqual(rand.cov(), test.cov(), 1e-6));
  }

  // TransformationWithCovariance& operator=(TransformationWithCovariance T);
  SECTION("assignment operator with unset covariance" ) {
    lgmath::se3::TransformationWithCovariance test = lgmath::se3::TransformationWithCovariance();
    INFO("tmat: " << Eigen::Matrix4d::Identity());
    INFO("test: " << test.matrix());
    CHECK(lgmath::common::nearEqual(Eigen::Matrix4d::Identity(), test.matrix(), 1e-6));

    // We should not be able to query the covariance
    INFO("Checking covarianceSet_ flag...");
    bool passed = false;
    Eigen::Matrix<double,6,6> testU;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = true;
    }
    CHECK(passed);
  }

  // TransformationWithCovariance(const Eigen::Matrix<double,6,1>& vec, unsigned int numTerms = 0);
  SECTION("exponential map" ) {
    Eigen::Matrix<double,6,1> vec = Eigen::Matrix<double,6,1>::Random();
    Eigen::Matrix4d tmat = lgmath::se3::vec2tran(vec);
    lgmath::se3::TransformationWithCovariance testAnalytical(vec);
    lgmath::se3::TransformationWithCovariance testNumerical(vec, 15);
    INFO("tmat: " << tmat);
    INFO("testAnalytical: " << testAnalytical.matrix());
    INFO("testNumerical: " << testNumerical.matrix());
    CHECK(lgmath::common::nearEqual(tmat, testAnalytical.matrix(), 1e-6));
    CHECK(lgmath::common::nearEqual(tmat, testNumerical.matrix(), 1e-6));

    // We should not be able to query the covariance
    INFO("Checking covarianceSet_ flag...");
    bool passed = false;
    Eigen::Matrix<double,6,6> testU;
    try {
      testU = testAnalytical.cov();
    }
    catch (const std::logic_error &e) {
      passed = true;
    }
    try {
      testU = testNumerical.cov();
    }
    catch (const std::logic_error &e) {
      passed = true;
    }
    CHECK(passed);
  }

  // TransformationWithCovariance(const Eigen::Matrix<double,6,1>& vec, Eigen::Matrix6d& U, unsigned int numTerms = 0);
  SECTION("exponential map, with covariance" ) {
    Eigen::Matrix<double,6,1> vec = Eigen::Matrix<double,6,1>::Random();
    Eigen::Matrix4d tmat = lgmath::se3::vec2tran(vec);
    lgmath::se3::TransformationWithCovariance testAnalytical(vec, U);
    lgmath::se3::TransformationWithCovariance testNumerical(vec, U, 15);
    INFO("tmat: " << tmat);
    INFO("testAnalytical: " << testAnalytical.matrix());
    INFO("testNumerical: " << testNumerical.matrix());
    CHECK(lgmath::common::nearEqual(tmat, testAnalytical.matrix(), 1e-6));
    CHECK(lgmath::common::nearEqual(tmat, testNumerical.matrix(), 1e-6));

    INFO("tmat: " << U);
    INFO("testAnalytical: " << testAnalytical.cov());
    INFO("testNumerical: " << testNumerical.cov());
    CHECK(lgmath::common::nearEqual(U, testAnalytical.cov(), 1e-6));
    CHECK(lgmath::common::nearEqual(U, testNumerical.cov(), 1e-6));
  }

  // TransformationWithCovariance(const Eigen::VectorXd& vec);
  SECTION("exponential map with VectorXd" ) {
    Eigen::VectorXd vec = Eigen::Matrix<double,6,1>::Random();
    Eigen::Matrix4d tmat = lgmath::se3::vec2tran(vec);
    lgmath::se3::TransformationWithCovariance test(vec);
    INFO("tmat: " << tmat);
    INFO("test: " << test.matrix());
    CHECK(lgmath::common::nearEqual(tmat, test.matrix(), 1e-6));

    // We should not be able to query the covariance
    INFO("Checking covarianceSet_ flag...");
    bool passed = false;
    Eigen::Matrix<double,6,6> testU;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = true;
    }
    CHECK(passed);
  }

  // TransformationWithCovariance(const Eigen::VectorXd& vec, Eigen::Matrix6d& U);
  SECTION("exponential map with VectorXd, with covariance" ) {
    Eigen::VectorXd vec = Eigen::Matrix<double,6,1>::Random();
    Eigen::Matrix4d tmat = lgmath::se3::vec2tran(vec);
    lgmath::se3::TransformationWithCovariance test(vec, U);
    INFO("tmat: " << tmat);
    INFO("test: " << test.matrix());
    CHECK(lgmath::common::nearEqual(tmat, test.matrix(), 1e-6));

    INFO("tmat: " << U);
    INFO("test: " << test.cov());
    CHECK(lgmath::common::nearEqual(U, test.cov(), 1e-6));
  }

  // TransformationWithCovariance(const Eigen::VectorXd& vec);
  SECTION("exponential map with bad VectorXd" ) {
    Eigen::VectorXd vec = Eigen::Matrix<double,6,1>::Random();
    lgmath::se3::TransformationWithCovariance test(vec);

    // Wrong size vector
    Eigen::VectorXd badvec = Eigen::Matrix<double,3,1>::Random();
    lgmath::se3::TransformationWithCovariance testFailure;
    try {
      testFailure = lgmath::se3::TransformationWithCovariance(badvec, U);
    } catch (const std::invalid_argument& e) {
      testFailure = test;
    }
    INFO("tmat: " << testFailure.matrix());
    INFO("test: " << test.matrix());
    CHECK(lgmath::common::nearEqual(testFailure.matrix(), test.matrix(), 1e-6));

    // We should not be able to query the covariance
    INFO("Checking covarianceSet_ flag...");
    bool passed = false;
    Eigen::Matrix<double,6,6> testU;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = true;
    }
    CHECK(passed);
  }

  // TransformationWithCovariance(const Eigen::VectorXd& vec, Eigen::Matrix6d& U);
  SECTION("exponential map with bad VectorXd, with covariance" ) {
    Eigen::VectorXd vec = Eigen::Matrix<double,6,1>::Random();
    lgmath::se3::TransformationWithCovariance test(vec, U);

    // Wrong size vector
    Eigen::VectorXd badvec = Eigen::Matrix<double,3,1>::Random();
    lgmath::se3::TransformationWithCovariance testFailure;
    try {
      testFailure = lgmath::se3::TransformationWithCovariance(badvec, U);
    } catch (const std::invalid_argument& e) {
      testFailure = test;
    }
    INFO("tmat: " << testFailure.matrix());
    INFO("test: " << test.matrix());
    CHECK(lgmath::common::nearEqual(testFailure.matrix(), test.matrix(), 1e-6));

    INFO("tmat: " << testFailure.cov());
    INFO("test: " << test.cov());
    CHECK(lgmath::common::nearEqual(testFailure.cov(), test.cov(), 1e-6));
  }

  //TransformationWithCovariance(const Eigen::Matrix3d& C_ba,
  //                             const Eigen::Vector3d& r_ba_ina, bool reproj = true);
  SECTION("test C/r constructor" ) {
    lgmath::se3::TransformationWithCovariance tmat(C_ba, r_ba_ina);
    Eigen::Matrix4d test = Eigen::Matrix4d::Identity();
    test.topLeftCorner<3,3>() = C_ba;
    test.topRightCorner<3,1>() = -C_ba*r_ba_ina;
    INFO("tmat: " << tmat.matrix());
    INFO("test: " << test);
    CHECK(lgmath::common::nearEqual(tmat.matrix(), test, 1e-6));

    // We should not be able to query the covariance
    INFO("Checking covarianceSet_ flag...");
    Eigen::Matrix<double,6,6> testU;
    bool passed = false;
    try {
      testU = tmat.cov();
    }
    catch (const std::logic_error &e) {
      passed = true;
    }
    CHECK(passed);

    // Test manual with no reprojection
    Eigen::Matrix3d notRotation = Eigen::Matrix3d::Random();
    Eigen::Matrix4d notTransform = Eigen::Matrix4d::Identity();
    notTransform.topLeftCorner<3,3>() = notRotation;
    notTransform.topRightCorner<3,1>() = -notRotation*r_ba_ina;
    lgmath::se3::TransformationWithCovariance test_bad(notRotation, r_ba_ina, false); // don't project
    INFO("cmat: " << test_bad.matrix());
    INFO("test: " << notTransform.matrix());
    CHECK(lgmath::common::nearEqual(test_bad.matrix(), notTransform.matrix(), 1e-6));

    // We should not be able to query the covariance
    INFO("Checking covarianceSet_ flag...");
    passed = false;
    try {
      testU = test_bad.cov();
    }
    catch (const std::logic_error &e) {
      passed = true;
    }
    CHECK(passed);
  }

  //TransformationWithCovariance(const Eigen::Matrix3d& C_ba,
  //                             const Eigen::Vector3d& r_ba_ina, Eigen::Matrix6d& U, bool reproj = true);
  SECTION("test C/r constructor with covariance" ) {
    lgmath::se3::TransformationWithCovariance tmat(C_ba, r_ba_ina, U);
    Eigen::Matrix4d test = Eigen::Matrix4d::Identity();
    test.topLeftCorner<3,3>() = C_ba;
    test.topRightCorner<3,1>() = -C_ba*r_ba_ina;
    INFO("tmat: " << tmat.matrix());
    INFO("test: " << test);
    CHECK(lgmath::common::nearEqual(tmat.matrix(), test, 1e-6));

    INFO("tmat: " << tmat.cov());
    INFO("test: " << U);
    CHECK(lgmath::common::nearEqual(tmat.cov(), U, 1e-6));

    // Test manual with no reprojection
    Eigen::Matrix3d notRotation = Eigen::Matrix3d::Random();
    Eigen::Matrix4d notTransform = Eigen::Matrix4d::Identity();
    notTransform.topLeftCorner<3,3>() = notRotation;
    notTransform.topRightCorner<3,1>() = -notRotation*r_ba_ina;
    lgmath::se3::TransformationWithCovariance test_bad(notRotation, r_ba_ina, U, false); // don't project
    INFO("cmat: " << test_bad.matrix());
    INFO("test: " << notTransform.matrix());
    CHECK(lgmath::common::nearEqual(test_bad.matrix(), notTransform.matrix(), 1e-6));

    INFO("cmat: " << test_bad.matrix());
    INFO("test: " << notTransform.matrix());
    CHECK(lgmath::common::nearEqual(test_bad.cov(), U, 1e-6));
  }

} // TEST_CASE

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test some get methods
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("TransformationWithCovariance get methods.", "[lgmath]" ) {

  // Generate random transform from most basic constructor
  Eigen::Matrix<double,3,3> C_ba = lgmath::so3::vec2rot(Eigen::Matrix<double,3,1>::Random());
  Eigen::Matrix<double,3,1> r_ba_ina = Eigen::Matrix<double,3,1>::Random();
  Eigen::Matrix<double,6,6> U = Eigen::Matrix<double,6,6>::Random();
  lgmath::se3::TransformationWithCovariance T_ba(C_ba, r_ba_ina, U);

  // Construct simple eigen matrix from random rotation and translation
  Eigen::Matrix4d test = Eigen::Matrix4d::Identity();
  Eigen::Matrix<double,3,1> r_ab_inb = -C_ba*r_ba_ina;
  test.topLeftCorner<3,3>() = C_ba;
  test.topRightCorner<3,1>() = r_ab_inb;

  // Test matrix()
  INFO("T_ba: " << T_ba.matrix());
  INFO("test: " << test);
  CHECK(lgmath::common::nearEqual(T_ba.matrix(), test, 1e-6));

  // Test C_ba()
  INFO("T_ba: " << T_ba.C_ba());
  INFO("C_ba: " << C_ba);
  CHECK(lgmath::common::nearEqual(T_ba.C_ba(), C_ba, 1e-6));

  // Test r_ba_ina()
  INFO("T_ba: " << T_ba.r_ba_ina());
  INFO("r_ba_ina: " << r_ba_ina);
  CHECK(lgmath::common::nearEqual(T_ba.r_ba_ina(), r_ba_ina, 1e-6));

  // Test r_ab_inb()
  INFO("T_ba: " << T_ba.r_ab_inb());
  INFO("r_ab_inb: " << r_ab_inb);
  CHECK(lgmath::common::nearEqual(T_ba.r_ab_inb(), r_ab_inb, 1e-6));

} // TEST_CASE


/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test some get methods
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("TransformationWithCovariance operations.", "[lgmath]" ) {

  // Generate random transform from most basic constructor
  Eigen::Matrix<double,3,3> C1 = lgmath::so3::vec2rot(Eigen::Matrix<double,3,1>::Random());
  Eigen::Matrix<double,3,1> r1 = Eigen::Matrix<double,3,1>::Random();
  Eigen::Matrix<double,6,6> U1 = Eigen::Matrix<double,6,6>::Random();
  lgmath::se3::TransformationWithCovariance T1(C1, r1, U1);

  Eigen::Matrix<double,3,3> C2 = lgmath::so3::vec2rot(Eigen::Matrix<double,3,1>::Random());
  Eigen::Matrix<double,3,1> r2 = Eigen::Matrix<double,3,1>::Random();
  Eigen::Matrix<double,6,6> U2 = Eigen::Matrix<double,6,6>::Random();
  lgmath::se3::TransformationWithCovariance T2(C2, r2, U2);

  Eigen::Matrix<double,3,3> C3 = lgmath::so3::vec2rot(Eigen::Matrix<double,3,1>::Random());
  Eigen::Matrix<double,3,1> r3 = Eigen::Matrix<double,3,1>::Random();
  lgmath::se3::TransformationWithCovariance T3(C3, r3);

  Eigen::Matrix<double,3,3> C4 = lgmath::so3::vec2rot(Eigen::Matrix<double,3,1>::Random());
  Eigen::Matrix<double,3,1> r4 = Eigen::Matrix<double,3,1>::Random();
  lgmath::se3::Transformation T4(C4, r4);


  SECTION("test TWC * TWC") {
    lgmath::se3::TransformationWithCovariance test = T1*T2;
    Eigen::Matrix4d tmat = T1.matrix()*T2.matrix();
    INFO("T: " << test.matrix());
    INFO("tmat: " << tmat);
    CHECK(lgmath::common::nearEqual(test.matrix(), tmat, 1e-6));

    INFO("Checking covarianceSet_ flag...");
    Eigen::Matrix<double, 6, 6> testU;
    Eigen::Matrix<double, 6, 6> Ad = T1.adjoint();
    Eigen::Matrix<double, 6, 6> tmatU = U1 + Ad * U2 * Ad.transpose();

    bool passed = true;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = false;
      testU = tmatU;
    }
    CHECK(passed);
    CHECK(lgmath::common::nearEqual(testU, tmatU, 1e-6));
  }

  SECTION("test TWC * T") {
    lgmath::se3::TransformationWithCovariance test = T1*T4;
    Eigen::Matrix4d tmat = T1.matrix()*T4.matrix();
    INFO("T: " << test.matrix());
    INFO("tmat: " << tmat);
    CHECK(lgmath::common::nearEqual(test.matrix(), tmat, 1e-6));

    INFO("Checking covarianceSet_ flag...");
    Eigen::Matrix<double, 6, 6> testU;
    Eigen::Matrix<double, 6, 6> tmatU = U1;

    bool passed = true;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = false;
      testU = tmatU;
    }
    CHECK(passed);
    CHECK(lgmath::common::nearEqual(testU, tmatU, 1e-6));
  }

  SECTION("test T * TWC") {
    lgmath::se3::TransformationWithCovariance test = T4*T1;
    Eigen::Matrix4d tmat = T4.matrix()*T1.matrix();
    INFO("T: " << test.matrix());
    INFO("tmat: " << tmat);
    CHECK(lgmath::common::nearEqual(test.matrix(), tmat, 1e-6));

    INFO("Checking covarianceSet_ flag...");
    Eigen::Matrix<double, 6, 6> testU;
    Eigen::Matrix<double, 6, 6> Ad = T4.adjoint();
    Eigen::Matrix<double, 6, 6> tmatU = Ad * U1 * Ad.transpose();

    bool passed = true;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = false;
      testU = tmatU;
    }
    CHECK(passed);
    CHECK(lgmath::common::nearEqual(testU, tmatU, 1e-6));
  }

  SECTION("test TWC * TWC with unset covariance") {
    lgmath::se3::TransformationWithCovariance test = T1*T3;

    bool passed = false;
    Eigen::Matrix<double,6,6> testU;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = true;
    }
    CHECK(passed);
  }

  SECTION("test TWC * T with unset covariance") {
    lgmath::se3::TransformationWithCovariance test = T3*T4;

    bool passed = false;
    Eigen::Matrix<double,6,6> testU;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = true;
    }
    CHECK(passed);
  }



  SECTION("test TWC / TWC") {
    lgmath::se3::TransformationWithCovariance test = T1 / T2;
    Eigen::Matrix4d tmat = T1.matrix() * T2.matrix().inverse();
    INFO("T: " << test.matrix());
    INFO("tmat: " << tmat);
    CHECK(lgmath::common::nearEqual(test.matrix(), tmat, 1e-6));

    INFO("Checking covarianceSet_ flag...");
    Eigen::Matrix<double, 6, 6> testU;
    Eigen::Matrix<double, 6, 6> Ad1 = T1.adjoint();
    Eigen::Matrix<double, 6, 6> Ad2inv = T2.inverse().adjoint();
    Eigen::Matrix<double, 6, 6> Ad12 = Ad1 * Ad2inv;
    Eigen::Matrix<double, 6, 6> tmatU = U1 + Ad12 * U2 * Ad12.transpose();

    bool passed = true;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = false;
      testU = tmatU;
    }
    CHECK(passed);
    CHECK(lgmath::common::nearEqual(testU, tmatU, 1e-6));
  }

  SECTION("test TWC / T") {
    lgmath::se3::TransformationWithCovariance test = T1/T4;
    Eigen::Matrix4d tmat = T1.matrix()*T4.matrix().inverse();
    INFO("T: " << test.matrix());
    INFO("tmat: " << tmat);
    CHECK(lgmath::common::nearEqual(test.matrix(), tmat, 1e-6));

    INFO("Checking covarianceSet_ flag...");
    Eigen::Matrix<double, 6, 6> testU;
    Eigen::Matrix<double, 6, 6> tmatU = U1;

    bool passed = true;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = false;
      testU = tmatU;
    }
    CHECK(passed);
    CHECK(lgmath::common::nearEqual(testU, tmatU, 1e-6));
  }

  SECTION("test T / TWC") {
    lgmath::se3::TransformationWithCovariance test = T4/T1;
    Eigen::Matrix4d tmat = T4.matrix()*T1.matrix().inverse();
    INFO("T: " << test.matrix());
    INFO("tmat: " << tmat);
    CHECK(lgmath::common::nearEqual(test.matrix(), tmat, 1e-6));

    INFO("Checking covarianceSet_ flag...");
    Eigen::Matrix<double, 6, 6> testU;
    Eigen::Matrix<double, 6, 6> Ad4 = T4.adjoint();
    Eigen::Matrix<double, 6, 6> Ad1inv = T1.inverse().adjoint();
    Eigen::Matrix<double, 6, 6> Ad41 = Ad4 * Ad1inv;
    Eigen::Matrix<double, 6, 6> tmatU = Ad41 * U1 * Ad41.transpose();

    bool passed = true;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = false;
      testU = tmatU;
    }
    CHECK(passed);
    CHECK(lgmath::common::nearEqual(testU, tmatU, 1e-6));
  }

  SECTION("test TWC / TWC with unset covariance") {
    lgmath::se3::TransformationWithCovariance test = T1/T3;

    bool passed = false;
    Eigen::Matrix<double,6,6> testU;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = true;
    }
    CHECK(passed);
  }

  SECTION("test TWC * T with unset covariance") {
    lgmath::se3::TransformationWithCovariance test = T3/T4;

    bool passed = false;
    Eigen::Matrix<double,6,6> testU;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = true;
    }
    CHECK(passed);
  }

  SECTION("test TWC inverse") {
    lgmath::se3::TransformationWithCovariance test = T1.inverse();
    Eigen::Matrix4d tmat = T1.matrix().inverse();
    INFO("T: " << test.matrix());
    INFO("tmat: " << tmat);
    CHECK(lgmath::common::nearEqual(test.matrix(), tmat, 1e-6));

    INFO("Checking covarianceSet_ flag...");
    Eigen::Matrix<double, 6, 6> testU;
    Eigen::Matrix<double, 6, 6> tmatU = test.adjoint() * U1 * test.adjoint().transpose();

    bool passed = true;
    try {
      testU = test.cov();
    }
    catch (const std::logic_error &e) {
      passed = false;
      testU = tmatU;
    }
    CHECK(passed);
    CHECK(lgmath::common::nearEqual(testU, tmatU, 1e-6));
  }

  SECTION("test setting covariance") {
    lgmath::se3::TransformationWithCovariance T5(C3, r3);
    T5.setCovariance(U2);

    INFO("Checking covarianceSet_ flag...");
    Eigen::Matrix<double, 6, 6> testU;
    Eigen::Matrix<double, 6, 6> tmatU = U2;

    bool passed = true;
    try {
      testU = T5.cov();
    }
    catch (const std::logic_error &e) {
      passed = false;
      testU = tmatU;
    }
    CHECK(passed);
    CHECK(lgmath::common::nearEqual(testU, tmatU, 1e-6));


    T3.setZeroCovariance();

    INFO("Checking covarianceSet_ flag...");
    tmatU = Eigen::Matrix<double, 6, 6>::Zero();

    passed = true;
    try {
      testU = T3.cov();
    }
    catch (const std::logic_error &e) {
      passed = false;
      testU = tmatU;
    }
    CHECK(passed);
    CHECK(lgmath::common::nearEqual(testU, tmatU, 1e-6));

  }

} // TEST_CASE



