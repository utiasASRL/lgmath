//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TransformWithCovarianceTests.cpp
/// \brief Unit tests for the implementation of the transformation with covariance class.
/// \details Unit tests for the various TransformWithCovariance class operations, that test
///          both functionality and correct interoperation with the Transform class.
///
/// \author Kai van Es
//////////////////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <iostream>
#include <iomanip>
#include <ios>
#include <typeinfo>

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
/// Convenience functions and macros
/////////////////////////////////////////////////////////////////////////////////////////////

// Convenience function to convert the presence/absense of an exception into a boolean test value
bool covAccessDidRaise(lgmath::se3::TransformationWithCovariance& T) {
  bool raised = false;

  try {
    // We don't need the value, we just want to know if it works
    (void)T.cov();
  }
  catch (const std::logic_error& e) {
    raised = true;
  }

  return raised;
}

// Convenience function to wrap retreiving the covariance in an exception handler and return an
// impossible value on failure.  This prevents the test cases from crashing in the event that
// something was implemented poorly.
Eigen::Matrix<double,6,6> covSafe(lgmath::se3::TransformationWithCovariance& T) {
  Eigen::Matrix<double,6,6> U;

  try {
    U = T.cov();
  }
  catch (const std::logic_error& err) {
    // This should safely fail any comparison test, as a covariance can't be all negative
    // and we test using uniform matrices on [0,1]
    U = Eigen::Matrix<double,6,6>::Ones()*-100;
  }

  return U;
}

// Convenience macro: test that accessing covariance doesn't raise an error, and that covarianceSet_ is true
#define CHECK_HAS_COVARIANCE(T) \
  INFO("Checking for covarianceSet_(true): " << std::boolalpha << T.covarianceSet() <<"\n"); \
  CHECK(!covAccessDidRaise(T)); \
  CHECK(T.covarianceSet());

// Convenience macro: test that accessing covariance raises an error, and that covarianceSet_ is false
#define CHECK_NO_COVARIANCE(T) \
  INFO("Checking for covarianceSet_(false): " << std::boolalpha << T.covarianceSet() <<"\n");\
  CHECK(covAccessDidRaise(T));  \
  CHECK(!T.covarianceSet());

// Checks that a covariance is present, as well as that it is equal to something
#define CHECK_EQ_COVARIANCE(T,U) \
  INFO("true cov: \n" << U <<"\n\ntest cov: \n" << covSafe(T) <<"\n");  \
  CHECK(lgmath::common::nearEqual(U, covSafe(T), 1e-6));

// Checks that two matrices are equal
#define CHECK_EQ(A,B) \
  INFO("true mat: \n" << A <<"\n\ntest mat: \n" << B <<"\n");  \
  CHECK(lgmath::common::nearEqual(A, B, 1e-6));


/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of transformation constructors
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("TransformationWithCovariance Constructors.", "[lgmath]" ) {

  // Generate random transform from most basic constructor
  Eigen::Matrix<double,3,3> C_ba = lgmath::so3::vec2rot(Eigen::Matrix<double,3,1>::Random());
  Eigen::Matrix<double,3,1> r_ba_ina = Eigen::Matrix<double,3,1>::Random();
  Eigen::Matrix<double,6,6> U = Eigen::Matrix<double,6,6>::Random();
  Eigen::Matrix<double,6,6> Z = Eigen::Matrix<double,6,6>::Zero();
  lgmath::se3::Transformation randBase(C_ba, r_ba_ina);
  lgmath::se3::TransformationWithCovariance rand(C_ba, r_ba_ina, U);

  // TransformationWithCovariance();
  SECTION("default" ) {
    lgmath::se3::TransformationWithCovariance tmatrix;
    Eigen::Matrix4d test = Eigen::Matrix4d::Identity();

    CHECK_EQ(test, tmatrix.matrix())
    CHECK_NO_COVARIANCE(tmatrix);
  }

  // TransformationWithCovariance(const TransformationWithCovariance& T);
  SECTION("copy constructor" ) {
    lgmath::se3::TransformationWithCovariance test(rand);
    CHECK_EQ(rand.matrix(), test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, covSafe(rand));
  }


  // TransformationWithCovariance(const Transformation& T);
  SECTION("copy constructor (from base)" ) {
    // With no flag set, we shouldn't have a covariance
    lgmath::se3::TransformationWithCovariance test(randBase);
    CHECK_EQ(randBase.matrix(), test.matrix());
    CHECK_NO_COVARIANCE(test);

    // With the flag set, we should have a zero covariance
    lgmath::se3::TransformationWithCovariance test2(randBase, true);
    CHECK_EQ(randBase.matrix(), test2.matrix());
    CHECK_HAS_COVARIANCE(test2);
    CHECK_EQ_COVARIANCE(test2, Z);
  }

  // TransformationWithCovariance(const Transformation& T, Eigen::Matrix6d& U);
  SECTION("copy constructor (from base, with covariance)" ) {
    lgmath::se3::TransformationWithCovariance test(randBase, U);

    CHECK_EQ(randBase.matrix(), test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, U);
  }

  // Transformation(const Eigen::Matrix4d& T, bool reproj = true);
  SECTION("matrix constructor" ) {
    lgmath::se3::TransformationWithCovariance test(rand.matrix());

    CHECK_EQ(rand.matrix(), test.matrix());
    CHECK_NO_COVARIANCE(test);

    // Test manual with no reprojection
    Eigen::Matrix3d notRotation = Eigen::Matrix3d::Random();
    Eigen::Matrix4d notTransform = Eigen::Matrix4d::Identity();
    notTransform.topLeftCorner<3,3>() = notRotation;
    notTransform.topRightCorner<3,1>() = -notRotation*r_ba_ina;
    lgmath::se3::TransformationWithCovariance test_bad(notTransform, false); // don't project

    CHECK_EQ(notTransform, test_bad.matrix());
    CHECK_NO_COVARIANCE(test_bad);
  }

  // Transformation(const Eigen::Matrix4d& T, const Eigen::Matrix6d& U, bool reproj = true);
  SECTION("matrix constructor with covariance" ) {
    lgmath::se3::TransformationWithCovariance test(rand.matrix(), covSafe(rand));

    CHECK_EQ(rand.matrix(), test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, covSafe(rand));

    // Test manual with no reprojection
    Eigen::Matrix3d notRotation = Eigen::Matrix3d::Random();
    Eigen::Matrix4d notTransform = Eigen::Matrix4d::Identity();
    notTransform.topLeftCorner<3,3>() = notRotation;
    notTransform.topRightCorner<3,1>() = -notRotation*r_ba_ina;
    lgmath::se3::TransformationWithCovariance test_bad(notTransform, U, false); // don't project

    CHECK_EQ(notTransform, test_bad.matrix());
    CHECK_HAS_COVARIANCE(test_bad);
    CHECK_EQ_COVARIANCE(test_bad, U);
  }

  // TransformationWithCovariance& operator=(TransformationWithCovariance T);
  SECTION("assignment operator" ) {
    lgmath::se3::TransformationWithCovariance test = rand;

    CHECK_EQ(rand.matrix(), test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, covSafe(rand));
  }

  // TransformationWithCovariance& operator=(TransformationWithCovariance T);
  SECTION("assignment operator with unset covariance" ) {
    lgmath::se3::TransformationWithCovariance test = lgmath::se3::TransformationWithCovariance();

    CHECK_EQ(Eigen::Matrix4d::Identity(), test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  // TransformationWithCovariance(const Eigen::Matrix<double,6,1>& vec, unsigned int numTerms = 0);
  SECTION("exponential map" ) {
    Eigen::Matrix<double,6,1> vec = Eigen::Matrix<double,6,1>::Random();
    Eigen::Matrix4d tmat = lgmath::se3::vec2tran(vec);
    lgmath::se3::TransformationWithCovariance testAnalytical(vec);
    lgmath::se3::TransformationWithCovariance testNumerical(vec, 15);

    INFO("Analytical Test: \n\n");
    CHECK_EQ(tmat, testAnalytical.matrix());
    CHECK_NO_COVARIANCE(testAnalytical);

    INFO("Numerical Test: \n\n");
    CHECK_EQ(tmat, testNumerical.matrix());
    CHECK_NO_COVARIANCE(testNumerical);
  }

  // TransformationWithCovariance(const Eigen::Matrix<double,6,1>& vec, Eigen::Matrix6d& U, unsigned int numTerms = 0);
  SECTION("exponential map, with covariance" ) {
    Eigen::Matrix<double,6,1> vec = Eigen::Matrix<double,6,1>::Random();
    Eigen::Matrix4d tmat = lgmath::se3::vec2tran(vec);
    lgmath::se3::TransformationWithCovariance testAnalytical(vec, U);
    lgmath::se3::TransformationWithCovariance testNumerical(vec, U, 15);

    INFO("Analytical Test: \n\n");
    CHECK_EQ(tmat, testAnalytical.matrix());
    CHECK_HAS_COVARIANCE(testAnalytical);
    CHECK_EQ_COVARIANCE(testAnalytical, U);

    INFO("Numerical Test: \n\n");
    CHECK_EQ(tmat, testNumerical.matrix());
    CHECK_HAS_COVARIANCE(testNumerical);
    CHECK_EQ_COVARIANCE(testNumerical, U);
  }

  // TransformationWithCovariance(const Eigen::VectorXd& vec);
  SECTION("exponential map with VectorXd" ) {
    Eigen::VectorXd vec = Eigen::Matrix<double,6,1>::Random();
    Eigen::Matrix4d tmat = lgmath::se3::vec2tran(vec);
    lgmath::se3::TransformationWithCovariance test(vec);

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  // TransformationWithCovariance(const Eigen::VectorXd& vec, Eigen::Matrix6d& U);
  SECTION("exponential map with VectorXd, with covariance" ) {
    Eigen::VectorXd vec = Eigen::Matrix<double,6,1>::Random();
    Eigen::Matrix4d tmat = lgmath::se3::vec2tran(vec);
    lgmath::se3::TransformationWithCovariance test(vec, U);

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, U);
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

    CHECK_EQ(test.matrix(), testFailure.matrix());
    CHECK_NO_COVARIANCE(test);
    CHECK_NO_COVARIANCE(testFailure);

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

    CHECK_EQ(test.matrix(), testFailure.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, U);
    CHECK_HAS_COVARIANCE(testFailure);
    CHECK_EQ_COVARIANCE(testFailure, U);
  }

  //TransformationWithCovariance(const Eigen::Matrix3d& C_ba,
  //                             const Eigen::Vector3d& r_ba_ina, bool reproj = true);
  SECTION("test C/r constructor" ) {
    lgmath::se3::TransformationWithCovariance test(C_ba, r_ba_ina);
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    tmat.topLeftCorner<3,3>() = C_ba;
    tmat.topRightCorner<3,1>() = -C_ba*r_ba_ina;

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);

    // Test manual with no reprojection
    Eigen::Matrix3d notRotation = Eigen::Matrix3d::Random();
    Eigen::Matrix4d notTransform = Eigen::Matrix4d::Identity();
    notTransform.topLeftCorner<3,3>() = notRotation;
    notTransform.topRightCorner<3,1>() = -notRotation*r_ba_ina;
    lgmath::se3::TransformationWithCovariance test_bad(notRotation, r_ba_ina, false); // don't project

    CHECK_EQ(notTransform, test_bad.matrix());
    CHECK_NO_COVARIANCE(test_bad);
  }

  //TransformationWithCovariance(const Eigen::Matrix3d& C_ba,
  //                             const Eigen::Vector3d& r_ba_ina, Eigen::Matrix6d& U, bool reproj = true);
  SECTION("test C/r constructor with covariance" ) {
    lgmath::se3::TransformationWithCovariance test(C_ba, r_ba_ina, U);
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    tmat.topLeftCorner<3,3>() = C_ba;
    tmat.topRightCorner<3,1>() = -C_ba*r_ba_ina;

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, U);

    // Test manual with no reprojection
    Eigen::Matrix3d notRotation = Eigen::Matrix3d::Random();
    Eigen::Matrix4d notTransform = Eigen::Matrix4d::Identity();
    notTransform.topLeftCorner<3,3>() = notRotation;
    notTransform.topRightCorner<3,1>() = -notRotation*r_ba_ina;
    lgmath::se3::TransformationWithCovariance test_bad(notRotation, r_ba_ina, U, false); // don't project

    CHECK_EQ(notTransform, test_bad.matrix());
    CHECK_HAS_COVARIANCE(test_bad);
    CHECK_EQ_COVARIANCE(test_bad, U);
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
/// \brief Test math operations and flag propagation
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


  /// Operator: *=

  SECTION("test TWC *= TWC") {
    lgmath::se3::TransformationWithCovariance test(T1);
    test *= T2;
    Eigen::Matrix4d tmat = T1.matrix()*T2.matrix();

    Eigen::Matrix<double, 6, 6> Ad = T1.adjoint();
    Eigen::Matrix<double, 6, 6> tmatU = U1 + Ad * U2 * Ad.transpose();

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  SECTION("test TWC *= T") {
    lgmath::se3::TransformationWithCovariance test(T1);
    test *= T4;
    Eigen::Matrix4d tmat = T1.matrix()*T4.matrix();

    Eigen::Matrix<double, 6, 6> tmatU = U1;

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  SECTION("test T *= TWC") {
    lgmath::se3::Transformation test(T4);
    test *= T1;
    Eigen::Matrix4d tmat = T4.matrix()*T1.matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK(typeid(test) == typeid(lgmath::se3::Transformation));  //Make sure nothing weird happened
  }

  SECTION("test TWC *= TWC(unset)") {
    lgmath::se3::TransformationWithCovariance test(T1);
    test *= T3;
    Eigen::Matrix4d tmat = T1.matrix()*T3.matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  SECTION("test TWC(unset) *= TWC") {
    lgmath::se3::TransformationWithCovariance test(T3);
    test *= T1;
    Eigen::Matrix4d tmat = T3.matrix()*T1.matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  SECTION("test TWC(unset) *= TWC(unset)") {
    lgmath::se3::TransformationWithCovariance test(T3);
    test *= T3;
    Eigen::Matrix4d tmat = T3.matrix()*T3.matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  SECTION("test T *= TWC(unset)") {
    lgmath::se3::Transformation test(T4);
    test *= T3;
    Eigen::Matrix4d tmat = T4.matrix()*T3.matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK(typeid(test) == typeid(lgmath::se3::Transformation));  //Make sure nothing weird happened
  }

  SECTION("test TWC(unset) *= T") {
    lgmath::se3::TransformationWithCovariance test(T3);
    test *= T4;
    Eigen::Matrix4d tmat = T3.matrix()*T4.matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }


  /// Operator: /=

  SECTION("test TWC /= TWC") {
    lgmath::se3::TransformationWithCovariance test(T1);
    test /= T2;
    Eigen::Matrix4d tmat = T1.matrix()*T2.matrix().inverse();

    Eigen::Matrix<double, 6, 6> Ad1 = T1.adjoint();
    Eigen::Matrix<double, 6, 6> Ad2inv = T2.inverse().adjoint();
    Eigen::Matrix<double, 6, 6> Ad12 = Ad1 * Ad2inv;
    Eigen::Matrix<double, 6, 6> tmatU = U1 + Ad12 * U2 * Ad12.transpose();

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  SECTION("test TWC /= T") {
    lgmath::se3::TransformationWithCovariance test(T1);
    test /= T4;
    Eigen::Matrix4d tmat = T1.matrix()*T4.matrix().inverse();

    Eigen::Matrix<double, 6, 6> tmatU = U1;

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  SECTION("test T /= TWC") {
    lgmath::se3::Transformation test(T4);
    test /= T1;
    Eigen::Matrix4d tmat = T4.matrix()*T1.matrix().inverse();

    CHECK_EQ(tmat, test.matrix());
    CHECK(typeid(test) == typeid(lgmath::se3::Transformation));  //Make sure nothing weird happened
  }

  SECTION("test TWC /= TWC(unset)") {
    lgmath::se3::TransformationWithCovariance test(T1);
    test /= T3;
    Eigen::Matrix4d tmat = T1.matrix()*T3.matrix().inverse();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  SECTION("test TWC(unset) /= TWC") {
    lgmath::se3::TransformationWithCovariance test(T3);
    test /= T1;
    Eigen::Matrix4d tmat = T3.matrix()*T1.matrix().inverse();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  SECTION("test TWC(unset) /= TWC(unset)") {
    lgmath::se3::TransformationWithCovariance test(T3);
    test /= T3;
    Eigen::Matrix4d tmat = T3.matrix()*T3.matrix().inverse();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  SECTION("test T /= TWC(unset)") {
    lgmath::se3::Transformation test(T4);
    test /= T3;
    Eigen::Matrix4d tmat = T4.matrix()*T3.matrix().inverse();

    CHECK_EQ(tmat, test.matrix());
    CHECK(typeid(test) == typeid(lgmath::se3::Transformation));  //Make sure nothing weird happened
  }

  SECTION("test TWC(unset) /= T") {
    lgmath::se3::TransformationWithCovariance test(T3);
    test /= T4;
    Eigen::Matrix4d tmat = T3.matrix()*T4.matrix().inverse();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }


  /// Operator: *

  SECTION("test TWC * TWC") {
    lgmath::se3::TransformationWithCovariance test = T1*T2;
    Eigen::Matrix4d tmat = T1.matrix()*T2.matrix();

    Eigen::Matrix<double, 6, 6> Ad = T1.adjoint();
    Eigen::Matrix<double, 6, 6> tmatU = U1 + Ad * U2 * Ad.transpose();

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  SECTION("test TWC * T") {
    lgmath::se3::TransformationWithCovariance test = T1*T4;
    Eigen::Matrix4d tmat = T1.matrix()*T4.matrix();

    Eigen::Matrix<double, 6, 6> testU;
    Eigen::Matrix<double, 6, 6> tmatU = U1;

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  SECTION("test T * TWC") {
    lgmath::se3::TransformationWithCovariance test = T4*T1;
    Eigen::Matrix4d tmat = T4.matrix()*T1.matrix();

    Eigen::Matrix<double, 6, 6> Ad = T4.adjoint();
    Eigen::Matrix<double, 6, 6> tmatU = Ad * U1 * Ad.transpose();

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  SECTION("test TWC * TWC(unset)") {
    lgmath::se3::TransformationWithCovariance test = T1*T3;
    Eigen::Matrix4d tmat = T1.matrix()*T3.matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  SECTION("test TWC(unset) * TWC") {
    lgmath::se3::TransformationWithCovariance test = T3*T1;
    Eigen::Matrix4d tmat = T3.matrix()*T1.matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  SECTION("test TWC(unset) * TWC(unset)") {
    lgmath::se3::TransformationWithCovariance test = T3*T3;
    Eigen::Matrix4d tmat = T3.matrix()*T3.matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  SECTION("test TWC(unset) * T") {
    lgmath::se3::TransformationWithCovariance test = T3*T4;
    Eigen::Matrix4d tmat = T3.matrix()*T4.matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  SECTION("test T * TWC(unset)") {
    lgmath::se3::TransformationWithCovariance test = T4*T3;
    Eigen::Matrix4d tmat = T4.matrix()*T3.matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }


  /// Operator: /

  SECTION("test TWC / TWC") {
    lgmath::se3::TransformationWithCovariance test = T1 / T2;
    Eigen::Matrix4d tmat = T1.matrix() * T2.matrix().inverse();

    Eigen::Matrix<double, 6, 6> Ad1 = T1.adjoint();
    Eigen::Matrix<double, 6, 6> Ad2inv = T2.inverse().adjoint();
    Eigen::Matrix<double, 6, 6> Ad12 = Ad1 * Ad2inv;
    Eigen::Matrix<double, 6, 6> tmatU = U1 + Ad12 * U2 * Ad12.transpose();

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  SECTION("test TWC / T") {
    lgmath::se3::TransformationWithCovariance test = T1/T4;
    Eigen::Matrix4d tmat = T1.matrix()*T4.matrix().inverse();

    Eigen::Matrix<double, 6, 6> tmatU = U1;

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  SECTION("test T / TWC") {
    lgmath::se3::TransformationWithCovariance test = T4/T1;
    Eigen::Matrix4d tmat = T4.matrix()*T1.matrix().inverse();

    Eigen::Matrix<double, 6, 6> Ad4 = T4.adjoint();
    Eigen::Matrix<double, 6, 6> Ad1inv = T1.inverse().adjoint();
    Eigen::Matrix<double, 6, 6> Ad41 = Ad4 * Ad1inv;
    Eigen::Matrix<double, 6, 6> tmatU = Ad41 * U1 * Ad41.transpose();

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  SECTION("test TWC / TWC(unset)") {
    lgmath::se3::TransformationWithCovariance test = T1/T3;
    Eigen::Matrix4d tmat = T1.matrix()*T3.matrix().inverse();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  SECTION("test TWC(unset) / TWC") {
    lgmath::se3::TransformationWithCovariance test = T3/T1;
    Eigen::Matrix4d tmat = T3.matrix()*T1.matrix().inverse();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  SECTION("test TWC(unset) / TWC(unset)") {
    lgmath::se3::TransformationWithCovariance test = T3/T3;
    Eigen::Matrix4d tmat = T3.matrix()*T3.matrix().inverse();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  SECTION("test TWC(unset) / T") {
    lgmath::se3::TransformationWithCovariance test = T3/T4;
    Eigen::Matrix4d tmat = T3.matrix()*T4.matrix().inverse();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  SECTION("test T / TWC(unset)") {
    lgmath::se3::TransformationWithCovariance test = T4/T3;
    Eigen::Matrix4d tmat = T4.matrix()*T3.matrix().inverse();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }


  /// Misc

  SECTION("test TWC inverse") {
    lgmath::se3::TransformationWithCovariance test = T1.inverse();
    Eigen::Matrix4d tmat = T1.matrix().inverse();

    Eigen::Matrix<double, 6, 6> tmatU = test.adjoint() * U1 * test.adjoint().transpose();

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  SECTION("test setting covariance") {
    lgmath::se3::TransformationWithCovariance T5(C3, r3);
    T5.setCovariance(U2);
    Eigen::Matrix<double, 6, 6> tmatU = U2;

    CHECK_HAS_COVARIANCE(T5);
    CHECK_EQ_COVARIANCE(T5, tmatU);

    T3.setZeroCovariance();
    tmatU = Eigen::Matrix<double, 6, 6>::Zero();

    CHECK_HAS_COVARIANCE(T3);
    CHECK_EQ_COVARIANCE(T3, tmatU);

  }

} // TEST_CASE



