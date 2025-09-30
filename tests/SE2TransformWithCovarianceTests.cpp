//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SE2TransformWithCovarianceTests.cpp
/// \brief Unit tests for the implementation of the SE(2) transformation with
/// covariance class. \details Unit tests for the various
/// SE(2) TransformWithCovariance class operations, that test both functionality and correct
/// interoperation with the Transform class.
///
/// \author Daniil Lisus
//////////////////////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>

#include <math.h>
#include <iomanip>
#include <ios>
#include <iostream>
#include <typeinfo>

#include <Eigen/Dense>
#include <lgmath/CommonMath.hpp>

#include <lgmath/se2/Operations.hpp>
#include <lgmath/se2/TransformationWithCovariance.hpp>
#include <lgmath/so2/Operations.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////
///
/// UNIT TESTS OF SE(2) TRANSFORMATION MATRIX WITH COVARIANCE
///
/// NOTE: These tests are mainly comparitive against the base Transform, and
/// assume that the relevant methods in the base class have all passed testing.
///
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
/// Convenience functions and macros
/////////////////////////////////////////////////////////////////////////////////////////////

// Convenience function to convert the presence/absense of an exception into a
// boolean test value
bool covAccessDidRaise(lgmath::se2::TransformationWithCovariance& T) {
  bool raised = false;

  try {
    T.cov();
  } catch (const std::logic_error& e) {
    raised = true;
  }

  return raised;
}

// Convenience function to wrap retreiving the covariance in an exception
// handler and return an impossible value on failure.  This prevents the test
// cases from crashing in the event that something was implemented poorly.
Eigen::Matrix<double, 3, 3> covSafe(lgmath::se2::TransformationWithCovariance& T) {
  Eigen::Matrix<double, 3, 3> U;

  try {
    U = T.cov();
  } catch (const std::logic_error& err) {
    U.setConstant(-999);
  }

  return U;
}

// Convenience macro: test that accessing covariance doesn't raise an error, and
// that covarianceSet_ is true
#define CHECK_HAS_COVARIANCE(T)                                        \
  std::cout << "Checking for covarianceSet_(true): " << std::boolalpha \
            << T.covarianceSet() << std::endl;                         \
  EXPECT_TRUE(!covAccessDidRaise(T));                                  \
  EXPECT_TRUE(T.covarianceSet());

// Convenience macro: test that accessing covariance raises an error, and that
// covarianceSet_ is false
#define CHECK_NO_COVARIANCE(T)                                          \
  std::cout << "Checking for covarianceSet_(false): " << std::boolalpha \
            << T.covarianceSet() << std::endl;                          \
  EXPECT_TRUE(covAccessDidRaise(T));                                    \
  EXPECT_TRUE(!T.covarianceSet());

// Checks that a covariance is present, as well as that it is equal to something
#define CHECK_EQ_COVARIANCE(T, U)       \
  std::cout << "true cov: \n"           \
            << U << "\ntest cov: \n"    \
            << covSafe(T) << std::endl; \
  EXPECT_TRUE(lgmath::common::nearEqual(U, covSafe(T), 1e-6));

// Checks that two matrices are equal
#define CHECK_EQ(A, B)                                                    \
  std::cout << "true mat: \n" << A << "\ntest mat: \n" << B << std::endl; \
  EXPECT_TRUE(lgmath::common::nearEqual(A, B, 1e-6));

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SE(2) transformation with covariance constructors
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, SE2TransformationWithCovarianceConstructors) {
  // Generate random transform from most basic constructor
  Eigen::Matrix<double, 2, 2> C_ab =
      lgmath::so2::vec2rot(Eigen::Matrix<double, 1, 1>::Random()(0));
  Eigen::Matrix<double, 2, 1> r_ba_ina = Eigen::Matrix<double, 2, 1>::Random();
  Eigen::Matrix<double, 3, 3> U = Eigen::Matrix<double, 3, 3>::Random();
  Eigen::Matrix<double, 3, 3> Z = Eigen::Matrix<double, 3, 3>::Zero();
  lgmath::se2::Transformation randBase(C_ab, r_ba_ina);
  lgmath::se2::TransformationWithCovariance rand(C_ab, r_ba_ina, U);

  // TransformationWithCovariance();
  // default"
  {
    lgmath::se2::TransformationWithCovariance tmatrix;
    Eigen::Matrix3d test = Eigen::Matrix3d::Identity();

    CHECK_EQ(test, tmatrix.matrix())
    CHECK_NO_COVARIANCE(tmatrix);
  }

  // TransformationWithCovariance(const TransformationWithCovariance& T);
  // copy constructor"
  {
    lgmath::se2::TransformationWithCovariance test(rand);
    CHECK_EQ(rand.matrix(), test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, covSafe(rand));
  }

  // TransformationWithCovariance(const Transformation& T);
  // copy constructor (from base)"
  {
    // With no flag set, we shouldn't have a covariance
    lgmath::se2::TransformationWithCovariance test(randBase);
    CHECK_EQ(randBase.matrix(), test.matrix());
    CHECK_NO_COVARIANCE(test);

    // With the flag set, we should have a zero covariance
    lgmath::se2::TransformationWithCovariance test2(randBase, true);
    CHECK_EQ(randBase.matrix(), test2.matrix());
    CHECK_HAS_COVARIANCE(test2);
    CHECK_EQ_COVARIANCE(test2, Z);
  }

  // TransformationWithCovariance(const Transformation& T, Eigen::Matrix3d& U);
  // copy constructor (from base, with covariance)"
  {
    lgmath::se2::TransformationWithCovariance test(randBase, U);

    CHECK_EQ(randBase.matrix(), test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, U);
  }

  // Transformation(const Eigen::Matrix3d& T);
  // matrix constructor"
  {
    lgmath::se2::TransformationWithCovariance test(rand.matrix());

    CHECK_EQ(rand.matrix(), test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  // Transformation(const Eigen::Matrix3d& T, const Eigen::Matrix3d& U);
  // matrix constructor with covariance"
  {
    lgmath::se2::TransformationWithCovariance test(rand.matrix(),
                                                   covSafe(rand));

    CHECK_EQ(rand.matrix(), test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, covSafe(rand));
  }

  // TransformationWithCovariance& operator=(TransformationWithCovariance T);
  // assignment operator"
  {
    lgmath::se2::TransformationWithCovariance test;
    test = rand;

    CHECK_EQ(rand.matrix(), test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, covSafe(rand));
    EXPECT_TRUE(typeid(test) ==
                typeid(lgmath::se2::TransformationWithCovariance));
  }

  // TransformationWithCovariance& operator=(TransformationWithCovariance T);
  // assignment operator with unset covariance"
  {
    lgmath::se2::TransformationWithCovariance test;
    test = lgmath::se2::TransformationWithCovariance();

    CHECK_EQ(Eigen::Matrix3d::Identity(), test.matrix());
    CHECK_NO_COVARIANCE(test);
    EXPECT_TRUE(typeid(test) ==
                typeid(lgmath::se2::TransformationWithCovariance));
  }

  // TransformationWithCovariance& operator=(Transformation T);
  // assignment operator to base transform"
  {
    lgmath::se2::TransformationWithCovariance test;
    test = randBase;

    CHECK_EQ(randBase.matrix(), test.matrix());
    CHECK_NO_COVARIANCE(test);
    EXPECT_TRUE(typeid(test) ==
                typeid(lgmath::se2::TransformationWithCovariance));
  }

  // TransformationWithCovariance& operator=(Transformation T);
  // assignment operator of base transform to subclass"
  {
    lgmath::se2::Transformation test;
    test = rand;

    CHECK_EQ(rand.matrix(), test.matrix());
    EXPECT_TRUE(typeid(test) == typeid(lgmath::se2::Transformation));
  }

  // TransformationWithCovariance& operator=(Transformation T);
  // assignment operator to base transform, when covariance is previously set"
  {
    lgmath::se2::TransformationWithCovariance test(rand);
    test = randBase;

    CHECK_EQ(randBase.matrix(), test.matrix());
    CHECK_NO_COVARIANCE(test);
    EXPECT_TRUE(typeid(test) ==
                typeid(lgmath::se2::TransformationWithCovariance));
  }

  // TransformationWithCovariance(const Eigen::Vector3d& vec);
  // exponential map"
  {
    Eigen::Vector3d vec = Eigen::Vector3d::Random();
    Eigen::Matrix3d tmat = lgmath::se2::vec2tran(vec);
    lgmath::se2::TransformationWithCovariance test(vec);

    std::cout << "Exponential map test: \n" << std::endl;
    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  // TransformationWithCovariance(const Eigen::Vector3d& vec,
  // Eigen::Matrix3d& U); exponential map, with covariance"
  {
    Eigen::Vector3d vec = Eigen::Vector3d::Random();
    Eigen::Matrix3d tmat = lgmath::se2::vec2tran(vec);
    lgmath::se2::TransformationWithCovariance test(vec, U);

    std::cout << "Exponential map with covariance test: \n" << std::endl;
    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, U);
  }

  // TransformationWithCovariance(const Eigen::Matrix2d& C_ab,
  //                             const Eigen::Vector2d& r_ba_ina);
  // test C/r constructor"
  {
    lgmath::se2::TransformationWithCovariance test(C_ab, r_ba_ina);
    Eigen::Matrix3d tmat = Eigen::Matrix3d::Identity();
    tmat.topLeftCorner<2, 2>() = C_ab;
    tmat.topRightCorner<2, 1>() = r_ba_ina;

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  // TransformationWithCovariance(const Eigen::Matrix2d& C_ab,
  //                             const Eigen::Vector2d& r_ba_ina,
  //                             Eigen::Matrix3d& U);
  // test C/r constructor with covariance"
  {
    lgmath::se2::TransformationWithCovariance test(C_ab, r_ba_ina, U);
    Eigen::Matrix3d tmat = Eigen::Matrix3d::Identity();
    tmat.topLeftCorner<2, 2>() = C_ab;
    tmat.topRightCorner<2, 1>() = r_ba_ina;

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, U);
  }

  // TransformationWithCovariance(TransformationWithCovariance&&);
  // move constructor"
  {
    lgmath::se2::TransformationWithCovariance test(std::move(rand));

    CHECK_EQ(rand.matrix(), test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, covSafe(rand));
  }

  // TransformationWithCovariance = TransformationWithCovariance&&;
  // move assignment"
  {
    lgmath::se2::TransformationWithCovariance test;
    auto rand2 = rand;
    test = std::move(rand);
    rand = rand2;

    CHECK_EQ(rand.matrix(), test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, covSafe(rand));
  }

  // TransformationWithCovariance(Transformation&&);
  // move constructor (subclass from base)"
  {
    auto randBase2 = randBase;
    lgmath::se2::TransformationWithCovariance test(std::move(randBase));
    randBase = randBase2;

    CHECK_EQ(randBase.matrix(), test.matrix());
    EXPECT_TRUE(typeid(test) ==
                typeid(lgmath::se2::TransformationWithCovariance));
    CHECK_NO_COVARIANCE(test);
  }

  // TransformationWithCovariance = Transformation&&;
  // move assignment (subclass from base)"
  {
    lgmath::se2::TransformationWithCovariance test;
    auto randBase2 = randBase;
    test = std::move(randBase);
    randBase = randBase2;

    CHECK_EQ(randBase.matrix(), test.matrix());
    EXPECT_TRUE(typeid(test) ==
                typeid(lgmath::se2::TransformationWithCovariance));
    CHECK_NO_COVARIANCE(test);
  }

  // Transformation(TransformationWithCovariance&&);
  // move constructor (base from subclass)"
  {
    auto rand2 = rand;
    lgmath::se2::Transformation test(std::move(rand));
    rand = rand2;

    CHECK_EQ(rand.matrix(), test.matrix());
    EXPECT_TRUE(typeid(test) == typeid(lgmath::se2::Transformation));
  }

  // Transformation = TransformationWithCovariance&&;
  // base move assignment (base from subclass)"
  {
    lgmath::se2::Transformation test;
    auto rand2 = rand;
    test = std::move(rand);
    rand = rand2;

    CHECK_EQ(rand.matrix(), test.matrix());
    EXPECT_TRUE(typeid(test) == typeid(lgmath::se2::Transformation));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test some get methods for SE(2) TransformationWithCovariance
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, SE2TransformationWithCovarianceGetMethods) {
  // Generate random transform from most basic constructor
  Eigen::Matrix<double, 2, 2> C_ab =
      lgmath::so2::vec2rot(Eigen::Matrix<double, 1, 1>::Random()(0));
  Eigen::Matrix<double, 2, 1> r_ba_ina = Eigen::Matrix<double, 2, 1>::Random();
  Eigen::Matrix<double, 3, 3> U = Eigen::Matrix<double, 3, 3>::Random();
  lgmath::se2::TransformationWithCovariance T_ba(C_ab, r_ba_ina, U);

  // Construct simple eigen matrix from random rotation and translation
  Eigen::Matrix3d test = Eigen::Matrix3d::Identity();
  test.topLeftCorner<2, 2>() = C_ab;
  test.topRightCorner<2, 1>() = r_ba_ina;

  // Test matrix()
  std::cout << "T_ba: " << T_ba.matrix() << std::endl;
  std::cout << "test: " << test << std::endl;
  EXPECT_TRUE(lgmath::common::nearEqual(T_ba.matrix(), test, 1e-6));

  // Test C_ab()
  std::cout << "T_ba: " << T_ba.C_ab() << std::endl;
  std::cout << "C_ab: " << C_ab << std::endl;
  EXPECT_TRUE(lgmath::common::nearEqual(T_ba.C_ab(), C_ab, 1e-6));

  // Test r_ba_ina()
  std::cout << "T_ba: " << T_ba.r_ba_ina() << std::endl;
  std::cout << "r_ba_ina: " << r_ba_ina << std::endl;
  EXPECT_TRUE(lgmath::common::nearEqual(T_ba.r_ba_ina(), r_ba_ina, 1e-6));

  // Test r_ab_inb()
  Eigen::Vector2d r_ab_inb_computed = T_ba.r_ab_inb();
  Eigen::Vector2d r_ab_inb_expected = -C_ab.transpose() * r_ba_ina;
  std::cout << "T_ba: " << r_ab_inb_computed << std::endl;
  std::cout << "expected: " << r_ab_inb_expected << std::endl;
  EXPECT_TRUE(lgmath::common::nearEqual(r_ab_inb_computed, r_ab_inb_expected, 1e-6));
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test math operations and flag propagation for SE(2)
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, SE2TransformationWithCovarianceOperations) {
  // Generate random transform from most basic constructor
  Eigen::Matrix<double, 2, 2> C1 =
      lgmath::so2::vec2rot(Eigen::Matrix<double, 1, 1>::Random()(0));
  Eigen::Matrix<double, 2, 1> r1 = Eigen::Matrix<double, 2, 1>::Random();
  Eigen::Matrix<double, 3, 3> U1 = Eigen::Matrix<double, 3, 3>::Random();
  lgmath::se2::TransformationWithCovariance T1(C1, r1, U1);

  Eigen::Matrix<double, 2, 2> C2 =
      lgmath::so2::vec2rot(Eigen::Matrix<double, 1, 1>::Random()(0));
  Eigen::Matrix<double, 2, 1> r2 = Eigen::Matrix<double, 2, 1>::Random();
  Eigen::Matrix<double, 3, 3> U2 = Eigen::Matrix<double, 3, 3>::Random();
  lgmath::se2::TransformationWithCovariance T2(C2, r2, U2);

  Eigen::Matrix<double, 2, 2> C3 =
      lgmath::so2::vec2rot(Eigen::Matrix<double, 1, 1>::Random()(0));
  Eigen::Matrix<double, 2, 1> r3 = Eigen::Matrix<double, 2, 1>::Random();
  lgmath::se2::TransformationWithCovariance T3(C3, r3);

  Eigen::Matrix<double, 2, 2> C4 =
      lgmath::so2::vec2rot(Eigen::Matrix<double, 1, 1>::Random()(0));
  Eigen::Matrix<double, 2, 1> r4 = Eigen::Matrix<double, 2, 1>::Random();
  lgmath::se2::Transformation T4(C4, r4);

  /// Operator: *=

  // test TWC *= TWC
  {
    lgmath::se2::TransformationWithCovariance test(T1);
    test *= T2;
    Eigen::Matrix3d tmat = T1.matrix() * T2.matrix();

    Eigen::Matrix<double, 3, 3> Ad = T1.adjoint();
    Eigen::Matrix<double, 3, 3> tmatU = U1 + Ad * U2 * Ad.transpose();

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  // test TWC *= T
  {
    lgmath::se2::TransformationWithCovariance test(T1);
    test *= T4;
    Eigen::Matrix3d tmat = T1.matrix() * T4.matrix();

    Eigen::Matrix<double, 3, 3> tmatU = U1;

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  // test T *= TWC
  {
    lgmath::se2::Transformation test(T4);
    test *= T1;
    Eigen::Matrix3d tmat = T4.matrix() * T1.matrix();

    CHECK_EQ(tmat, test.matrix());
    EXPECT_TRUE(
        typeid(test) ==
        typeid(
            lgmath::se2::Transformation));  // Make sure nothing weird happened
  }

  // test TWC *= TWC(unset)
  {
    lgmath::se2::TransformationWithCovariance test(T1);
    test *= T3;
    Eigen::Matrix3d tmat = T1.matrix() * T3.matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  // test TWC(unset) *= TWC
  {
    lgmath::se2::TransformationWithCovariance test(T3);
    test *= T1;
    Eigen::Matrix3d tmat = T3.matrix() * T1.matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  // test TWC(unset) *= TWC(unset)
  {
    lgmath::se2::TransformationWithCovariance test(T3);
    test *= T3;
    Eigen::Matrix3d tmat = T3.matrix() * T3.matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_NO_COVARIANCE(test);
  }

  // test T *= TWC(unset)
  {
    lgmath::se2::Transformation test(T4);
    test *= T3;
    Eigen::Matrix3d tmat = T4.matrix() * T3.matrix();

    CHECK_EQ(tmat, test.matrix());
    EXPECT_TRUE(
        typeid(test) ==
        typeid(
            lgmath::se2::Transformation));  // Make sure nothing weird happened
  }

  /// Operator: *

  // test TWC * TWC
  {
    lgmath::se2::TransformationWithCovariance test = T1 * T2;
    Eigen::Matrix3d tmat = T1.matrix() * T2.matrix();

    Eigen::Matrix<double, 3, 3> Ad = T1.adjoint();
    Eigen::Matrix<double, 3, 3> tmatU = U1 + Ad * U2 * Ad.transpose();

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  // test TWC * T
  {
    lgmath::se2::TransformationWithCovariance test = T1 * T4;
    Eigen::Matrix3d tmat = T1.matrix() * T4.matrix();

    Eigen::Matrix<double, 3, 3> tmatU = U1;

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  // test T * TWC
  {
    lgmath::se2::TransformationWithCovariance test = T4 * T1;
    Eigen::Matrix3d tmat = T4.matrix() * T1.matrix();

    // When multiplying T * TWC, the result should have perfect certainty initially
    // plus the uncertainty of TWC transformed by the adjoint
    Eigen::Matrix<double, 3, 3> Ad = T4.adjoint();
    Eigen::Matrix<double, 3, 3> tmatU = Ad * U1 * Ad.transpose();

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  /// Operator: /=

  // test TWC /= TWC
  {
    lgmath::se2::TransformationWithCovariance test(T1);
    test /= T2;
    Eigen::Matrix3d tmat = T1.matrix() * T2.inverse().matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
  }

  // test TWC /= T
  {
    lgmath::se2::TransformationWithCovariance test(T1);
    test /= T4;
    Eigen::Matrix3d tmat = T1.matrix() * T4.inverse().matrix();

    Eigen::Matrix<double, 3, 3> tmatU = U1;

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  /// Operator: /

  // test TWC / TWC
  {
    lgmath::se2::TransformationWithCovariance test = T1 / T2;
    Eigen::Matrix3d tmat = T1.matrix() * T2.inverse().matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
  }

  // test TWC / T
  {
    lgmath::se2::TransformationWithCovariance test = T1 / T4;
    Eigen::Matrix3d tmat = T1.matrix() * T4.inverse().matrix();

    Eigen::Matrix<double, 3, 3> tmatU = U1;

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
    CHECK_EQ_COVARIANCE(test, tmatU);
  }

  // test T / TWC
  {
    lgmath::se2::TransformationWithCovariance test = T4 / T1;
    Eigen::Matrix3d tmat = T4.matrix() * T1.inverse().matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
  }

  /// Test inverse
  {
    lgmath::se2::TransformationWithCovariance test = T1.inverse();
    Eigen::Matrix3d tmat = T1.inverse().matrix();

    CHECK_EQ(tmat, test.matrix());
    CHECK_HAS_COVARIANCE(test);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
