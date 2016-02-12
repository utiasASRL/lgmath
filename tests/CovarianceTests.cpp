//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Unit tests for the transformation of points (with and without covariance)
///
/// \author Kirk MacTavish
//////////////////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <iostream>
#include <iomanip>
#include <ios>
#include <typeinfo>

#include <Eigen/Dense>
#include <lgmath.hpp>
#include <lgmath/CommonMath.hpp>

#include "catch.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////
///
/// UNIT TESTS OF POINTS WITH AND WITHOUT COVARIANCE
///
/////////////////////////////////////////////////////////////////////////////////////////////

using namespace lgmath;


/////////////////////////////////////////////////////////////////////////////////////////////
// HELPER CONSTANTS
/////////////////////////////////////////////////////////////////////////////////////////////

static const so3::RotationMatrix C_z180 =
    so3::Rotation(so3::AxisAngle(0.,0.,constants::PI)).matrix();
static const so3::RotationMatrix C_z90 =
    so3::Rotation(so3::AxisAngle(0.,0.,constants::PI_DIV_TWO)).matrix();

/////////////////////////////////////////////////////////////////////////////////////////////
// HELPER FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////////

template<typename Derived>
void check_approx(const Eigen::DenseBase<Derived> & a,
                  const Eigen::DenseBase<Derived> & b) {
  INFO(a << "\n==\n" << b);
  CHECK(a.isApprox(b));
}

/////////////////////////////////////////////////////////////////////////////////////////////
// MAIN TESTS
/////////////////////////////////////////////////////////////////////////////////////////////

SCENARIO("Point covariance can be transformed","[points]") {
  r3::CovarianceMatrix cov_a;
  cov_a.diagonal() << 1., 2., 3.;
  r3::HPoint p_a = (r3::Point()<< 2., 3., 4.).finished().homogeneous();

  GIVEN("A 180 degree transform") {
    se3::Transformation T_ba(C_z180, se3::TranslationVector::Zero());
    WHEN("We transform the covariance") {
      auto cov_b = r3::transformCovariance<r3::COVARIANCE_NOT_REQUIRED>(T_ba, cov_a);
      THEN("The covariance should be unchanged") {
        check_approx(cov_a, cov_b);
      } // THEN
    } // WHEN
  } // GIVEN

  GIVEN("A Z 90 degree transform") {
    se3::Transformation T_ba(C_z90, se3::TranslationVector::Zero());
    WHEN("We transform the covariance") {
      auto cov_b = r3::transformCovariance<r3::COVARIANCE_NOT_REQUIRED>(T_ba, cov_a);
      THEN("The covariance should have x and y swapped") {
        cov_a.row(0).swap(cov_a.row(1));
        cov_a.col(0).swap(cov_a.col(1));
        check_approx(cov_a, cov_b);
      } // THEN
    } // WHEN
  } // GIVEN

  GIVEN("Uncertain translation") {
    se3::LieAlgebraCovariance S_ba = se3::LieAlgebraCovariance::Zero();
    S_ba.topLeftCorner<3,3>() = cov_a;
    se3::TransformationWithCovariance T_ba (
          so3::RotationMatrix::Identity(),
          se3::TranslationVector::Zero(),
          S_ba);
    WHEN("We translate the point and covariance") {
      auto p_b = T_ba * p_a;
      r3::CovarianceMatrix cov_b;
      REQUIRE_NOTHROW(cov_b = r3::transformCovariance(T_ba, cov_a, p_b));
      THEN("The covariance should be additive") {
        r3::CovarianceMatrix cov_b_expect = cov_a*2.;
        check_approx(cov_b, cov_b_expect);
      } //THEN
    } // WHEN
  } // GIVEN

  GIVEN("Uninitialized uncertain transform") {
    se3::TransformationWithCovariance T_ba;
    r3::CovarianceMatrix cov_b;
    auto p_b = T_ba*p_a;
    WHEN("We transform the point without ignoring the 'covariance set' flag") {
      THEN("It should throw") {
        CHECK_THROWS(cov_b = r3::transformCovariance(T_ba, cov_a, p_b));
      } // THEN
    } // WHEN

    WHEN("We transform the point but ignore the 'covariance set' flag") {
      THEN("It shouldn't throw, and the covariance should be unchanged") {
        CHECK_NOTHROW(cov_b = r3::transformCovariance<r3::COVARIANCE_NOT_REQUIRED>(T_ba, cov_a, p_b));
        check_approx(cov_a, cov_b);
      } // THEN
    } // WHEN
  } // GIVEN

  GIVEN("Uncertain rotation") {
    se3::LieAlgebraCovariance S_ba; S_ba.setZero();
    S_ba(5,5) = 1;
    se3::TransformationWithCovariance T_ba(
          se3::TransformationMatrix(se3::TransformationMatrix::Identity()),
          S_ba);
    WHEN("We transform the point and covariance") {
      auto p_b = T_ba*p_a;
      THEN("The covariance should be unchanged in Z, and larger in X and Y") {
        auto cov_b = r3::transformCovariance(T_ba, cov_a, p_b);
        CHECK(cov_b(0,0) > cov_a(0,0)+1e-3);
        CHECK(cov_b(1,1) > cov_a(1,1)+1e-3);
        CHECK(cov_b(2,2) == Approx(cov_a(2,2)));
      } // THEN
    } // WHEN
  } // GIVEN

} // SCENARIO

SCENARIO("A point to be transformed","[points]") {
  GIVEN("A point :)") {
    r3::HPoint x; x << 1., 2., 3., 1.;

    WHEN("We transform it") {
      se3::Transformation T((se3::LieAlgebra()<<1.,0,0,constants::PI,0,0).finished());
      THEN("It should be in the right spot") {
        r3::HPoint x_tf; x_tf << 2., -2., -3., 1.;
        check_approx(T*x, x_tf);
      } // THEN
    } // WHEN
  } // GIVEN

  GIVEN("A point with covariance") {
    //point3::HPointWithCovariance x;
  }
} // SCENARIO
