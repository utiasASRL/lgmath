/**
 * \file CommonMath.cpp
 * \brief Implementation file for some common math functions
 * \details defines some constants, angle-based functions, and comparison
 * functions.
 *
 * \author Sean Anderson, ASRL
 */
#include <lgmath/CommonMath.hpp>

#include <math.h>

namespace lgmath {
namespace common {

double angleMod(double radians) {
  return (double)(radians - (constants::TWO_PI *
                             rint(radians * constants::ONE_DIV_TWO_PI)));
}

double deg2rad(double degrees) {
  return (double)(degrees * constants::DEG2RAD);
}

double rad2deg(double radians) {
  return (double)(radians * constants::RAD2DEG);
}

bool nearEqual(double a, double b, double tol) { return fabs(a - b) <= tol; }

bool nearEqual(Eigen::MatrixXd A, Eigen::MatrixXd B, double tol) {
  bool near = true;
  near = near & (A.cols() == B.cols());
  near = near & (A.rows() == B.rows());
  for (int j = 0; j < A.cols(); j++) {
    for (int i = 0; i < A.rows(); i++) {
      near = near & nearEqual(A(i, j), B(i, j), tol);
    }
  }
  return near;
}

bool nearEqualAngle(double radA, double radB, double tol) {
  return nearEqual(angleMod(radA - radB), 0.0, tol);
}

bool nearEqualAxisAngle(Eigen::Matrix<double, 3, 1> aaxis1,
                        Eigen::Matrix<double, 3, 1> aaxis2, double tol) {
  bool near = true;

  // get angles
  double a1 = aaxis1.norm();
  double a2 = aaxis2.norm();

  // check if both angles are near zero
  if (fabs(a1) < tol && fabs(a2) < tol) {
    return true;
  } else {  // otherwise, compare normalized axis

    // compare each element of axis
    Eigen::Matrix<double, 3, 1> axis1 = aaxis1 / a1;
    Eigen::Matrix<double, 3, 1> axis2 = aaxis1 / a2;
    for (int i = 0; i < 3; i++) {
      near = near & nearEqual(axis1(i), axis2(i), tol);
    }

    // compare wrapped angles
    near = near & nearEqualAngle(a1, a2, tol);
    return near;
  }
}

bool nearEqualLieAlg(Eigen::Matrix<double, 6, 1> vec1,
                     Eigen::Matrix<double, 6, 1> vec2, double tol) {
  bool near = true;
  near = near & nearEqualAxisAngle(vec1.tail<3>(), vec2.tail<3>(), tol);
  near = near & nearEqual(vec1.head<3>(), vec2.head<3>(), tol);
  return near;
}

}  // namespace common
}  // namespace lgmath
