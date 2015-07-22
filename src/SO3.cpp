//////////////////////////////////////////////////////////////////////////////////////////////
/// @file SO3.cpp
/// @brief Implementation file for the SO3 Lie Group math functions.
/// @details These namespace functions provide implementations of the special orthogonal (SO)
///          Lie group functions that we commonly use in robotics.
///
/// @author Sean Anderson
//////////////////////////////////////////////////////////////////////////////////////////////

#include <lgmath/SO3.hpp>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <stdio.h>

namespace lgmath {
namespace so3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// @brief builds the 3x3 skew symmetric matrix (see eq. 5 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,3,3> hat(const Eigen::Matrix<double,3,1>& vec) {
  Eigen::Matrix<double,3,3> mat;
  mat <<    0.0,  -vec[2],   vec[1],
         vec[2],      0.0,  -vec[0],
        -vec[1],   vec[0],      0.0;
  return mat;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// @brief builds a rotation matrix using the exponential map (see eq. 97 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,3,3> vec2rot(const Eigen::Matrix<double,3,1>& aaxis, unsigned int numTerms) {
  const double a = aaxis.norm(); // Get angle
  if(a < 1e-12) { // If angle is very small, return Identity
    return Eigen::Matrix<double,3,3>::Identity();
  }

  if (numTerms == 0) { // Analytical solution

    Eigen::Matrix<double,3,1> axis = aaxis/a;
    const double sa = sin(a);
    const double ca = cos(a);
    return ca*Eigen::Matrix<double,3,3>::Identity() + (1.0 - ca)*axis*axis.transpose() + sa*so3::hat(axis);

  } else { // Numerical Solution: Good for testing the analytical solution

    Eigen::Matrix<double,3,3> C = Eigen::Matrix<double,3,3>::Identity();

    // Incremental variables
    Eigen::Matrix<double,3,3> x_small = so3::hat(aaxis);
    Eigen::Matrix<double,3,3> x_small_n = Eigen::Matrix<double,3,3>::Identity();

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n*x_small/double(n);
      C += x_small_n;
    }
    return C;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// @brief efficiently builds a rotation matrix when the Jacobian is also needed using
///        the identity rot(v) = eye(3) + hat(v)*jac(v)
//////////////////////////////////////////////////////////////////////////////////////////////
void vec2rot(const Eigen::Matrix<double,3,1>& aaxis, Eigen::Matrix<double,3,3>* outRot, Eigen::Matrix<double,3,3>* outJac) {
  CHECK_NOTNULL(outRot);
  CHECK_NOTNULL(outJac);
  *outJac = so3::vec2jac(aaxis);
  *outRot = Eigen::Matrix<double,3,3>::Identity() + so3::hat(aaxis)*(*outJac);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// @brief compute the matrix log of a rotation matrix (see Barfoot-TRO-2014 Appendix B2)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,3,1> rot2vec(const Eigen::Matrix<double,3,3>& mat) {

  const double a = acos(0.5*(mat.trace()-1.0)); // Get angle
  const double sa = sin(a);

  if (fabs(sa) > 1e-9) { // General case, angle is NOT near 0, pi, or 2*pi
    Eigen::Matrix<double,3,1> axis;
    axis << mat(2,1) - mat(1,2),
            mat(0,2) - mat(2,0),
            mat(1,0) - mat(0,1);
    return (0.5*a/sa)*axis;
  } else if (fabs(a) > 1e-9) { // Angle is near pi or 2*pi
    // Note with this method we do not know the sign of 'a', however since we know a is close to
    // pi or 2*pi, the sign is unimportant..

    // Find the eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,3,3> > eigenSolver(mat);

    // Try each eigenvalue
    for (int i = 0; i < 3; i++) {
      // Check if eigen value is near +1.0
      if ( fabs(eigenSolver.eigenvalues()[i] - 1.0) < 1e-6 ) {
        // Get corresponding angle-axis
        Eigen::Matrix<double,3,1> aaxis = a*eigenSolver.eigenvectors().col(i);
        return aaxis;
      }
    }
    CHECK(false) << "rot2vec: angle is near pi or 2*pi, but none of the eigenvalues were near 1...";
  } else { // Angle is near zero
    return Eigen::Matrix<double,3,1>::Zero();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// @brief builds the 3x3 jacobian matrix of SO(3) (see eq. 98 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,3,3> vec2jac(const Eigen::Matrix<double,3,1>& aaxis, unsigned int numTerms) {
  const double a = aaxis.norm(); // Get angle
  if(a < 1e-12) {
    return Eigen::Matrix<double,3,3>::Identity(); // If angle is very small, return Identity
  }

  if (numTerms == 0) { // Analytical solution
    Eigen::Matrix<double,3,1> axis = aaxis/a;
    const double sa = sin(a)/a;
    const double ca = (1.0-cos(a))/a;
    return sa*Eigen::Matrix<double,3,3>::Identity() + (1.0 - sa)*axis*axis.transpose() + ca*so3::hat(axis);
  } else { // Numerical Solution: Good for testing the analytical solution
    Eigen::Matrix<double,3,3> J = Eigen::Matrix<double,3,3>::Identity();

    // Incremental variables
    Eigen::Matrix<double,3,3> x_small = so3::hat(aaxis);
    Eigen::Matrix<double,3,3> x_small_n = Eigen::Matrix<double,3,3>::Identity();

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n*x_small/double(n+1);
      J += x_small_n;
    }
    return J;
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// @brief builds the 3x3 inverse jacobian matrix of SO(3) (see eq. 99 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,3,3> vec2jacinv(const Eigen::Matrix<double,3,1>& aaxis, unsigned int numTerms) {
  const double a = aaxis.norm(); // Get angle
  if(a < 1e-12) {
    return Eigen::Matrix<double,3,3>::Identity(); // If angle is very small, return Identity
  }

  if (numTerms == 0) { // Analytical solution
    Eigen::Matrix<double,3,1> axis = aaxis/a;
    const double a2 = 0.5*a;
    const double a2cota2 = a2/tan(a2);
    return a2cota2*Eigen::Matrix<double,3,3>::Identity() + (1.0 - a2cota2)*axis*axis.transpose() - a2*so3::hat(axis);
  } else { // Numerical Solution: Good for testing the analytical solution
    CHECK(numTerms <= 20) << "Terms higher than 20 for vec2jacinv are not supported";
    Eigen::Matrix<double,3,3> J = Eigen::Matrix<double,3,3>::Identity();

    // Incremental variables
    Eigen::Matrix<double,3,3> x_small = so3::hat(aaxis);
    Eigen::Matrix<double,3,3> x_small_n = Eigen::Matrix<double,3,3>::Identity();

    // Boost has a bernoulli package... but we shouldn't need more than 20
    Eigen::Matrix<double,21,1> bernoulli;
    bernoulli << 1.0, -0.5, 1.0/6.0, 0.0, -1.0/30.0, 0.0, 1.0/42.0, 0.0, -1.0/30.0,
                 0.0, 5.0/66.0, 0.0, -691.0/2730.0, 0.0, 7.0/6.0, 0.0, -3617.0/510.0,
                 0.0, 43867.0/798.0, 0.0, -174611.0/330.0;

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n*x_small/double(n);
      J += bernoulli(n)*x_small_n;
    }
    return J;
  }
}

} // so3
} // lgmath
