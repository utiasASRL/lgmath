//////////////////////////////////////////////////////////////////////////////////////////////
/// \file Operations.cpp
/// \brief Implementation file for the SE3 Lie Group math functions.
/// \details These namespace functions provide implementations of the special Euclidean (SE)
///          Lie group functions that we commonly use in robotics.
///
/// \author Sean Anderson
//////////////////////////////////////////////////////////////////////////////////////////////

#include <lgmath/se3/Operations.hpp>

#include <lgmath/so3/Operations.hpp>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <stdio.h>

namespace lgmath {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief builds the 4x4 "skew symmetric matrix" (see eq. 4 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix4d hat(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis) {
  Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
  mat.topLeftCorner<3,3>() = so3::hat(aaxis);
  mat.topRightCorner<3,1>() = rho;
  return mat;
}
Eigen::Matrix4d hat(const Eigen::Matrix<double,6,1>& vec) {
  return hat(vec.head<3>(), vec.tail<3>());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief builds the 6x6 curly hat matrix (see eq. 12 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,6> curlyhat(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis) {
  Eigen::Matrix<double,6,6> mat = Eigen::Matrix<double,6,6>::Zero();
  mat.topLeftCorner<3,3>() = mat.bottomRightCorner<3,3>() = so3::hat(aaxis);
  mat.topRightCorner<3,3>() = so3::hat(rho);
  return mat;
}
Eigen::Matrix<double,6,6> curlyhat(const Eigen::Matrix<double,6,1> & vec) {
  return curlyhat(vec.head<3>(), vec.tail<3>());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief turns a 4x1 homogeneous point into the 4x6 matrix (see eq. 72 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,4,6> point2fs(const Eigen::Vector3d& p, double scale) {
  Eigen::Matrix<double,4,6> mat = Eigen::Matrix<double,4,6>::Zero();
  mat.topLeftCorner<3,3>() = scale*Eigen::Matrix3d::Identity();
  mat.topRightCorner<3,3>() = -so3::hat(p);
  return mat;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief turns a 4x1 homogeneous point into the 6x4 matrix (see eq. 72 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,4> point2sf(const Eigen::Vector3d& p, double scale) {
  Eigen::Matrix<double,6,4> mat = Eigen::Matrix<double,6,4>::Zero();
  mat.bottomLeftCorner<3,3>() = -so3::hat(p);
  mat.topRightCorner<3,1>() = p;
  return mat;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief builds a transformation matrix using the exponential map (see Barfoot-TRO-2014 Appendix B1)
//////////////////////////////////////////////////////////////////////////////////////////////
void vec2tran_analytical(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis, Eigen::Matrix3d* outRot, Eigen::Vector3d* outTrans) {

  // Check outputs
  CHECK_NOTNULL(outRot);
  CHECK_NOTNULL(outTrans);

  if(aaxis.norm() < 1e-12) { // If angle is very small, rotation is Identity
    *outRot = Eigen::Matrix3d::Identity();
    *outTrans = rho;
  } else {
    Eigen::Matrix3d jac;
    so3::vec2rot(aaxis, outRot, &jac);
    *outTrans = jac*rho;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief builds a transformation matrix using the first N terms of the infinite series (see eq. 96 in Barfoot-TRO-2014)
///        Not efficient, but mostly used to test the analytical method.
//////////////////////////////////////////////////////////////////////////////////////////////
void vec2tran_numerical(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis, Eigen::Matrix3d* outRot, Eigen::Vector3d* outTrans, unsigned int numTerms) {

  // Check outputs
  CHECK_NOTNULL(outRot);
  CHECK_NOTNULL(outTrans);

  // Init 4x4
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

  // Incremental variables
  Eigen::Matrix<double,6,1> vec; vec << rho, aaxis;
  Eigen::Matrix4d x_small = se3::hat(vec);
  Eigen::Matrix4d x_small_n = Eigen::Matrix4d::Identity();

  // Loop over sum up to the specified numTerms
  for (unsigned int n = 1; n <= numTerms; n++) {
    x_small_n = x_small_n*x_small/double(n);
    T += x_small_n;
  }

  // Fill output
  *outRot = T.topLeftCorner<3,3>();
  *outTrans = T.topRightCorner<3,1>();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief builds components of the transformation matrix, analytical or numeric is determined by numTerms
//////////////////////////////////////////////////////////////////////////////////////////////
void vec2tran(const Eigen::Matrix<double,6,1>& vec, Eigen::Matrix3d* outRot, Eigen::Vector3d* outTrans, unsigned int numTerms)
{
  if (numTerms == 0) {  // Analytical solution
    vec2tran_analytical(vec.head<3>(), vec.tail<3>(), outRot, outTrans);
  } else {              // Numerical Solution: Good for testing the analytical solution
    vec2tran_numerical(vec.head<3>(), vec.tail<3>(), outRot, outTrans, numTerms);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief builds a 4x4 transformation matrix, analytical or numeric is determined by numTerms
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix4d vec2tran(const Eigen::Matrix<double,6,1>& vec, unsigned int numTerms) {

  // Get rotation and translation
  Eigen::Matrix3d rot;
  Eigen::Vector3d tran;
  vec2tran(vec, &rot, &tran, numTerms);

  // Fill output
  Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
  mat.topLeftCorner<3,3>() = rot;
  mat.topRightCorner<3,1>() = tran;
  return mat;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief compute the matrix log of a transformation matrix (see Barfoot-TRO-2014 Appendix B2)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,1> tran2vec(const Eigen::Matrix3d& rot, const Eigen::Vector3d& trans) {
  Eigen::Matrix<double,6,1> xi;
  Eigen::Vector3d angi = so3::rot2vec(rot);
  Eigen::Vector3d rho = so3::vec2jacinv(angi)*trans;
  xi << rho, angi;
  return xi;
}
Eigen::Matrix<double,6,1> tran2vec(const Eigen::Matrix4d& mat) {
  return tran2vec(mat.topLeftCorner<3,3>(), mat.topRightCorner<3,1>());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief builds the 6x6 adjoint transformation matrix from a 4x4 one (see eq. 101 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,6> tranAd(const Eigen::Matrix3d& rot, const Eigen::Vector3d& trans) {
  Eigen::Matrix<double,6,6> adT = Eigen::Matrix<double,6,6>::Zero();
  adT.topLeftCorner<3,3>() = adT.bottomRightCorner<3,3>() = rot;
  adT.topRightCorner<3,3>() = so3::hat(trans)*rot;
  return adT;
}
Eigen::Matrix<double,6,6> tranAd(const Eigen::Matrix4d& mat) {
  return tranAd(mat.topLeftCorner<3,3>(), mat.topRightCorner<3,1>());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief construction of the 3x3 "Q" matrix, used in the 6x6 Jacobian of SE(3) (see eq. 102 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix3d vec2Q(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis) {
  const double ang = aaxis.norm();
  const double ang2 = ang*ang;
  const double ang3 = ang2*ang;
  const double ang4 = ang3*ang;
  const double ang5 = ang4*ang;
  const double cang = cos(ang);
  const double sang = sin(ang);
  const double m2 = (ang - sang)/ang3;
  const double m3 = (1.0 - 0.5*ang2 - cang)/ang4;
  const double m4 = 0.5*(m3 - 3*(ang - sang - ang3/6)/ang5);
  Eigen::Matrix3d rx = so3::hat(rho);
  Eigen::Matrix3d px = so3::hat(aaxis);
  Eigen::Matrix3d pxrx = px*rx;
  Eigen::Matrix3d rxpx = rx*px;
  Eigen::Matrix3d pxrxpx = pxrx*px;
  // TODO: Look into Eigen if there exists anything for optimizing operations. Example below. kcu
  //return 0.5 * rx + m2 * (pxrx + rxpx + pxrxpx) - m3 * (px*pxrx + rxpx*px - 3*pxrxpx).eval() - m4 * (pxrxpx*px + px*pxrxpx).eval();
  return 0.5 * rx + m2 * (pxrx + rxpx + pxrxpx) - m3 * (px*pxrx + rxpx*px - 3*pxrxpx) - m4 * (pxrxpx*px + px*pxrxpx);
}
Eigen::Matrix3d vec2Q(const Eigen::Matrix<double,6,1>& vec) {
  return vec2Q(vec.head<3>(), vec.tail<3>());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief construction of the 6x6 Jacobian of SE(3) (see eq. 100 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,6> vec2jac(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis) {

  Eigen::Matrix<double,6,6> jac = Eigen::Matrix<double,6,6>::Zero();
  if(aaxis.norm() < 1e-12) {  // If angle is very small, so3 jacobian is Identity
    jac.topLeftCorner<3,3>() = jac.bottomRightCorner<3,3>() = Eigen::Matrix3d::Identity();
    jac.topRightCorner<3,3>() = 0.5*so3::hat(rho);
  } else {                    // general scenario
    jac.topLeftCorner<3,3>() = jac.bottomRightCorner<3,3>() = so3::vec2jac(aaxis);
    jac.topRightCorner<3,3>() = se3::vec2Q(rho, aaxis);
  }
  return jac;
}

Eigen::Matrix<double,6,6> vec2jac(const Eigen::Matrix<double,6,1>& vec, unsigned int numTerms) {

  if (numTerms == 0) {  // Analytical solution
    return vec2jac(vec.head<3>(), vec.tail<3>());
  } else {              // Numerical Solution: Good for testing the analytical solution
    Eigen::Matrix<double,6,6> J = Eigen::Matrix<double,6,6>::Identity();

    // Incremental variables
    Eigen::Matrix<double,6,6> x_small = se3::curlyhat(vec);
    Eigen::Matrix<double,6,6> x_small_n = Eigen::Matrix<double,6,6>::Identity();

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n*x_small/double(n+1);
      J += x_small_n;
    }
    return J;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief construction of the 6x6 inverse Jacobian of SE(3) (see eq. 103 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,6> vec2jacinv(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis) {

  Eigen::Matrix<double,6,6> seJacInv = Eigen::Matrix<double,6,6>::Zero();
  if(aaxis.norm() < 1e-12) {  // If angle is very small, so3 jacobian is Identity
    seJacInv.topLeftCorner<3,3>() = seJacInv.bottomRightCorner<3,3>() = Eigen::Matrix3d::Identity();
    seJacInv.topRightCorner<3,3>() = -0.5*so3::hat(rho);
  } else {                    // general scenario
    Eigen::Matrix3d soJacInv = so3::vec2jacinv(aaxis);
    seJacInv.topLeftCorner<3,3>() = seJacInv.bottomRightCorner<3,3>() = soJacInv;
    seJacInv.topRightCorner<3,3>() = -soJacInv*se3::vec2Q(rho, aaxis)*soJacInv;
  }
  return seJacInv;
}

Eigen::Matrix<double,6,6> vec2jacinv(const Eigen::Matrix<double,6,1>& vec, unsigned int numTerms) {
  if (numTerms == 0) {  // Analytical solution
    return vec2jacinv(vec.head<3>(), vec.tail<3>());
  } else {              // Numerical Solution: Good for testing the analytical solution
    CHECK(numTerms <= 20) << "Terms higher than 20 for vec2jacinv are not supported";
    Eigen::Matrix<double,6,6> J = Eigen::Matrix<double,6,6>::Identity();

    // Incremental variables
    Eigen::Matrix<double,6,6> x_small = se3::curlyhat(vec);
    Eigen::Matrix<double,6,6> x_small_n = Eigen::Matrix<double,6,6>::Identity();

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

} // se3
} // lgmath
