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
/// \brief Builds the 4x4 "skew symmetric matrix"
///
/// The hat (^) operator, builds the 4x4 skew symmetric matrix from the 3x1 axis angle
/// vector and 3x1 translation vector.
///
/// hat(rho, aaxis) = [aaxis^ rho] = [0.0  -a3   a2  rho1]
///                   [  0^T    0]   [ a3  0.0  -a1  rho2]
///                                  [-a2   a1  0.0  rho3]
///                                  [0.0  0.0  0.0   0.0]
///
/// See eq. 4 in Barfoot-TRO-2014 for more information.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix4d hat(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis) {
  Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
  mat.topLeftCorner<3,3>() = so3::hat(aaxis);
  mat.topRightCorner<3,1>() = rho;
  return mat;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Builds the 4x4 "skew symmetric matrix"
///
/// The hat (^) operator, builds the 4x4 skew symmetric matrix from
/// the 6x1 se3 algebra vector, xi:
///
/// xi^ = [rho  ] = [aaxis^ rho] = [0.0  -a3   a2  rho1]
///       [aaxis]   [  0^T    0]   [ a3  0.0  -a1  rho2]
///                                [-a2   a1  0.0  rho3]
///                                [0.0  0.0  0.0   0.0]
///
/// See eq. 4 in Barfoot-TRO-2014 for more information.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix4d hat(const Eigen::Matrix<double,6,1>& xi) {
  return hat(xi.head<3>(), xi.tail<3>());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Builds the 6x6 "curly hat" matrix (related to the skew symmetric matrix)
///
/// The curly hat operator builds the 6x6 skew symmetric matrix from the 3x1 axis angle
/// vector and 3x1 translation vector.
///
/// curlyhat(rho, aaxis) = [aaxis^   rho^]
///                        [     0 aaxis^]
///
/// See eq. 12 in Barfoot-TRO-2014 for more information.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,6> curlyhat(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis) {
  Eigen::Matrix<double,6,6> mat = Eigen::Matrix<double,6,6>::Zero();
  mat.topLeftCorner<3,3>() = mat.bottomRightCorner<3,3>() = so3::hat(aaxis);
  mat.topRightCorner<3,3>() = so3::hat(rho);
  return mat;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Builds the 6x6 "curly hat" matrix (related to the skew symmetric matrix)
///
/// The curly hat operator builds the 6x6 skew symmetric matrix
/// from the 6x1 se3 algebra vector, xi:
///
/// curlyhat(xi) = curlyhat([rho  ]) = [aaxis^   rho^]
///                        ([aaxis])   [     0 aaxis^]
///
/// See eq. 12 in Barfoot-TRO-2014 for more information.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,6> curlyhat(const Eigen::Matrix<double,6,1> & xi) {
  return curlyhat(xi.head<3>(), xi.tail<3>());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Turns a homogeneous point into a special 4x6 matrix
///
/// See eq. 72 in Barfoot-TRO-2014 for more information.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,4,6> point2fs(const Eigen::Vector3d& p, double scale) {
  Eigen::Matrix<double,4,6> mat = Eigen::Matrix<double,4,6>::Zero();
  mat.topLeftCorner<3,3>() = scale*Eigen::Matrix3d::Identity();
  mat.topRightCorner<3,3>() = -so3::hat(p);
  return mat;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Turns a homogeneous point into a special 6x4 matrix
///
/// See eq. 72 in Barfoot-TRO-2014 for more information.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,4> point2sf(const Eigen::Vector3d& p, double scale) {
  Eigen::Matrix<double,6,4> mat = Eigen::Matrix<double,6,4>::Zero();
  mat.bottomLeftCorner<3,3>() = -so3::hat(p);
  mat.topRightCorner<3,1>() = p;
  return mat;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Builds a transformation matrix using the analytical exponential map
///
/// This function builds a transformation matrix, T_ab, using the analytical exponential map,
/// from the se3 algebra vector, xi_ba,
///
///   T_ab = exp(xi_ba^) = [ C_ab r_ba_ina],   xi_ba = [aaxis_ba]
///                        [  0^T        1]            [  rho_ba]
///
/// where C_ab is a 3x3 rotation matrix from 'b' to 'a', r_ba_ina is the 3x1 translation
/// vector from 'a' to 'b' expressed in frame 'a', aaxis_ba is a 3x1 axis-angle vector,
/// the magnitude of the angle of rotation can be recovered by finding the norm of the vector,
/// and the axis of rotation is the unit-length vector that arises from normalization.
/// Note that the angle around the axis, aaxis_ba, is a right-hand-rule (counter-clockwise
/// positive) angle from 'a' to 'b'.
///
/// The parameter, rho_ba, is a special translation-like parameter related to 'twist' theory.
/// It is most inuitively described as being like a constant linear velocity (expressed in
/// the smoothly-moving frame) for a fixed duration; for example, consider the curve of a
/// car driving 'x' meters while turning at a rate of 'y' rad/s.
///
/// For more information see Barfoot-TRO-2014 Appendix B1.
///
/// Alternatively, we that note that
///
///   T_ba = exp(-xi_ba^) = exp(xi_ab^).
///
/// Both the analytical (numTerms = 0) or the numerical (numTerms > 0) may be evaluated.
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
/// \brief Builds a transformation matrix using the first N terms of the infinite series
///
/// Builds a transformation matrix numerically using the infinite series evalation
/// of the exponential map.
///
/// For more information see eq. 96 in Barfoot-TRO-2014
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
/// \brief Builds the 3x3 rotation and 3x1 translation using the exponential map, the
///        default parameters (numTerms = 0) use the analytical solution.
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
/// \brief Builds a 4x4 transformation matrix using the exponential map, the
///        default parameters (numTerms = 0) use the analytical solution.
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
/// \brief Compute the matrix log of a transformation matrix (from the rotation and trans)
///
/// Compute the inverse of the exponential map (the logarithmic map). This lets us go from
/// a the 3x3 rotation and 3x1 translation vector back to a 6x1 se3 algebra vector (composed
/// of a 3x1 axis-angle vector and 3x1 twist-translation vector). In some cases, when the
/// rotation in the transformation matrix is 'numerically off', this involves some
/// 'projection' back to SE(3).
///
///   xi_ba = ln(T_ab)
///
/// where xi_ba is the 6x1 se3 algebra vector. Alternatively, we that note that
///
///   xi_ab = -xi_ba = ln(T_ba) = ln(T_ab^{-1})
///
/// See Barfoot-TRO-2014 Appendix B2 for more information.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,1> tran2vec(const Eigen::Matrix3d& rot, const Eigen::Vector3d& trans) {
  Eigen::Matrix<double,6,1> xi;
  Eigen::Vector3d angi = so3::rot2vec(rot);
  Eigen::Vector3d rho = so3::vec2jacinv(angi)*trans;
  xi << rho, angi;
  return xi;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Compute the matrix log of a transformation matrix
///
/// Compute the inverse of the exponential map (the logarithmic map). This lets us go from
/// a 4x4 transformation matrix back to a 6x1 se3 algebra vector (composed of a 3x1 axis-angle
/// vector and 3x1 twist-translation vector). In some cases, when the rotation in the
/// transformation matrix is 'numerically off', this involves some 'projection' back to SE(3).
///
///   xi_ba = ln(T_ab)
///
/// where xi_ba is the 6x1 se3 algebra vector. Alternatively, we that note that
///
///   xi_ab = -xi_ba = ln(T_ba) = ln(T_ab^{-1})
///
/// See Barfoot-TRO-2014 Appendix B2 for more information.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,1> tran2vec(const Eigen::Matrix4d& mat) {
  return tran2vec(mat.topLeftCorner<3,3>(), mat.topRightCorner<3,1>());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Builds the 6x6 adjoint transformation matrix from the 3x3 rotation matrix and 3x1
///        translation vector.
///
/// Builds the 6x6 adjoint transformation matrix from the 3x3 rotation matrix and 3x1
///        translation vector.
///
///  Adjoint(T_ab) = Adjoint([C_ab r_ba_ina]) = [C_ab r_ba_ina^*C_ab] = exp(curlyhat(xi_ba))
///                         ([ 0^T        1])   [   0           C_ab]
///
/// See eq. 101 in Barfoot-TRO-2014 for more information.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,6> tranAd(const Eigen::Matrix3d& rot, const Eigen::Vector3d& trans) {
  Eigen::Matrix<double,6,6> adT = Eigen::Matrix<double,6,6>::Zero();
  adT.topLeftCorner<3,3>() = adT.bottomRightCorner<3,3>() = rot;
  adT.topRightCorner<3,3>() = so3::hat(trans)*rot;
  return adT;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Builds the 6x6 adjoint transformation matrix from a 4x4 one
///
/// Builds the 6x6 adjoint transformation matrix from a 4x4 transformation matrix
///
///  Adjoint(T_ab) = Adjoint([C_ab r_ba_ina]) = [C_ab r_ba_ina^*C_ab] = exp(curlyhat(xi_ba))
///                         ([ 0^T        1])   [   0           C_ab]
///
/// See eq. 101 in Barfoot-TRO-2014 for more information.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,6> tranAd(const Eigen::Matrix4d& mat) {
  return tranAd(mat.topLeftCorner<3,3>(), mat.topRightCorner<3,1>());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Construction of the 3x3 "Q" matrix, used in the 6x6 Jacobian of SE(3)
///
/// See eq. 102 in Barfoot-TRO-2014 for more information
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

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Construction of the 3x3 "Q" matrix, used in the 6x6 Jacobian of SE(3)
///
/// See eq. 102 in Barfoot-TRO-2014 for more information
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix3d vec2Q(const Eigen::Matrix<double,6,1>& vec) {
  return vec2Q(vec.head<3>(), vec.tail<3>());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Builds the 6x6 Jacobian matrix of SE(3) using the analytical expression
///
/// Build the 6x6 left Jacobian of SE(3).
///
/// For the sake of a notation, we assign subscripts consistence with the transformation,
///
///   J_ab = J(xi_ba)
///
/// Where applicable, we also note that
///
///   J(xi_ba) = Adjoint(exp(xi_ba^)) * J(-xi_ba),
///
/// and
///
///   Adjoint(exp(xi_ba^)) = identity + curlyhat(xi_ba) * J(xi_ba).
///
/// For more information see eq. 100 in Barfoot-TRO-2014.
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

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Builds the 6x6 Jacobian matrix of SE(3) from the se(3) algebra; note that the
///        default parameter (numTerms = 0) will call the analytical solution, but the
///        numerical solution can also be evaluating to some number of terms.
///
/// For more information see eq. 100 in Barfoot-TRO-2014.
//////////////////////////////////////////////////////////////////////////////////////////////
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
/// \brief Builds the 6x6 inverse Jacobian matrix of SE(3) using the analytical expression
///
/// Build the 6x6 inverse left Jacobian of SE(3).
///
/// For the sake of a notation, we assign subscripts consistence with the transformation,
///
///   J_ab_inverse = J(xi_ba)^{-1},
///
/// Please note that J_ab_inverse is not equivalent to J_ba:
///
///   J(xi_ba)^{-1} != J(-xi_ba)
///
/// For more information see eq. 103 in Barfoot-TRO-2014.
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

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Builds the 6x6 inverse Jacobian matrix of SE(3) from the se(3) algebra; note that
///        the default parameter (numTerms = 0) will call the analytical solution, but the
///        numerical solution can also be evaluating to some number of terms.
///
/// For more information see eq. 103 in Barfoot-TRO-2014.
//////////////////////////////////////////////////////////////////////////////////////////////
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
