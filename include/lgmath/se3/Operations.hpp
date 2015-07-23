//////////////////////////////////////////////////////////////////////////////////////////////
/// \file Operations.hpp
/// \brief Header file for the SE3 Lie Group math functions.
/// \details These namespace functions provide implementations of the special Euclidean (SE)
///          Lie group functions that we commonly use in robotics.
///
/// \author Sean Anderson
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef LGM_SE3_PUBLIC_HPP
#define LGM_SE3_PUBLIC_HPP

#include <Eigen/Core>

/////////////////////////////////////////////////////////////////////////////////////////////
/// Lie Group Math - Special Euclidean Group
/////////////////////////////////////////////////////////////////////////////////////////////
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
Eigen::Matrix4d hat(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis);

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
Eigen::Matrix4d hat(const Eigen::Matrix<double,6,1>& xi);

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
Eigen::Matrix<double,6,6> curlyhat(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis);

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
Eigen::Matrix<double,6,6> curlyhat(const Eigen::Matrix<double,6,1>& vec);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Turns a homogeneous point into a special 4x6 matrix
///
/// See eq. 72 in Barfoot-TRO-2014 for more information.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,4,6> point2fs(const Eigen::Vector3d& p, double scale = 1);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Turns a homogeneous point into a special 6x4 matrix
///
/// See eq. 72 in Barfoot-TRO-2014 for more information.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,4> point2sf(const Eigen::Vector3d& p, double scale = 1);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Builds a transformation matrix using the analytical exponential map
//////////////////////////////////////////////////////////////////////////////////////////////
void vec2tran_analytical(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis,
                         Eigen::Matrix3d* outRot, Eigen::Vector3d* outTrans);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief builds a transformation matrix using the first N terms of the infinite series
///
/// For more information see eq. 96 in Barfoot-TRO-2014
//////////////////////////////////////////////////////////////////////////////////////////////
void vec2tran_numerical(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis,
                        Eigen::Matrix3d* outRot, Eigen::Vector3d* outTrans,
                        unsigned int numTerms = 0);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Builds the 3x3 rotation and 3x1 translation using the exponential map
//////////////////////////////////////////////////////////////////////////////////////////////
void vec2tran(const Eigen::Matrix<double,6,1>& vec, Eigen::Matrix3d* outRot,
              Eigen::Vector3d* outTrans, unsigned int numTerms = 0);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Builds a transformation matrix using the exponential map
///
/// This function builds a transformation matrix, T_ab, using the exponential map (from the
/// se3 algebra vector, xi_ba,
///
///   T_ab = exp(xi_ba^), xi_ba = [aaxis_ba]
///                               [  rho_ba]
///
/// where aaxis_ba is a 3x1 axis-angle vector, where the magnitude of the angle of rotation
/// can be recovered by finding the norm of the vector, and the axis of rotation is the unit
/// length vector that arises from normalization. Note that the angle around the axis,
/// aaxis_ba, is a right-hand-rule (counter-clockwise positive) angle from 'a' to 'b'. The
/// parameter, rho_ba, is a special translation-like parameter related to 'twist' theory.
/// Assuming that the transformation was applied in a smooth fashion, rho_ba, is most
/// intuitively described as being the fixed-length translation along the curve that that the
/// rotating and translating body would track through space. Alternatively, similar to how
/// an axis-angle vector can be thought of as the application of a constant angular velocity
/// over a fixed period of time, the translation parameter, rho, can be thought of as the
/// application of a linear velocity (expressed in the 'moving' frame) over a the same
/// fixed time (e.g. a car drives 'x' meters while turning at a rate of 'y' rad/s)
///
/// For more information see Barfoot-TRO-2014 Appendix B1.
///
/// Alternatively, we that note that
///
///   T_ba = exp(-xi_ba^) = exp(xi_ab^).
///
/// Both the analytical (numTerms = 0) or the numerical (numTerms > 0) may be evaluated.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix4d vec2tran(const Eigen::Matrix<double,6,1>& vec, unsigned int numTerms = 0);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief compute the matrix log of a transformation matrix (see Barfoot-TRO-2014 Appendix B2)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,1> tran2vec(const Eigen::Matrix3d& rot, const Eigen::Vector3d& trans);
Eigen::Matrix<double,6,1> tran2vec(const Eigen::Matrix4d& mat);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief builds the 6x6 adjoint transformation matrix from a 4x4 one (see eq. 101 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,6> tranAd(const Eigen::Matrix3d& rot, const Eigen::Vector3d& trans);
Eigen::Matrix<double,6,6> tranAd(const Eigen::Matrix4d& mat);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief construction of the 3x3 "Q" matrix, used in the 6x6 Jacobian of SE(3) (see eq. 102 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix3d vec2Q(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis);
Eigen::Matrix3d vec2Q(const Eigen::Matrix<double,6,1>& vec);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief construction of the 6x6 Jacobian of SE(3) (see eq. 100 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,6> vec2jac(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis);
Eigen::Matrix<double,6,6> vec2jac(const Eigen::Matrix<double,6,1>& vec, unsigned int numTerms = 0);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief construction of the 6x6 inverse Jacobian of SE(3) (see eq. 103 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,6> vec2jacinv(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis);
Eigen::Matrix<double,6,6> vec2jacinv(const Eigen::Matrix<double,6,1>& vec, unsigned int numTerms = 0);

} // se3
} // lgmath

#endif // LGM_SE3_PUBLIC_HPP
