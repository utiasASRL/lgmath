//////////////////////////////////////////////////////////////////////////////////////////////
/// \file Operations.hpp
/// \brief Header file for the SO3 Lie Group math functions.
/// \details These namespace functions provide implementations of the special orthogonal (SO)
///          Lie group functions that we commonly use in robotics.
///
/// \author Sean Anderson
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef LGM_SO3_PUBLIC_HPP
#define LGM_SO3_PUBLIC_HPP

#include <Eigen/Core>

/////////////////////////////////////////////////////////////////////////////////////////////
/// Lie Group Math - Special Orthogonal Group
/////////////////////////////////////////////////////////////////////////////////////////////
namespace lgmath {
namespace so3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief builds the 3x3 skew symmetric matrix (see eq. 5 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,3,3> hat(const Eigen::Matrix<double,3,1>& vec);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief builds a rotation matrix using the exponential map (see eq. 97 in Barfoot-TRO-2014)
///
///        **For right-hand-rule rotations:
///            C_ab = exp(phi_ba), where phi_ba is counter-clockwise positive
///        **For left-hand-rule rotations:
///            C_ba = exp(phi_ab), where phi_ab is clockwise positive (-phi_ba)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,3,3> vec2rot(const Eigen::Matrix<double,3,1>& aaxis, unsigned int numTerms = 0);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief efficiently builds the rotation matrix and Jacobian (faster then finding each
///        individually) using the identity rot(a) = eye(3) + hat(a)*jac(a)
///
///        **For right-hand-rule rotations:
///            C_ab = 1 + hat(phi_ba)*J(phi_ba), where phi_ba is counter-clockwise positive
///        **For left-hand-rule rotations:
///            C_ba = 1 + hat(phi_ab)*J(phi_ab), where phi_ab is clockwise positive (-phi_ba)
//////////////////////////////////////////////////////////////////////////////////////////////
void vec2rot(const Eigen::Matrix<double,3,1>& aaxis, Eigen::Matrix<double,3,3>* outRot,
             Eigen::Matrix<double,3,3>* outJac);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief compute the matrix log of a rotation matrix (see Barfoot-TRO-2014 Appendix B2)
///
///        **For right-hand-rule rotations:
///            phi_ba = ln(C_ab), where phi_ba is counter-clockwise positive
///        **For left-hand-rule rotations:
///            phi_ab = ln(C_ba), where phi_ab is clockwise positive (-phi_ba)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,3,1> rot2vec(const Eigen::Matrix<double,3,3>& mat);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief builds the 3x3 jacobian matrix of SO(3) (see eq. 98 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,3,3> vec2jac(const Eigen::Matrix<double,3,1>& aaxis, unsigned int numTerms = 0);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief builds the 3x3 inverse jacobian matrix of SO(3) (see eq. 99 in Barfoot-TRO-2014)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,3,3> vec2jacinv(const Eigen::Matrix<double,3,1>& aaxis, unsigned int numTerms = 0);

} // so3
} // lgmath


#endif // LGM_SO3_PUBLIC_HPP
