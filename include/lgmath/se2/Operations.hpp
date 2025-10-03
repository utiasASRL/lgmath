/**
 * \file Operations.hpp
 * \brief Header file for the SE2 Lie Group math functions.
 * \details These namespace functions provide implementations of the special
 * Euclidean (SE) in 2D Lie group functions that we commonly use in robotics.
 *
 * \author Daniil Lisus
 */
#pragma once

#include <Eigen/Core>

/// Lie Group Math - Special Euclidean Group in 2D
namespace lgmath {
namespace se2 {

/**
 * \brief Builds the 3x3 "skew symmetric matrix"
 * \details
 * The hat (^) operator, builds the 3x3 skew symmetric matrix from the scalar angle
 * and 2x1 translation vector.
 *
 * hat(rho, angle) = [angle^ rho] = [0.0    -angle rho1]
 *                   [  0^T    0]   [angle  0.0    rho2]
 *                                  [0.0    0.0    0.0]
 *
 */
Eigen::Matrix3d hat(const Eigen::Vector2d& rho, const double angle);

/**
 * \brief Builds the 3x3 "skew symmetric matrix"
 * \details
 * The hat (^) operator, builds the 3x3 skew symmetric matrix from
 * the 3x1 se2 algebra vector, xi:
 *
 * xi^ = [rho  ] = [angle^ rho] = [0.0    -angle rho1]
 *       [angle]   [  0^T    0]   [angle  0.0    rho2]
 *                                [0.0    0.0    0.0]
 *
 */
Eigen::Matrix3d hat(const Eigen::Matrix<double, 3, 1>& xi);

/**
 * \brief Builds the 3x3 "curly hat" matrix (related to the skew symmetric
 * matrix)
 * \details
 * The curly hat operator builds the 3x3 skew symmetric matrix from the scalar
 * angle and 2x1 translation vector.
 *
 * curlyhat(rho, angle) = [angle^   -S*rho^]; , S = [0 -1; 1 0]
 *                        [     0      0   ]
 *
 * See eq. 104 in Integral Forms in Matrix Lie Groups by Barfoot for more information.
 */
Eigen::Matrix<double, 3, 3> curlyhat(const Eigen::Vector2d& rho,
                                     const double angle);

/**
 * \brief Builds the 3x3 "curly hat" matrix (related to the skew symmetric
 * matrix)
 * \details
 * The curly hat operator builds the 6x6 skew symmetric matrix
 * from the 3x1 se2 algebra vector, xi:
 *
 * curlyhat(xi) = curlyhat([rho  ]) = [angle^   -S*rho^]; , S = [0 -1; 1 0]
 *                        ([angle])   [     0      0   ]
 *
 * See eq. 104 in Integral Forms in Matrix Lie Groups by Barfoot for more information.
 */
Eigen::Matrix<double, 3, 3> curlyhat(const Eigen::Matrix<double, 3, 1>& xi);

/**
 * \brief Turns a homogeneous point into a special 3x3 matrix (circle-dot
 * operator)
 * \details
 * See eq. 72 in Barfoot-TRO-2014 for more information on the SE(3) version.
 * The SE(2) version can be derived in a similar manner and is
 * as follows for a homogeneous point p = [epsilon, s]^T = \[px, py, s]^T
 * with scale s:
 * 
 * point2fs(p) = [ s*I   S*epsilon ]
 *               [ 0^T       0     ]
 * where S = [0 -1; 1 0] and I is the 2x2 identity matrix.
 */
Eigen::Matrix<double, 3, 3> point2fs(const Eigen::Vector2d& p,
                                     double scale = 1);

/**
 * \brief Turns a homogeneous point into a special 3x3 matrix (double-circle
 * operator)
 *
 * See eq. 72 in Barfoot-TRO-2014 for more information on the SE(3) version.
 * The SE(2) version can be derived in a similar manner and is
 * as follows for a homogeneous point p = [epsilon, s]^T = \[px, py, s]^T
 * with scale s:
 * 
 * point2sf(p) = [ 0*I              epsilon ]
 *               [ -(S*epsilon)^T     0     ]
 * where S = [0 -1; 1 0] and I is the 2x2 identity matrix.
 */
Eigen::Matrix<double, 3, 3> point2sf(const Eigen::Vector2d& p,
                                     double scale = 1);

/**
 * \brief Builds a transformation matrix using the analytical exponential map
 * \details
 * This function builds a transformation matrix, T_ab, using the analytical
 * exponential map, from the se2 algebra vector, xi_ba,
 *
 *   T_ab = exp(xi_ba^) = [ C_ab r_ba_ina],   xi_ba = [  rho_ba]
 *                        [  0^T        1]            [angle_ba]
 *
 * where C_ab is a 2x2 rotation matrix from 'b' to 'a', r_ba_ina is the 2x1
 * translation vector from 'a' to 'b' expressed in frame 'a', angle_ba is a
 * scalar angle. Note that the angle angle_ba, is a right-hand-rule
 * (counter-clockwise positive) angle from 'a' to 'b'.
 *
 * The parameter, rho_ba, is a special translation-like parameter related to
 * 'twist' theory. It is most intuitively described as being like a constant
 * linear velocity (expressed in the smoothly-moving frame) for a fixed
 * duration; for example, consider the curve of a car driving 'x' meters while
 * turning at a rate of 'y' rad/s.
 *
 * For more information see Barfoot-TRO-2014 Appendix B1.
 *
 * Alternatively, we that note that
 *
 *   T_ba = exp(-xi_ba^) = exp(xi_ab^).
 */
Eigen::Matrix3d vec2tran(const Eigen::Matrix<double, 3, 1>& xi_ba);

/**
 * \brief Builds the 2x2 rotation and 2x1 translation using the exponential
 * map.
 */
void vec2tran(const Eigen::Matrix<double, 3, 1>& xi_ba,
            Eigen::Matrix2d* out_C_ab, Eigen::Vector2d* out_r_ba_ina);

/**
 * \brief Compute the matrix log of a transformation matrix (from the rotation
 * and trans)
 * \details
 * Compute the inverse of the exponential map (the logarithmic map). This lets
 * us go from a the 2x2 rotation and 2x1 translation vector back to a 3x1 se2
 * algebra vector (composed of a scalar angle and 2x1 twist-translation
 * vector). In some cases, when the rotation in the transformation matrix is
 * 'numerically off', this involves some 'projection' back to SE(2).
 *
 *   xi_ba = ln(T_ab)
 *
 * where xi_ba is the 3x1 se2 algebra vector. Alternatively, we that note that
 *
 *   xi_ab = -xi_ba = ln(T_ba) = ln(T_ab^{-1})
 */
Eigen::Matrix<double, 3, 1> tran2vec(const Eigen::Matrix2d& C_ab,
                                     const Eigen::Vector2d& r_ba_ina);

/**
 * \brief Compute the matrix log of a transformation matrix
 * \details
 * Compute the inverse of the exponential map (the logarithmic map). This lets
 * us go from a 3x3 transformation matrix back to a 3x1 se2 algebra vector
 * (composed of a scalar angle and 2x1 twist-translation vector). In
 * some cases, when the rotation in the transformation matrix is 'numerically
 * off', this involves some 'projection' back to SE(2).
 *
 *   xi_ba = ln(T_ab)
 *
 * where xi_ba is the 3x1 se2 algebra vector. Alternatively, we that note that
 *
 *   xi_ab = -xi_ba = ln(T_ba) = ln(T_ab^{-1})
 */
Eigen::Matrix<double, 3, 1> tran2vec(const Eigen::Matrix3d& T_ab);

/**
 * \brief Builds the 3x3 adjoint transformation matrix from the 2x2 rotation
 * matrix and 2x1 translation vector.
 * \details
 * Builds the 3x3 adjoint transformation matrix from the 2x2 rotation matrix and
 * 2x1 translation vector.
 *
 *  Adjoint(T_ab) =  [C_ab   -S*r_ba_ina] , S = [0 -1; 1 0]
 *                   [  0^T        1    ]
 *
 * See eq. 104 in Integral Forms in Matrix Lie Groups by Barfoot for more information.
 */
Eigen::Matrix<double, 3, 3> tranAd(const Eigen::Matrix2d& C_ab,
                                   const Eigen::Vector2d& r_ba_ina);

/**
 * \brief Builds the 3x3 adjoint transformation matrix from a 3x3 one
 * \details
 * Builds the 3x3 adjoint transformation matrix from a 3x3 transformation matrix
 *
 *  Adjoint(T_ab) =  [C_ab   -S*r_ba_ina] , S = [0 -1; 1 0]
 *                   [  0^T        1    ]
 *
 * See eq. 104 in Integral Forms in Matrix Lie Groups by Barfoot for more information.
 */
Eigen::Matrix<double, 3, 3> tranAd(const Eigen::Matrix3d& T_ab);

/**
 * \brief Construction of the 2x2 "Gamma 1" matrix, used in SE(2)
 * \details
 * See eq. 109a in Integral Forms in Matrix Lie Groups by Barfoot for more information.
 */
Eigen::Matrix2d vec2Gamma1(const double angle);

/**
 * \brief Construction of the 2x2 "Gamma 1" matrix, used in SE(2)
 * \details
 * See eq. 109a in Integral Forms in Matrix Lie Groups by Barfoot for more information.
 */
Eigen::Matrix2d vec2Gamma1(const Eigen::Matrix<double, 3, 1>& xi_ba);

/**
 * \brief Construction of the 2x2 "Gamma 2" matrix, used in SE(2)
 * \details
 * See eq. 109b in Integral Forms in Matrix Lie Groups by Barfoot for more information.
 */
Eigen::Matrix2d vec2Gamma2(const double angle);

/**
 * \brief Construction of the 2x2 "Gamma 2" matrix, used in SE(2)
 * \details
 * See eq. 109b in Integral Forms in Matrix Lie Groups by Barfoot for more information.
 */
Eigen::Matrix2d vec2Gamma2(const Eigen::Matrix<double, 3, 1>& xi_ba);

/**
 * \brief Builds the 3x3 Jacobian matrix of SE(2) using the analytical
 * expression
 * \details
 * Build the 3x3 left Jacobian of SE(3).
 *
 * For the sake of a notation, we assign subscripts consistence with the
 * transformation,
 *
 *   J_ab = J(xi_ba)
 *
 * Where applicable, we also note that
 *
 *   J(xi_ba) = Adjoint(exp(xi_ba^)) * J(-xi_ba),
 *
 * and
 *
 *   Adjoint(exp(xi_ba^)) = identity + curlyhat(xi_ba) * J(xi_ba).
 *
 * For more information see eq. 108 in in Integral Forms in Matrix Lie Groups by Barfoot
 */
Eigen::Matrix<double, 3, 3> vec2jac(const Eigen::Vector2d& rho_ba,
                                    const double angle_ba);

/**
 * \brief Builds the 3x3 Jacobian matrix of SE(2) from the se(2) algebra.
 * \details
 * For more information see eq. 108 in Integral Forms in Matrix Lie Groups by Barfoot.
 */
Eigen::Matrix<double, 3, 3> vec2jac(const Eigen::Matrix<double, 3, 1>& xi_ba);

/**
 * \brief Builds the 3x3 inverse Jacobian matrix of SE(2) using the analytical
 * expression
 * \details
 * Build the 3x3 inverse left Jacobian of SE(2).
 *
 * For the sake of a notation, we assign subscripts consistence with the
 * transformation,
 *
 *   J_ab_inverse = J(xi_ba)^{-1},
 *
 * Please note that J_ab_inverse is not equivalent to J_ba:
 *
 *   J(xi_ba)^{-1} != J(-xi_ba)
 *
 * We find the analytical inverse of the Jacobian by inverting
 * eq. 108 in Integral Forms in Matrix Lie Groups by Barfoot:
 * 
 *  J_ab_inverse = [Gamma_1^{-1}   Gamma_1^{-1}*S*Gamma_2*rho_ba]
 *                 [     0^T                1                   ]
 */
Eigen::Matrix<double, 3, 3> vec2jacinv(const Eigen::Vector2d& rho_ba,
                                       const double angle_ba);

/**
 * \brief Builds the 6x6 inverse Jacobian matrix of SE(2) from the se(2)
 * algebra.
 * \details
 * For more information see eq. 108 in Integral Forms in Matrix Lie Groups by Barfoot.
 */
Eigen::Matrix<double, 3, 3> vec2jacinv(const Eigen::Matrix<double, 3, 1>& xi_ba);

}  // namespace se3
}  // namespace lgmath
