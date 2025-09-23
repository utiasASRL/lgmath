/**
 * \file Operations.hpp
 * \brief Header file for the SO2 Lie Group math functions.
 * \details These namespace functions provide implementations of the planar special
 * orthogonal (SO) Lie group functions that we commonly use in robotics.
 *
 * \author Daniil Lisus
 */
#pragma once

#include <Eigen/Core>

/// Lie Group Math - Special Orthogonal Group
namespace lgmath {
namespace so2 {

/**
 * \brief Builds the 2x2 skew symmetric matrix
 * \details
 * The hat (^) operator, builds the 2x2 skew symmetric matrix from the scalar
 * angle:
 *
 * v^ = [0.0  -v]
 *      [v   0.0]
 *
 */
Eigen::Matrix2d hat(const double angle);

/**
 * \brief Builds a rotation matrix using the exponential map
 * \details
 * This function builds a rotation matrix, C_ab, using the exponential map
 *
 *   C_ab = exp(angle_ba^),
 *
 * where angle_ba is a right-hand-rule (counter-clockwise positive) angle from 'a' to 'b'.
 * Note, that this follows the 'robotics convention' and is a parallel to the SO(3)
 * exponential map. Make sure that you are actually feeding in angle_ba, not angle_ab.
 * For more information see sec. 7.3.2 in State Estimation for Robotics by Barfoot.
 */
Eigen::Matrix2d vec2rot(const double angle_ba);

/**
 * \brief Compute the matrix log of a rotation matrix
 * \details
 * Compute the inverse of the exponential map (the logarithmic map). This lets
 * us go from a 2x2 rotation matrix back to a scalar angle parameterization.
 * In some cases, when the rotation matrix is 'numerically off', this involves
 * some 'projection' back to SO(2).
 *
 *   angle_ba = ln(C_ab)
 *
 * where angle_ba is a scalar right-hand-rule (counter-clockwise positive) angle from 'a' to 'b'.
 *
 * Alternatively, we that note that
 *
 *   angle_ab = -angle_ba = ln(C_ba) = ln(C_ab^T)
 */
double rot2vec(const Eigen::Matrix2d& C_ab);

/**
 * \brief Builds the scalar Jacobian of SO(2)
 * \details
 * Build the scalar left Jacobian of SO(2).
 *
 * This Jacobian is equal to the right Jacobian and is simply 1.
 * We keep the function for consistency with SO(3).
 */
double vec2jac();

/**
 * \brief Builds the scalar inverse Jacobian of SO(2)
 * \details
 * Builds the scalar inverse Jacobian of SO(2).
 *
 * This is just equal to the inverse of the Jacobian, which is also 1.
 * We keep the function for consistency with SO(3).
 */
double vec2jacinv();

}  // namespace so2
}  // namespace lgmath
