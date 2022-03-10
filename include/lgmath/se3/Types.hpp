/**
 * \brief \brief Header file for the SE3 Lie Group types.
 * \details These types provide a standardized definition for various SE3
 * quantities.
 *
 * \author Kirk MacTavish
 */
#pragma once

#include <Eigen/Core>

/// Lie Group Math - Special Orthogonal Group
namespace lgmath {
namespace se3 {

/**
 * \brief  A translation vector, r_ba_ina, which translates points from the
 * origin to b in frame a
 */
using TranslationVector = Eigen::Vector3d;

/**
 * \brief  A Lie algebra vector composed of a stacked translation and axis-angle
 * rotation.
 *
 *   xi_ba = [  rho_ba]
 *           [aaxis_ba]
 *
 */
using LieAlgebra = Eigen::Matrix<double, 6, 1>;

/**
 * \brief  The covariance matrix of a Lie algebra vector
 */
using LieAlgebraCovariance = Eigen::Matrix<double, 6, 6>;

/**
 * \brief  A transformation, T_ba, transforms points from frame a to frame b.
 *
 *   T_ba = [ C_ba, -C_ba*r_ba_ina]
 *          [0 0 0,              1]
 */
using TransformationMatrix = Eigen::Matrix4d;

}  // namespace se3
}  // namespace lgmath
