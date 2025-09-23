/**
 * \brief \brief Header file for the SE2 Lie Group types.
 * \details These types provide a standardized definition for various SE2
 * quantities.
 *
 * \author Daniil Lisus
 */
#pragma once

#include <Eigen/Core>

/// Lie Group Math - Special Euclidean Group in two dimensions
namespace lgmath {
namespace se2 {

/**
 * \brief  A translation vector, r_ba_ina, which translates points from the
 * origin to b in frame a
 */
using TranslationVector = Eigen::Vector2d;

/**
 * \brief  A Lie algebra vector composed of a stacked translation and axis-angle
 * rotation.
 *
 *   xi_ba = [  rho_ba]
 *           [angle_ba]
 *
 */
using LieAlgebra = Eigen::Matrix<double, 3, 1>;

/**
 * \brief  The covariance matrix of a Lie algebra vector
 */
using LieAlgebraCovariance = Eigen::Matrix<double, 3, 3>;

/**
 * \brief  A transformation, T_ab, transforms points from frame a to frame b.
 */
using TransformationMatrix = Eigen::Matrix3d;

}  // namespace se2
}  // namespace lgmath
