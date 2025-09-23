/**
 * \brief Header file for the SO2 Lie Group types.
 * \details These types provide a standardized definition for various SO2
 * quantities.
 *
 * \author Daniil Lisus
 */
#pragma once

#include <Eigen/Core>

/// Lie Group Math - Special Orthogonal Group
namespace lgmath {
namespace so2 {

/**
 * \brief An angle denoting rotation.
 * \details Angle is a double scalar denoting the magnitude of rotation.
 * The angle is a right-hand-rule (counter-clockwise positive) angle from 'a' to 'b'.
 */
using Angle = double;

/**
 * \brief A rotation matrix.
 * \details The convention is that C_ba rotates points from frame a to frame b.
 */
using RotationMatrix = Eigen::Matrix2d;

}  // namespace so2
}  // namespace lgmath
