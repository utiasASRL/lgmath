/**
 * \brief Header file for the SO3 Lie Group types.
 * \details These types provide a standardized definition for various SO3
 * quantities.
 *
 * \author Kirk MacTavish
 */
#pragma once

#include <Eigen/Core>

/// Lie Group Math - Special Orthogonal Group
namespace lgmath {
namespace so3 {

/**
 * \brief An axis angle rotation.
 * \details AxisAngle is a 3x1 axis-angle vector, the magnitude of the angle of
 * rotation can be recovered by finding the norm of the vector, and the axis of
 * rotation is the unit- length vector that arises from normalization. Note that
 * the angle around the axis, aaxis_ba, is a right-hand-rule (counter-clockwise
 * positive) angle from 'a' to 'b'.
 */
using AxisAngle = Eigen::Vector3d;

/**
 * \brief A rotation matrix.
 * \details The convention is that C_ba rotates points from frame a to frame b.
 */
using RotationMatrix = Eigen::Matrix3d;

}  // namespace so3
}  // namespace lgmath
