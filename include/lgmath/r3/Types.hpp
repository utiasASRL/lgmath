/**
 * \brief Header file for Euclidean space R3 types.
 * \details Mostly type aliases to standardize interfacing points with
 * transformations.
 *
 * \author Kirk MacTavish
 */
#pragma once

#include <Eigen/Dense>

namespace lgmath {
namespace r3 {

/// A 3D point
using Point = Eigen::Vector3d;
using PointRef = Eigen::Ref<Point>;
using PointConstRef = Eigen::Ref<const Point>;

/// A 3D homogeneous point
using HPoint = Eigen::Vector4d;
using HPointRef = Eigen::Ref<HPoint>;
using HPointConstRef = Eigen::Ref<const HPoint>;

/// A 3x3 covariance for a 3D point
using CovarianceMatrix = Eigen::Matrix3d;
using CovarianceMatrixRef = Eigen::Ref<CovarianceMatrix>;
using CovarianceMatrixConstRef = Eigen::Ref<const CovarianceMatrix>;

}  // namespace r3
}  // namespace lgmath
