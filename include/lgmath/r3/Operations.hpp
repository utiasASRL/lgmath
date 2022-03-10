/**
 * \brief Header file for point and point with covariance definitions.
 * \details Mostly typedefs to standardize interfacing points with
 * transformations.
 *
 * \author Kirk MacTavish
 */
#pragma once

#include <stdexcept>

#include <Eigen/Dense>
#include <lgmath/r3/Types.hpp>
#include <lgmath/se3/Operations.hpp>
#include <lgmath/se3/TransformationWithCovariance.hpp>

namespace lgmath {
namespace r3 {

/** \brief The transform covariance is required to be set */
static constexpr bool COVARIANCE_REQUIRED = true;
/** \brief The transform covariance is not required to be set */
static constexpr bool COVARIANCE_NOT_REQUIRED = false;

/**
 * \brief Transforms a 3x3 covariance for a 3D point (with an assumed certain
 * transform).
 *
 * \note THROW_IF_UNSET Will complain (at compile time) if the transform does
 * not have an explicit covariance. This is always the case for
 * se3::Transformation, so a static_assert prevents compiling without explicitly
 * ignoring the warning (by templating on false).
 *
 * \param T_ba The certain transform that will be used to transform the point
 * covariance
 * \param cov_a The covariance is still in the original frame, A.
 * \param p_b The point is unused since the transformation has no uncertainty
 *
 * \details
 * See eq. 12 in clearpath_virtual_roadways/pm2/mel.pdf for more information.
 */
template <bool THROW_IF_UNSET = COVARIANCE_REQUIRED>
CovarianceMatrix transformCovariance(const se3::Transformation &T_ba,
                                     const CovarianceMatrixConstRef &cov_a,
                                     const HPointConstRef &p_b = HPoint()) {
  (void)&p_b;  // unused
  static_assert(!THROW_IF_UNSET,
                "Error: Transformation never has covariance explicitly set");

  // The component from the point noise
  return T_ba.C_ba() * cov_a * T_ba.C_ba().transpose();
}

/**
 * \brief Transforms a 3x3 covariance for a 3D point (with an uncertain
 * transform).
 *
 * \note THROW_IF_UNSET Will complain (at run time) if the transform does not
 * have an explicit covariance by throwing a runtime_error unless explictly told
 * not to.
 *
 * \param T_ba The certain transform that will be used to transform the point
 * covariance
 * \param cov_a The covariance is still in the original frame, A.
 * \param p_b Note that the point has already been transformed to frame B: p_b =
 * T_ba*p_a
 *
 * \details
 * See eq. 12 in clearpath_virtual_roadways/pm2/mel.pdf for more information.
 */
template <bool THROW_IF_UNSET = COVARIANCE_REQUIRED>
CovarianceMatrix transformCovariance(
    const se3::TransformationWithCovariance &T_ba,
    const CovarianceMatrixConstRef &cov_a, const HPointConstRef &p_b) {
  if (THROW_IF_UNSET && !T_ba.covarianceSet()) {
    throw std::runtime_error(
        "Error: TransformationWithCovariance does not have covariance set");
  }

  // The component from the point noise (reuse the base Transform function)
  const auto &T_ba_base = static_cast<const se3::Transformation &>(T_ba);
  CovarianceMatrix cov_b = transformCovariance<false>(T_ba_base, cov_a, p_b);

  // The component from the transform noise
  if (T_ba.covarianceSet()) {
    auto jacobian = se3::point2fs(p_b.hnormalized()).topRows<3>();
    cov_b += jacobian * T_ba.cov() * jacobian.transpose();
  }

  return cov_b;
}

}  // namespace r3
}  // namespace lgmath