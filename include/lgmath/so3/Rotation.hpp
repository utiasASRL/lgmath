/**
 * \file Rotation.hpp
 * \brief Header file for a rotation matrix class.
 * \details Light weight rotation class, intended to be fast, and not to provide
 *          unnecessary functionality.
 *
 * \author Sean Anderson
 */
#pragma once

#include <Eigen/Dense>

namespace lgmath {
namespace so3 {

class Rotation {
 public:
  /** \brief Default constructor */
  Rotation();

  /** \brief Copy constructor. */
  Rotation(const Rotation&) = default;

  /** \brief Move constructor. */
  Rotation(Rotation&& C) = default;

  /** \brief Copy constructor (from Eigen) */
  Rotation(const Eigen::Matrix3d& C);

  /** \brief Constructor. The rotation will be C_ba = vec2rot(aaxis_ab) */
  explicit Rotation(const Eigen::Vector3d& aaxis_ab, unsigned int numTerms = 0);

  /**
   * \brief Constructor. The rotation will be C_ba = vec2rot(aaxis_ab),
   * aaxis_ab must be 3x1
   */
  explicit Rotation(const Eigen::VectorXd& aaxis_ab);

  /** \brief Destructor. */
  ~Rotation() = default;

  /** \brief Copy assignment operator. */
  Rotation& operator=(const Rotation&) = default;

  /** \brief Move assignment operator. */
  Rotation& operator=(Rotation&& C) = default;

  /** \brief Gets the underlying rotation matrix */
  const Eigen::Matrix3d& matrix() const;

  /**
   * \brief Get the corresponding Lie algebra (axis-angle) using the logarithmic
   * map
   */
  Eigen::Vector3d vec() const;

  /** \brief Get the inverse (transpose) matrix */
  Rotation inverse() const;

  /**
   * \brief Reproject the rotation matrix back onto SO(3).
   * \param[in] force Setting force to false triggers a conditional reproject
   * that only happens if the determinant is of the rotation matrix is poor;
   * this is more efficient than always performing it.
   */
  void reproject(bool force = true);

  /** \brief In-place right-hand side multiply C_rhs */
  Rotation& operator*=(const Rotation& C_rhs);

  /** \brief Right-hand side multiply C_rhs */
  Rotation operator*(const Rotation& C_rhs) const;

  /**
   * \brief In-place right-hand side multiply this matrix by the inverse of
   * C_rhs
   */
  Rotation& operator/=(const Rotation& C_rhs);

  /** \brief Right-hand side multiply this matrix by the inverse of C_rhs */
  Rotation operator/(const Rotation& C_rhs) const;

  /** \brief Right-hand side multiply this matrix by the point vector p_a */
  Eigen::Vector3d operator*(const Eigen::Ref<const Eigen::Vector3d>& p_a) const;

 private:
  /** \brief Rotation matrix from a to b */
  Eigen::Matrix3d C_ba_;
};

}  // namespace so3
}  // namespace lgmath

/** \brief print transformation */
std::ostream& operator<<(std::ostream& out, const lgmath::so3::Rotation& T);
