/**
 * \file Rotation.hpp
 * \brief Header file for a rotation matrix class.
 * \details Light weight rotation class, intended to be fast, and not to provide
 *          unnecessary functionality.
 *
 * \author Daniil Lisus
 */
#pragma once

#include <Eigen/Dense>

namespace lgmath {
namespace so2 {

class Rotation {
 public:
  /** \brief Default constructor */
  Rotation();

  /** \brief Copy constructor. */
  Rotation(const Rotation&) = default;

  /** \brief Move constructor. */
  Rotation(Rotation&&) = default;

  /** \brief Copy constructor (from Eigen) */
  explicit Rotation(const Eigen::Matrix2d& C);

  /** \brief Constructor. The rotation will be C_ab = vec2rot(angle_ba) */
  explicit Rotation(const double angle_ba, unsigned int numTerms = 0);

  /**
   * \brief Constructor.
   * The rotation will be C_ab = vec2rot(angle_ba)
   */
  explicit Rotation(const double angle_ba);

  /** \brief Destructor. */
  virtual ~Rotation() = default;

  /** \brief Copy assignment operator. */
  virtual Rotation& operator=(const Rotation&) = default;

  /** \brief Move assignment operator. */
  virtual Rotation& operator=(Rotation&&) = default;

  /** \brief Gets the underlying rotation matrix */
  const Eigen::Matrix2d& matrix() const;

  /** \brief Get the corresponding Lie algebra using the logarithmic map */
  double vec() const;

  /** \brief Get the inverse (transpose) matrix */
  Rotation inverse() const;

  /**
   * \brief Reproject the rotation matrix back onto SO(2).
   * \param[in] force Setting force to false triggers a conditional reproject
   * that only happens if the determinant is of the rotation matrix is poor;
   * this is more efficient than always performing it.
   */
  void reproject(bool force = true);

  /** \brief In-place right-hand side multiply C_rhs */
  virtual Rotation& operator*=(const Rotation& C_rhs);

  /** \brief Right-hand side multiply C_rhs */
  virtual Rotation operator*(const Rotation& C_rhs) const;

  /** \brief In-place right-hand side multiply the inverse of C_rhs */
  virtual Rotation& operator/=(const Rotation& C_rhs);

  /** \brief Right-hand side multiply the inverse of C_rhs */
  virtual Rotation operator/(const Rotation& C_rhs) const;

  /** \brief Right-hand side multiply the point vector p_a */
  Eigen::Vector2d operator*(const Eigen::Ref<const Eigen::Vector2d>& p_a) const;

 private:
  /** \brief Rotation matrix from b to a */
  Eigen::Matrix2d C_ab_;
};

}  // namespace so2
}  // namespace lgmath

/** \brief print transformation */
std::ostream& operator<<(std::ostream& out, const lgmath::so2::Rotation& T);
