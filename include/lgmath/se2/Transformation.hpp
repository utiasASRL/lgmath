/**
 * \file Transformation.hpp
 * \brief Header file for an SE(2) transformation matrix class.
 * \details Light weight transformation class, intended to be fast, and not to
 * provide unnecessary functionality.
 *
 * \author Daniil Lisus
 */
#pragma once

#include <Eigen/Dense>

// Forward declaration to avoid circular dependency
namespace lgmath { namespace se3 { class Transformation; } }

namespace lgmath {
namespace se2 {

class Transformation {
 public:
  /** \brief Default constructor */
  Transformation();

  /** \brief Copy constructor. */
  Transformation(const Transformation&) = default;

  /** \brief Move constructor. */
  Transformation(Transformation&&) = default;

  /** \brief Copy constructor (from Eigen) */
  explicit Transformation(const Eigen::Matrix3d& T);

  /**
   * \brief Constructor.
   * The transformation will be T_ba = [C_ba, -C_ba*r_ba_ina; 0 0 0 1]
   */
  explicit Transformation(const Eigen::Matrix2d& C_ba,
                          const Eigen::Vector2d& r_ba_ina);

  /**
   * \brief Constructor.
   * The transformation will be T_ba = vec2tran(xi_ab), xi_ab must be 3x1
   */
  explicit Transformation(const Eigen::Vector3d& xi_ab);

  /**
   * \brief Constructor.
   * The transformation will be T_ba = vec2tran(xi_ab), xi_ab must be 3x1
   */
  explicit Transformation(const Eigen::VectorXd& xi_ab);

  /** \brief Destructor. Default implementation. */
  virtual ~Transformation() = default;

  /** \brief Copy assignment operator. */
  virtual Transformation& operator=(const Transformation&) = default;

  /** \brief Move assignment operator. */
  virtual Transformation& operator=(Transformation&& T) = default;

  /** \brief Gets basic matrix representation of the transformation */
  Eigen::Matrix3d matrix() const;

  /** \brief Gets the underlying rotation matrix */
  const Eigen::Matrix2d& C_ba() const;

  /** \brief Gets r_ba_ina = -C_ba.transpose() * r_ab_inb */
  Eigen::Vector2d r_ba_ina() const;

  /** \brief Gets the underlying r_ab_inb vector. */
  const Eigen::Vector2d& r_ab_inb() const;

  /** \brief Get the corresponding Lie algebra using the logarithmic map */
  Eigen::Matrix<double, 3, 1> vec() const;

  /** \brief Get the inverse matrix */
  Transformation inverse() const;

  /** \brief Get the 3x3 adjoint transformation matrix */
  Eigen::Matrix<double, 3, 3> adjoint() const;

  /**
   * \brief Reproject the transformation matrix back onto SE(2).
   */
  void reproject();

  /** \brief In-place right-hand side multiply T_rhs */
  virtual Transformation& operator*=(const Transformation& T_rhs);

  /** \brief Right-hand side multiply T_rhs */
  virtual Transformation operator*(const Transformation& T_rhs) const;

  /** \brief In-place right-hand side multiply the inverse of T_rhs */
  virtual Transformation& operator/=(const Transformation& T_rhs);

  /** \brief Right-hand side multiply the inverse of T_rhs */
  virtual Transformation operator/(const Transformation& T_rhs) const;

  /** \brief Right-hand side multiply the homogeneous vector p_a */
  Eigen::Vector3d operator*(const Eigen::Ref<const Eigen::Vector3d>& p_a) const;

  /**
   * \brief Convert to SE(3) transformation by embedding in the xy-plane
   * \details Embeds the SE(2) transformation into SE(3) by placing it in the
   * xy-plane with z=0.
   */
  se3::Transformation toSE3() const;

 private:
  /** \brief Rotation matrix from a to b */
  Eigen::Matrix2d C_ba_;

  /** \brief Translation vector from b to a, expressed in frame b */
  Eigen::Vector2d r_ab_inb_;
};

}  // namespace se2
}  // namespace lgmath

/** \brief print transformation */
std::ostream& operator<<(std::ostream& out,
                         const lgmath::se2::Transformation& T);
