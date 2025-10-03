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
   * The transformation will be T_ab = [C_ab, r_ba_ina; 0 0 0 1]
   */
  explicit Transformation(const Eigen::Matrix2d& C_ab,
                          const Eigen::Vector2d& r_ba_ina);

  /**
   * \brief Constructor.
   * The transformation will be T_ab = vec2tran(xi_ba), xi_ba must be 3x1
   */
  explicit Transformation(const Eigen::Vector3d& xi_ba);

  /**
   * \brief Constructor.
   * The transformation will be T_ab = vec2tran(xi_ba), xi_ba must be 3x1
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
  const Eigen::Matrix2d& C_ab() const;

  /** \brief Gets the underlying r_ba_ina vector. */
  const Eigen::Vector2d r_ba_ina() const;

  /** \brief Gets r_ba_inb = -C_ab.transpose() * r_ba_ina */
  Eigen::Vector2d r_ab_inb() const;

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

  /** \brief Right-hand side multiply the homogeneous vector p_b */
  Eigen::Vector3d operator*(const Eigen::Ref<const Eigen::Vector3d>& p_b) const;

 private:
  /** \brief Rotation matrix from b to a */
  Eigen::Matrix2d C_ab_;

  /** \brief Translation vector from b to a, expressed in frame a */
  Eigen::Vector2d r_ba_ina_;
};

}  // namespace se2
}  // namespace lgmath

/** \brief print transformation */
std::ostream& operator<<(std::ostream& out,
                         const lgmath::se2::Transformation& T);
