/**
 * \file Transformation.hpp
 * \brief Header file for a transformation matrix class.
 * \details Light weight transformation class, intended to be fast, and not to
 * provide unnecessary functionality.
 *
 * \author Sean Anderson
 */
#pragma once

#include <Eigen/Dense>

// Forward declaration to avoid circular dependency  
namespace lgmath { namespace se2 { class Transformation; } }

namespace lgmath {
namespace se3 {

class Transformation {
 public:
  /** \brief Default constructor */
  Transformation();

  /** \brief Copy constructor. */
  Transformation(const Transformation&) = default;

  /** \brief Move constructor. */
  Transformation(Transformation&&) = default;

  /** \brief Copy constructor (from Eigen) */
  explicit Transformation(const Eigen::Matrix4d& T);

  /**
   * \brief Constructor.
   * The transformation will be T_ba = [C_ba, -C_ba*r_ba_ina; 0 0 0 1]
   */
  explicit Transformation(const Eigen::Matrix3d& C_ba,
                          const Eigen::Vector3d& r_ba_ina);

  /** \brief Constructor. The transformation will be T_ba = vec2tran(xi_ab) */
  explicit Transformation(const Eigen::Matrix<double, 6, 1>& xi_ab,
                          unsigned int numTerms = 0);

  /**
   * \brief Constructor.
   * The transformation will be T_ba = vec2tran(xi_ab), xi_ab must be 6x1
   */
  explicit Transformation(const Eigen::VectorXd& xi_ab);

  /** \brief Destructor. Default implementation. */
  virtual ~Transformation() = default;

  /** \brief Copy assignment operator. */
  virtual Transformation& operator=(const Transformation&) = default;

  /** \brief Move assignment operator. */
  virtual Transformation& operator=(Transformation&& T) = default;

  /** \brief Gets basic matrix representation of the transformation */
  Eigen::Matrix4d matrix() const;

  /** \brief Gets the underlying rotation matrix */
  const Eigen::Matrix3d& C_ba() const;

  /** \brief Gets r_ba_ina = -C_ba.transpose() * r_ab_inb */
  Eigen::Vector3d r_ba_ina() const;

  /** \brief Gets the underlying r_ab_inb vector. */
  const Eigen::Vector3d& r_ab_inb() const;

  /** \brief Get the corresponding Lie algebra using the logarithmic map */
  Eigen::Matrix<double, 6, 1> vec() const;

  /** \brief Get the inverse matrix */
  Transformation inverse() const;

  /** \brief Get the 6x6 adjoint transformation matrix */
  Eigen::Matrix<double, 6, 6> adjoint() const;

  /**
   * \brief Reproject the transformation matrix back onto SE(3).
   * \param[in] force Setting force to false triggers a conditional reproject
   * that only happens if the determinant is of the rotation matrix is poor;
   * this is more efficient than always performing it.
   */
  void reproject(bool force = true);

  /** \brief In-place right-hand side multiply T_rhs */
  virtual Transformation& operator*=(const Transformation& T_rhs);

  /** \brief Right-hand side multiply T_rhs */
  virtual Transformation operator*(const Transformation& T_rhs) const;

  /** \brief In-place right-hand side multiply the inverse of T_rhs */
  virtual Transformation& operator/=(const Transformation& T_rhs);

  /** \brief Right-hand side multiply the inverse of T_rhs */
  virtual Transformation operator/(const Transformation& T_rhs) const;

  /** \brief Right-hand side multiply the homogeneous vector p_a */
  Eigen::Vector4d operator*(const Eigen::Ref<const Eigen::Vector4d>& p_a) const;

  /**
   * \brief Project to SE(2) transformation by extracting xy-plane motion
   * \details Projects the SE(3) transformation onto the xy-plane, discarding
   * z-translation and rotations around x/y axes. This is useful for 2D navigation
   * applications where only planar motion is relevant.
   * \return SE(2) transformation representing the xy-plane motion
   */
  lgmath::se2::Transformation toSE2() const;

 private:
  /** \brief Rotation matrix from a to b */
  Eigen::Matrix3d C_ba_;

  /** \brief Translation vector from b to a, expressed in frame b */
  Eigen::Vector3d r_ab_inb_;
};

}  // namespace se3
}  // namespace lgmath

/** \brief print transformation */
std::ostream& operator<<(std::ostream& out,
                         const lgmath::se3::Transformation& T);
