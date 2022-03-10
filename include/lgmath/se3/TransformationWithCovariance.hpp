/**
 * \file TransformationWithCovariance.hpp
 * \brief Header file for a transformation matrix class with associated
 * covariance.
 * \details Light weight transformation class with added covariance propagation,
 * intended to be fast, and not to provide unnecessary functionality, but still
 * much slower than the base Transformation class due to extra matrix
 * multiplications associated with covariance propagation.  Only use this class
 * if you need covariance.
 *
 * \author Kai van Es
 */
#pragma once

#include <Eigen/Core>

#include <lgmath/se3/Transformation.hpp>

namespace lgmath {
namespace se3 {

class TransformationWithCovariance : public Transformation {
 public:
  /** \brief Default constructor */
  TransformationWithCovariance(bool initCovarianceToZero = false);

  /** \brief Copy constructor. */
  TransformationWithCovariance(const TransformationWithCovariance&) = default;

  /** \brief Move constructor. */
  TransformationWithCovariance(TransformationWithCovariance&& T) = default;

  /** \brief Copy constructor from basic Transformation */
  TransformationWithCovariance(const Transformation& T,
                               bool initCovarianceToZero = false);

  /** \brief Move constructor from basic Transformation */
  TransformationWithCovariance(Transformation&& T,
                               bool initCovarianceToZero = false);

  /** \brief Copy constructor from basic Transformation, with covariance */
  TransformationWithCovariance(const Transformation& T,
                               const Eigen::Matrix<double, 6, 6>& covariance);

  /** \brief Constructor */
  TransformationWithCovariance(const Eigen::Matrix4d& T);

  /** \brief Constructor with covariance */
  TransformationWithCovariance(const Eigen::Matrix4d& T,
                               const Eigen::Matrix<double, 6, 6>& covariance);

  /**
   * \brief Constructor.
   * The transformation will be T_ba = [C_ba, -C_ba*r_ba_ina; 0 0 0 1]
   */
  TransformationWithCovariance(const Eigen::Matrix3d& C_ba,
                               const Eigen::Vector3d& r_ba_ina);

  /**
   * \brief Constructor with covariance. The transformation will be
   * T_ba = [C_ba, -C_ba*r_ba_ina; 0 0 0 1]
   */
  TransformationWithCovariance(const Eigen::Matrix3d& C_ba,
                               const Eigen::Vector3d& r_ba_ina,
                               const Eigen::Matrix<double, 6, 6>& covariance);

  /**
   * \brief Constructor.
   * The transformation will be T_ba = vec2tran(xi_ab)
   */
  TransformationWithCovariance(const Eigen::Matrix<double, 6, 1>& xi_ab,
                               unsigned int numTerms = 0);

  /**
   * \brief Constructor with covariance.
   * The transformation will be T_ba = vec2tran(xi_ab)
   */
  TransformationWithCovariance(const Eigen::Matrix<double, 6, 1>& xi_ab,
                               const Eigen::Matrix<double, 6, 6>& covariance,
                               unsigned int numTerms = 0);

  /**
   * \brief Constructor.
   * The transformation will be T_ba = vec2tran(xi_ab), xi_ab must be 6x1
   */
  TransformationWithCovariance(const Eigen::VectorXd& xi_ab);

  /**
   * \brief Constructor.
   * The transformation will be T_ba = vec2tran(xi_ab), xi_ab must be 6x1
   */
  TransformationWithCovariance(const Eigen::VectorXd& xi_ab,
                               const Eigen::Matrix<double, 6, 6>& covariance);

  /** \brief Destructor. Default implementation. */
  ~TransformationWithCovariance() = default;

  /** \brief Copy assignment operator. */
  TransformationWithCovariance& operator=(const TransformationWithCovariance&) =
      default;

  /** \brief Move assignment operator. */
  TransformationWithCovariance& operator=(TransformationWithCovariance&& T) =
      default;

  /**
   * \brief Copy assignment operator from basic Transform.
   * \details This assignment resets the covariance to the uninitialized state.
   * You must manually call setZeroCovariance() or
   * setCovariance(const Eigen::Matrix6d&) before querying it with the public
   * method cov(), or an exception will be thrown.
   */
  virtual TransformationWithCovariance& operator=(const Transformation& T);

  /**
   * \brief Move assignment operator from basic Transform.
   * \details This assignment resets the covariance to the uninitialized state.
   * You must manually call setZeroCovariance() or
   * setCovariance(const Eigen::Matrix6d&) before querying it with the public
   * method cov(), or an exception will be thrown.
   */
  virtual TransformationWithCovariance& operator=(Transformation&& T);

  /** \brief Gets the underlying covariance matrix */
  const Eigen::Matrix<double, 6, 6>& cov() const;

  /**
   * \brief Returns whether or not a covariance has been set. If it is unset,
   * then querying it with the public method cov() will throw an exception.
   */
  bool covarianceSet() const;

  /** \brief Sets the underlying rotation matrix */
  void setCovariance(const Eigen::Matrix<double, 6, 6>& covariance);

  /**
   * \brief Sets the underlying rotation matrix to the 6x6 zero matrix (perfect
   * certainty)
   */
  void setZeroCovariance();

  /** \brief Get the inverse matrix */
  TransformationWithCovariance inverse() const;

  /** \brief In-place right-hand side multiply T_rhs. */
  TransformationWithCovariance& operator*=(
      const TransformationWithCovariance& T_rhs);

  /**
   * \brief In-place right-hand side multiply basic (certain) T_rhs
   * \note Assumes that the Transformation matrix has perfect certainty
   */
  TransformationWithCovariance& operator*=(const Transformation& T_rhs);

  /**
   * \brief In-place right-hand side multiply this matrix by the inverse of
   * T_rhs
   */
  TransformationWithCovariance& operator/=(
      const TransformationWithCovariance& T_rhs);

  /**
   * \brief In-place right-hand side multiply this matrix by the inverse of a
   * basic (certain) T_rhs
   *
   * \note Assumes that the Transformation matrix has perfect certainty
   */
  TransformationWithCovariance& operator/=(const Transformation& T_rhs);

 private:
  /** \brief Covariance */
  Eigen::Matrix<double, 6, 6> covariance_;

  /** \brief Covariance flag */
  bool covarianceSet_;
};

/**
 * \brief Multiplication of TransformWithCovariance by TransformWithCovariance
 */
TransformationWithCovariance operator*(
    TransformationWithCovariance T_lhs,
    const TransformationWithCovariance& T_rhs);

/**
 * \brief Multiplication of TransformWithCovariance by Transform
 * \note Assumes that the Transformation matrix has perfect certainty
 */
TransformationWithCovariance operator*(TransformationWithCovariance T_lhs,
                                       const Transformation& T_rhs);

/**
 * \brief Multiplication of Transform by TransformWithCovariance
 * \note Assumes that the Transformation matrix has perfect certainty
 */
TransformationWithCovariance operator*(
    const Transformation& T_lhs, const TransformationWithCovariance& T_rhs);

/**
 * \brief Multiplication of TransformWithCovariance by inverse
 * TransformWithCovariance
 */
TransformationWithCovariance operator/(
    TransformationWithCovariance T_lhs,
    const TransformationWithCovariance& T_rhs);

/**
 * \brief Multiplication of TransformWithCovariance by inverse Transform
 * \note Assumes that the Transformation matrix has perfect certainty
 */
TransformationWithCovariance operator/(TransformationWithCovariance T_lhs,
                                       const Transformation& T_rhs);

/**
 * \brief Multiplication of Transform by inverse TransformWithCovariance
 * \note Assumes that the Transformation matrix has perfect certainty
 */
TransformationWithCovariance operator/(
    const Transformation& T_lhs, const TransformationWithCovariance& T_rhs);

}  // namespace se3
}  // namespace lgmath

/** \brief print transformation */
std::ostream& operator<<(std::ostream& out,
                         const lgmath::se3::TransformationWithCovariance& T);
