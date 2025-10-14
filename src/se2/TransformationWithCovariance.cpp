/**
 * \file TransformationWithCovariance.cpp
 * \details Light weight transformation class with added covariance propagation,
 * intended to be fast, and not to provide unnecessary functionality, but still
 * much slower than the base Transformation class due to extra matrix
 * multiplications associated with covariance propagation.  Only use this class
 * if you need covariance.
 *
 * \author Daniil Lisus
 */
#include <lgmath/se2/TransformationWithCovariance.hpp>

#include <stdexcept>

#include <lgmath/se2/Operations.hpp>
#include <lgmath/so2/Operations.hpp>
#include <lgmath/se3/TransformationWithCovariance.hpp>

namespace lgmath {
namespace se2 {

TransformationWithCovariance::TransformationWithCovariance(
    bool initCovarianceToZero)
    : Transformation(),
      covariance_(Eigen::Matrix<double, 3, 3>::Zero()),
      covarianceSet_(initCovarianceToZero) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Transformation& T, bool initCovarianceToZero)
    : Transformation(T),
      covariance_(Eigen::Matrix<double, 3, 3>::Zero()),
      covarianceSet_(initCovarianceToZero) {}

TransformationWithCovariance::TransformationWithCovariance(
    Transformation&& T, bool initCovarianceToZero)
    : Transformation(T),
      covariance_(Eigen::Matrix<double, 3, 3>::Zero()),
      covarianceSet_(initCovarianceToZero) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Transformation& T, const Eigen::Matrix<double, 3, 3>& covariance)
    : Transformation(T), covariance_(covariance), covarianceSet_(true) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::Matrix3d& T)
    : Transformation(T),
      covariance_(Eigen::Matrix<double, 3, 3>::Zero()),
      covarianceSet_(false) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::Matrix3d& T, const Eigen::Matrix<double, 3, 3>& covariance)
    : Transformation(T), covariance_(covariance), covarianceSet_(true) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::Matrix2d& C_ba, const Eigen::Vector2d& r_ba_ina)
    : Transformation(C_ba, r_ba_ina),
      covariance_(Eigen::Matrix<double, 3, 3>::Zero()),
      covarianceSet_(false) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::Matrix2d& C_ba, const Eigen::Vector2d& r_ba_ina,
    const Eigen::Matrix<double, 3, 3>& covariance)
    : Transformation(C_ba, r_ba_ina),
      covariance_(covariance),
      covarianceSet_(true) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::Vector3d& xi_ba)
    : Transformation(xi_ba),
      covariance_(Eigen::Matrix<double, 3, 3>::Zero()),
      covarianceSet_(false) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::Vector3d& xi_ba, const Eigen::Matrix<double, 3, 3>& covariance)
    : Transformation(xi_ba), covariance_(covariance), covarianceSet_(true) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::VectorXd& xi_ba)
    : Transformation(xi_ba),
      covariance_(Eigen::Matrix<double, 3, 3>::Zero()),
      covarianceSet_(false) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::VectorXd& xi_ba, const Eigen::Matrix<double, 3, 3>& covariance)
    : Transformation(xi_ba), covariance_(covariance), covarianceSet_(true) {}

TransformationWithCovariance& TransformationWithCovariance::operator=(
    const Transformation& T) noexcept {
  // Call the assignment operator on the super class, as the internal members
  // are not accessible here
  Transformation::operator=(T);

  // The covarianceSet_ flag is set to false to prevent unintentional bad
  // covariance propagation
  this->covariance_.setZero();
  this->covarianceSet_ = false;

  return (*this);
}

TransformationWithCovariance& TransformationWithCovariance::operator=(
    Transformation&& T) noexcept {
  // Call the assignment operator on the super class, as the internal members
  // are not accessible here
  Transformation::operator=(T);

  // The covarianceSet_ flag is set to false to prevent unintentional bad
  // covariance propagation
  this->covariance_.setZero();
  this->covarianceSet_ = false;

  return (*this);
}

const Eigen::Matrix<double, 3, 3>& TransformationWithCovariance::cov() const {
  if (!covarianceSet_) {
    throw std::logic_error(
        "Covariance accessed before being set.  "
        "Use setCovariance or initialize with a covariance.");
  }
  return covariance_;
}

bool TransformationWithCovariance::covarianceSet() const {
  return covarianceSet_;
}

void TransformationWithCovariance::setCovariance(
    const Eigen::Matrix<double, 3, 3>& covariance) {
  covariance_ = covariance;
  covarianceSet_ = true;
}

void TransformationWithCovariance::setZeroCovariance() {
  covariance_.setZero();
  covarianceSet_ = true;
}

TransformationWithCovariance TransformationWithCovariance::inverse() const {
  TransformationWithCovariance temp(Transformation::inverse(), false);
  Eigen::Matrix<double, 3, 3> adjointOfInverse = temp.adjoint();
  temp.setCovariance(adjointOfInverse * covariance_ *
                     adjointOfInverse.transpose());
  return temp;
}

se3::TransformationWithCovariance TransformationWithCovariance::toSE3() const {
  // Use the base class toSE3() method to create the SE(3) transformation
  se3::Transformation base_transform = Transformation::toSE3();
  
  // Create the SE(3) transformation with covariance
  se3::TransformationWithCovariance T_se3(base_transform);

  if (this->covarianceSet()) {
    // Create the SE(3) covariance matrix, filling in the known parts and
    // setting the rest to zero
    Eigen::Matrix<double, 6, 6> cov_se3 = Eigen::Matrix<double, 6, 6>::Zero();
    // Set x, y, yaw covariance
    cov_se3(0, 0) = this->covariance_(0, 0);
    cov_se3(1, 1) = this->covariance_(1, 1);
    cov_se3(5, 5) = this->covariance_(2, 2);
    // Set cross-correlations
    cov_se3(0, 1) = this->covariance_(0, 1);
    cov_se3(1, 0) = this->covariance_(1, 0);
    cov_se3(0, 5) = this->covariance_(0, 2);
    cov_se3(5, 0) = this->covariance_(2, 0);
    cov_se3(1, 5) = this->covariance_(1, 2);
    cov_se3(5, 1) = this->covariance_(2, 1);
    // Set small covariance for z, roll, pitch to avoid singular covariance matrix
    cov_se3(2, 2) = 1e-6;
    cov_se3(3, 3) = 1e-6;
    cov_se3(4, 4) = 1e-6;
    T_se3.setCovariance(cov_se3);
  }

  return T_se3;
}

TransformationWithCovariance& TransformationWithCovariance::operator*=(
    const TransformationWithCovariance& T_rhs) {
  // The covarianceSet_ flag is only set to true if BOTH transforms have a
  // properly set covariance
  Eigen::Matrix<double, 3, 3> Ad_lhs = Transformation::adjoint();
  this->covariance_ =
      this->covariance_ + Ad_lhs * T_rhs.covariance_ * Ad_lhs.transpose();
  this->covarianceSet_ = (this->covarianceSet_ && T_rhs.covarianceSet_);

  // Compound mean transform
  Transformation::operator*=(T_rhs);
  return *this;
}

TransformationWithCovariance& TransformationWithCovariance::operator*=(
    const Transformation& T_rhs) {
  Transformation::operator*=(T_rhs);
  return *this;
}

TransformationWithCovariance& TransformationWithCovariance::operator/=(
    const TransformationWithCovariance& T_rhs) {
  // Note very carefully that we modify the internal transform before taking the
  // adjoint in order to avoid having to convert the rhs covariance explicitly
  Transformation::operator/=(T_rhs);
  Eigen::Matrix<double, 3, 3> Ad_lhs_rhs = Transformation::adjoint();
  this->covariance_ = this->covariance_ +
                      Ad_lhs_rhs * T_rhs.covariance_ * Ad_lhs_rhs.transpose();
  this->covarianceSet_ = (this->covarianceSet_ && T_rhs.covarianceSet_);
  return *this;
}

TransformationWithCovariance& TransformationWithCovariance::operator/=(
    const Transformation& T_rhs) {
  Transformation::operator/=(T_rhs);
  return *this;
}

TransformationWithCovariance operator*(
    TransformationWithCovariance T_lhs,
    const TransformationWithCovariance& T_rhs) {
  T_lhs *= T_rhs;
  return T_lhs;
}

TransformationWithCovariance operator*(TransformationWithCovariance T_lhs,
                                       const Transformation& T_rhs) {
  T_lhs *= T_rhs;
  return T_lhs;
}

TransformationWithCovariance operator*(
    const Transformation& T_lhs, const TransformationWithCovariance& T_rhs) {
  // Convert the Transform to a TransformWithCovariance with perfect certainty
  TransformationWithCovariance temp(T_lhs, true);
  temp *= T_rhs;
  return temp;
}

TransformationWithCovariance operator/(
    TransformationWithCovariance T_lhs,
    const TransformationWithCovariance& T_rhs) {
  T_lhs /= T_rhs;
  return T_lhs;
}

TransformationWithCovariance operator/(TransformationWithCovariance T_lhs,
                                       const Transformation& T_rhs) {
  T_lhs /= T_rhs;
  return T_lhs;
}

TransformationWithCovariance operator/(
    const Transformation& T_lhs, const TransformationWithCovariance& T_rhs) {
  // Convert the Transform to a TransformWithCovariance with perfect certainty
  TransformationWithCovariance temp(T_lhs, true);
  temp /= T_rhs;
  return temp;
}

}  // namespace se2
}  // namespace lgmath

std::ostream& operator<<(std::ostream& out,
                         const lgmath::se2::TransformationWithCovariance& T) {
  out << std::endl << T.matrix() << std::endl;
  if (T.covarianceSet()) {
    out << std::endl << T.cov() << std::endl;
  } else {
    out << std::endl << "unset covariance" << std::endl;
  }
  return out;
}
