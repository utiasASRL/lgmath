/**
 * \file TransformationWithCovariance.cpp
 * \details Light weight transformation class with added covariance propagation,
 * intended to be fast, and not to provide unnecessary functionality, but still
 * much slower than the base Transformation class due to extra matrix
 * multiplications associated with covariance propagation.  Only use this class
 * if you need covariance.
 *
 * \author Kai van Es
 */
#include <lgmath/se3/TransformationWithCovariance.hpp>

#include <stdexcept>

#include <lgmath/se3/Operations.hpp>
#include <lgmath/so3/Operations.hpp>

namespace lgmath {
namespace se3 {

TransformationWithCovariance::TransformationWithCovariance(
    bool initCovarianceToZero)
    : Transformation(),
      covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
      covarianceSet_(initCovarianceToZero) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Transformation& T, bool initCovarianceToZero)
    : Transformation(T),
      covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
      covarianceSet_(initCovarianceToZero) {}

TransformationWithCovariance::TransformationWithCovariance(
    Transformation&& T, bool initCovarianceToZero)
    : Transformation(T),
      covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
      covarianceSet_(initCovarianceToZero) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Transformation& T, const Eigen::Matrix<double, 6, 6>& covariance)
    : Transformation(T), covariance_(covariance), covarianceSet_(true) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::Matrix4d& T)
    : Transformation(T),
      covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
      covarianceSet_(false) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::Matrix4d& T, const Eigen::Matrix<double, 6, 6>& covariance)
    : Transformation(T), covariance_(covariance), covarianceSet_(true) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::Matrix3d& C_ba, const Eigen::Vector3d& r_ba_ina)
    : Transformation(C_ba, r_ba_ina),
      covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
      covarianceSet_(false) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::Matrix3d& C_ba, const Eigen::Vector3d& r_ba_ina,
    const Eigen::Matrix<double, 6, 6>& covariance)
    : Transformation(C_ba, r_ba_ina),
      covariance_(covariance),
      covarianceSet_(true) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::Matrix<double, 6, 1>& xi_ab, unsigned int numTerms)
    : Transformation(xi_ab, numTerms),
      covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
      covarianceSet_(false) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::Matrix<double, 6, 1>& xi_ab,
    const Eigen::Matrix<double, 6, 6>& covariance, unsigned int numTerms)
    : Transformation(xi_ab, numTerms),
      covariance_(covariance),
      covarianceSet_(true) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::VectorXd& xi_ab)
    : Transformation(xi_ab),
      covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
      covarianceSet_(false) {}

TransformationWithCovariance::TransformationWithCovariance(
    const Eigen::VectorXd& xi_ab, const Eigen::Matrix<double, 6, 6>& covariance)
    : Transformation(xi_ab), covariance_(covariance), covarianceSet_(true) {}

TransformationWithCovariance& TransformationWithCovariance::operator=(
    const Transformation& T) {
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
    Transformation&& T) {
  // Call the assignment operator on the super class, as the internal members
  // are not accessible here
  Transformation::operator=(T);

  // The covarianceSet_ flag is set to false to prevent unintentional bad
  // covariance propagation
  this->covariance_.setZero();
  this->covarianceSet_ = false;

  return (*this);
}

const Eigen::Matrix<double, 6, 6>& TransformationWithCovariance::cov() const {
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
    const Eigen::Matrix<double, 6, 6>& covariance) {
  covariance_ = covariance;
  covarianceSet_ = true;
}

void TransformationWithCovariance::setZeroCovariance() {
  covariance_.setZero();
  covarianceSet_ = true;
}

TransformationWithCovariance TransformationWithCovariance::inverse() const {
  TransformationWithCovariance temp(Transformation::inverse(), false);
  Eigen::Matrix<double, 6, 6> adjointOfInverse = temp.adjoint();
  temp.setCovariance(adjointOfInverse * covariance_ *
                     adjointOfInverse.transpose());
  return temp;
}

TransformationWithCovariance& TransformationWithCovariance::operator*=(
    const TransformationWithCovariance& T_rhs) {
  // The covarianceSet_ flag is only set to true if BOTH transforms have a
  // properly set covariance
  Eigen::Matrix<double, 6, 6> Ad_lhs = Transformation::adjoint();
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
  Eigen::Matrix<double, 6, 6> Ad_lhs_rhs = Transformation::adjoint();
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

}  // namespace se3
}  // namespace lgmath

std::ostream& operator<<(std::ostream& out,
                         const lgmath::se3::TransformationWithCovariance& T) {
  out << std::endl << T.matrix() << std::endl;
  if (T.covarianceSet()) {
    out << std::endl << T.cov() << std::endl;
  } else {
    out << std::endl << "unset covariance" << std::endl;
  }
  return out;
}
