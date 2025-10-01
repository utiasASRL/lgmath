/**
 * \file Rotation.cpp
 * \brief Implementation file for a rotation matrix class.
 * \details Light weight rotation class, intended to be fast, and not to provide
 *          unnecessary functionality.
 *
 * \author Daniil Lisus
 */
#include <lgmath/so2/Rotation.hpp>

#include <stdexcept>

#include <lgmath/so2/Operations.hpp>

namespace lgmath {
namespace so2 {

Rotation::Rotation() : C_ab_(Eigen::Matrix2d::Identity()) {}

Rotation::Rotation(const Eigen::Matrix2d& C) : C_ab_(C) {
  // Force reprojection on construction
  this->reproject(true);
}

Rotation::Rotation(const double angle_ba) {
  C_ab_ = lgmath::so2::vec2rot(angle_ba);
}

Rotation::Rotation(const Eigen::VectorXd& angle_ba) {
  // Throw logic error
  if (angle_ba.rows() != 1) {
    throw std::invalid_argument(
        "Tried to initialize a rotation "
        "from a VectorXd that was not dimension 1");
  }

  // Construct using exponential map
  C_ab_ = lgmath::so2::vec2rot(angle_ba(0));
}

const Eigen::Matrix2d& Rotation::matrix() const { return this->C_ab_; }

double Rotation::vec() const {
  return lgmath::so2::rot2vec(this->C_ab_);
}

Rotation Rotation::inverse() const {
  Rotation temp;
  temp.C_ab_ = C_ab_.transpose();
  temp.reproject(
      false);  // Trigger a conditional reprojection, depending on determinant
  return temp;
}

void Rotation::reproject(bool force) {
  if (force || fabs(1.0 - this->C_ab_.determinant()) > 1e-6) {
    C_ab_ = so2::vec2rot(so2::rot2vec(C_ab_));
  }
}

Rotation& Rotation::operator*=(const Rotation& C_rhs) {
  // Perform operation
  this->C_ab_ = this->C_ab_ * C_rhs.C_ab_;

  // Trigger a conditional reprojection, depending on determinant
  this->reproject(false);

  return *this;
}

Rotation Rotation::operator*(const Rotation& C_rhs) const {
  Rotation temp(*this);
  temp *= C_rhs;
  return temp;
}

Rotation& Rotation::operator/=(const Rotation& C_rhs) {
  // Perform operation
  this->C_ab_ = this->C_ab_ * C_rhs.C_ab_.transpose();

  // Trigger a conditional reprojection, depending on determinant
  this->reproject(false);

  return *this;
}

Rotation Rotation::operator/(const Rotation& C_rhs) const {
  Rotation temp(*this);
  temp /= C_rhs;
  return temp;
}

Eigen::Vector2d Rotation::operator*(
    const Eigen::Ref<const Eigen::Vector2d>& p_a) const {
  return this->C_ab_ * p_a;
}

}  // namespace so2
}  // namespace lgmath

std::ostream& operator<<(std::ostream& out, const lgmath::so2::Rotation& T) {
  out << std::endl << T.matrix() << std::endl;
  return out;
}
