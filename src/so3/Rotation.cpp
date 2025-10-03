/**
 * \file Rotation.cpp
 * \brief Implementation file for a rotation matrix class.
 * \details Light weight rotation class, intended to be fast, and not to provide
 *          unnecessary functionality.
 *
 * \author Sean Anderson
 */
#include <lgmath/so3/Rotation.hpp>

#include <iostream>
#include <stdexcept>

#include <lgmath/so3/Operations.hpp>

namespace lgmath {
namespace so3 {

Rotation::Rotation() : C_ba_(Eigen::Matrix3d::Identity()) {}

Rotation::Rotation(const Eigen::Matrix3d& C) : C_ba_(C) {
  // Trigger a conditional reprojection, depending on determinant
  this->reproject(false);
}

Rotation::Rotation(const Eigen::Vector3d& aaxis_ab, unsigned int numTerms) {
  C_ba_ = lgmath::so3::vec2rot(aaxis_ab, numTerms);
}

Rotation::Rotation(const Eigen::VectorXd& aaxis_ab) {
  // Throw logic error
  if (aaxis_ab.rows() != 3) {
    throw std::invalid_argument(
        "Tried to initialize a rotation "
        "from a VectorXd that was not dimension 3");
  }

  // Construct using exponential map
  C_ba_ = lgmath::so3::vec2rot(aaxis_ab);
}

const Eigen::Matrix3d& Rotation::matrix() const { return this->C_ba_; }

Eigen::Vector3d Rotation::vec() const {
  return lgmath::so3::rot2vec(this->C_ba_);
}

Rotation Rotation::inverse() const {
  Rotation temp;
  temp.C_ba_ = C_ba_.transpose();
  temp.reproject(
      false);  // Trigger a conditional reprojection, depending on determinant
  return temp;
}

void Rotation::reproject(bool force) {
  // Compute determinant error
  double det_err = fabs(1.0 - this->C_ba_.determinant());
  // Check if matrix is extremely poor and output a warning
  if (det_err > 1e-3) {
    std::cerr << "Warning: SO(3) rotation matrix " << this->C_ba_
              << " has very poor determinant: "
              << this->C_ba_.determinant() << std::endl;
  }
  if (force || det_err > 1e-8) {
    C_ba_ = so3::vec2rot(so3::rot2vec(C_ba_));
  }
}

Rotation& Rotation::operator*=(const Rotation& C_rhs) {
  // Perform operation
  this->C_ba_ = this->C_ba_ * C_rhs.C_ba_;

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
  this->C_ba_ = this->C_ba_ * C_rhs.C_ba_.transpose();

  // Trigger a conditional reprojection, depending on determinant
  this->reproject(false);

  return *this;
}

Rotation Rotation::operator/(const Rotation& C_rhs) const {
  Rotation temp(*this);
  temp /= C_rhs;
  return temp;
}

Eigen::Vector3d Rotation::operator*(
    const Eigen::Ref<const Eigen::Vector3d>& p_a) const {
  return this->C_ba_ * p_a;
}

}  // namespace so3
}  // namespace lgmath

std::ostream& operator<<(std::ostream& out, const lgmath::so3::Rotation& T) {
  out << std::endl << T.matrix() << std::endl;
  return out;
}
