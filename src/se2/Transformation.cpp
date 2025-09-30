/**
 * \file Transformation.cpp
 * \brief Implementation file for an SE(2) transformation matrix class.
 * \details Light weight transformation class, intended to be fast, and not to
 * provide unnecessary functionality.
 *
 * \author Daniil Lisus
 */
#include <lgmath/se2/Transformation.hpp>

#include <iostream>
#include <stdexcept>

#include <lgmath/se2/Operations.hpp>
#include <lgmath/so2/Operations.hpp>

namespace lgmath {
namespace se2 {

Transformation::Transformation()
    : C_ab_(Eigen::Matrix2d::Identity()), r_ba_ina_(Eigen::Vector2d::Zero()) {}

Transformation::Transformation(const Eigen::Matrix3d& T)
    : C_ab_(T.block<2, 2>(0, 0)), r_ba_ina_(T.block<2, 1>(0, 2)) {
  // Trigger a conditional reprojection, depending on determinant
  this->reproject();
}

Transformation::Transformation(const Eigen::Matrix2d& C_ab,
                               const Eigen::Vector2d& r_ba_ina) {
  C_ab_ = C_ab;
  // Trigger reprojection
  this->reproject();
  r_ba_ina_ = r_ba_ina;
}

Transformation::Transformation(const Eigen::Vector3d& xi_ba) {
  // Throw logic error
  if (xi_ba.rows() != 3) {
    throw std::invalid_argument(
        "Tried to initialize a transformation "
        "from a VectorXd that was not dimension 6");
  }

  // Construct using exponential map
  lgmath::se2::vec2tran(xi_ba, &C_ab_, &r_ba_ina_);
}

Eigen::Matrix3d Transformation::matrix() const {
  Eigen::Matrix3d T_ab = Eigen::Matrix3d::Identity();
  T_ab.topLeftCorner<2, 2>() = C_ab_;
  T_ab.topRightCorner<2, 1>() = r_ba_ina_;
  return T_ab;
}

const Eigen::Matrix2d& Transformation::C_ab() const { return C_ab_; }

const Eigen::Vector2d Transformation::r_ba_ina() const {
  return r_ba_ina_;
}

Eigen::Vector2d Transformation::r_ab_inb() const {
  return (-1.0) * C_ab_.transpose() * r_ba_ina_;
}

Eigen::Matrix<double, 3, 1> Transformation::vec() const {
  return lgmath::se2::tran2vec(this->C_ab_, this->r_ba_ina_);
}

Transformation Transformation::inverse() const {
  Transformation temp;
  temp.C_ab_ = C_ab_.transpose();
  // Trigger a reprojection
  temp.reproject();
  temp.r_ba_ina_ = (-1.0) * temp.C_ab_ * r_ba_ina_;
  return temp;
}

Eigen::Matrix<double, 3, 3> Transformation::adjoint() const {
  return lgmath::se2::tranAd(C_ab_, r_ba_ina_);
}

void Transformation::reproject() {
  // Note that the translation parameter always belongs to SE(2), but the
  // rotation can incur numerical error that accumulates.
  C_ab_ = so2::vec2rot(so2::rot2vec(C_ab_));
}

Transformation& Transformation::operator*=(const Transformation& T_rhs) {
  // Perform operation
  this->r_ba_ina_ += this->C_ab_ * T_rhs.r_ba_ina_;
  this->C_ab_ = this->C_ab_ * T_rhs.C_ab_;

  // Trigger a reprojection
  this->reproject();

  return *this;
}

Transformation Transformation::operator*(const Transformation& T_rhs) const {
  Transformation temp(*this);
  temp *= T_rhs;
  return temp;
}

Transformation& Transformation::operator/=(const Transformation& T_rhs) {
  // Perform operation
  this->C_ab_ = this->C_ab_ * T_rhs.C_ab_.transpose();
  this->r_ba_ina_ += (-1) * this->C_ab_ * T_rhs.r_ba_ina_;

  // Trigger a reprojection
  this->reproject();

  return *this;
}

Transformation Transformation::operator/(const Transformation& T_rhs) const {
  Transformation temp(*this);
  temp /= T_rhs;
  return temp;
}

Eigen::Vector3d Transformation::operator*(
    const Eigen::Ref<const Eigen::Vector3d>& p_b) const {
  Eigen::Vector3d p_a;
  p_a.head<2>() = C_ab_ * p_b.head<2>() + r_ba_ina_ * p_b[2];
  p_a[2] = p_b[2];
  return p_a;
}

}  // namespace se2
}  // namespace lgmath

std::ostream& operator<<(std::ostream& out,
                         const lgmath::se2::Transformation& T) {
  out << std::endl << T.matrix() << std::endl;
  return out;
}
