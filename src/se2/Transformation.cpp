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
#include <lgmath/so2/Rotation.hpp>
#include <lgmath/se3/Transformation.hpp>

namespace lgmath {
namespace se2 {

Transformation::Transformation()
    : C_ba_(Eigen::Matrix2d::Identity()), r_ab_inb_(Eigen::Vector2d::Zero()) {}

Transformation::Transformation(const Eigen::Matrix3d& T)
    : C_ba_(T.block<2, 2>(0, 0)), r_ab_inb_(T.block<2, 1>(0, 2)) {
  // Trigger a conditional reprojection, depending on determinant
  this->reproject();
}

Transformation::Transformation(const Eigen::Matrix2d& C_ba,
                               const Eigen::Vector2d& r_ba_ina) {
  C_ba_ = C_ba;
  // Trigger reprojection
  this->reproject();
  r_ab_inb_ = - C_ba_ * r_ba_ina;
}

Transformation::Transformation(const Eigen::Matrix<double, 3, 1>& xi_ab) {
  lgmath::se2::vec2tran(xi_ab, &C_ba_, &r_ab_inb_);
}

Transformation::Transformation(const Eigen::VectorXd& xi_ab) {
  // Throw logic error
  if (xi_ab.rows() != 3) {
    throw std::invalid_argument(
        "Tried to initialize a transformation "
        "from a VectorXd that was not dimension 3");
  }

  // Construct using exponential map
  lgmath::se2::vec2tran(xi_ab, &C_ba_, &r_ab_inb_);
}

Eigen::Matrix3d Transformation::matrix() const {
  Eigen::Matrix3d T_ba = Eigen::Matrix3d::Identity();
  T_ba.topLeftCorner<2, 2>() = C_ba_;
  T_ba.topRightCorner<2, 1>() = r_ab_inb_;
  return T_ba;
}

const Eigen::Matrix2d& Transformation::C_ba() const { return C_ba_; }

Eigen::Vector2d Transformation::r_ba_ina() const {
  return -C_ba_.transpose() * r_ab_inb_;
}

const Eigen::Vector2d& Transformation::r_ab_inb() const { return r_ab_inb_; }

Eigen::Matrix<double, 3, 1> Transformation::vec() const {
  return lgmath::se2::tran2vec(C_ba_, r_ab_inb_);
}

Transformation Transformation::inverse() const {
  Transformation temp;
  temp.C_ba_ = C_ba_.transpose();
  // Trigger a reprojection
  temp.reproject();
  temp.r_ab_inb_ = -temp.C_ba_ * r_ab_inb_;
  return temp;
}

Eigen::Matrix<double, 3, 3> Transformation::adjoint() const {
  return lgmath::se2::tranAd(C_ba_, r_ab_inb_);
}

void Transformation::reproject() {
  // Note that the translation parameter always belongs to SE(2), but the
  // rotation can incur numerical error that accumulates.
  so2::Rotation rotation(C_ba_);
  rotation.reproject();
  C_ba_ = rotation.matrix();
}

Transformation& Transformation::operator*=(const Transformation& T_rhs) {
  // Perform operation
  this->r_ab_inb_ += this->C_ba_ * T_rhs.r_ab_inb_;
  this->C_ba_ = this->C_ba_ * T_rhs.C_ba_;

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
  this->C_ba_ = this->C_ba_ * T_rhs.C_ba_.transpose();
  this->r_ab_inb_ += (-1) * this->C_ba_ * T_rhs.r_ab_inb_;

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
    const Eigen::Ref<const Eigen::Vector3d>& p_a) const {
  Eigen::Vector3d p_b;
  p_b.head<2>() = C_ba_ * p_a.head<2>() + r_ab_inb_ * p_a[2];
  p_b[2] = p_a[2];
  return p_b;
}

se3::Transformation Transformation::toSE3() const {
  // Create a 4x4 transformation matrix in SE(3)
  Eigen::Matrix4d T_ba_3d = Eigen::Matrix4d::Identity();
  // Fill in rotation part
  T_ba_3d.block<2, 2>(0, 0) = C_ba_;
  
  // Fill in translation part
  T_ba_3d.block<2, 1>(0, 3) = r_ab_inb_;
  
  return se3::Transformation(T_ba_3d);
}
}  // namespace se2
}  // namespace lgmath

std::ostream& operator<<(std::ostream& out,
                         const lgmath::se2::Transformation& T) {
  out << std::endl << T.matrix() << std::endl;
  return out;
}
