/**
 * \file Transformation.cpp
 * \brief Implementation file for a transformation matrix class.
 * \details Light weight transformation class, intended to be fast, and not to
 * provide unnecessary functionality.
 *
 * \author Sean Anderson
 */
#include <lgmath/se3/Transformation.hpp>

#include <iostream>
#include <stdexcept>

#include <lgmath/se3/Operations.hpp>
#include <lgmath/so3/Operations.hpp>
#include <lgmath/so3/Rotation.hpp>
#include <lgmath/se2/Transformation.hpp>

namespace lgmath {
namespace se3 {

Transformation::Transformation()
    : C_ba_(Eigen::Matrix3d::Identity()), r_ab_inb_(Eigen::Vector3d::Zero()) {}

Transformation::Transformation(const Eigen::Matrix4d& T)
    : C_ba_(T.block<3, 3>(0, 0)), r_ab_inb_(T.block<3, 1>(0, 3)) {
  // Trigger a conditional reprojection, depending on determinant
  this->reproject(false);
}

Transformation::Transformation(const Eigen::Matrix3d& C_ba,
                               const Eigen::Vector3d& r_ba_ina) {
  C_ba_ = C_ba;
  // Trigger a conditional reprojection, depending on determinant
  this->reproject(false);
  r_ab_inb_ = -C_ba_ * r_ba_ina;
}

Transformation::Transformation(const Eigen::Matrix<double, 6, 1>& xi_ab,
                               unsigned int numTerms) {
  lgmath::se3::vec2tran(xi_ab, &C_ba_, &r_ab_inb_, numTerms);
}

Transformation::Transformation(const Eigen::VectorXd& xi_ab) {
  // Throw logic error
  if (xi_ab.rows() != 6) {
    throw std::invalid_argument(
        "Tried to initialize a transformation "
        "from a VectorXd that was not dimension 6");
  }

  // Construct using exponential map
  lgmath::se3::vec2tran(xi_ab, &C_ba_, &r_ab_inb_, 0);
}

Eigen::Matrix4d Transformation::matrix() const {
  Eigen::Matrix4d T_ba = Eigen::Matrix4d::Identity();
  T_ba.topLeftCorner<3, 3>() = C_ba_;
  T_ba.topRightCorner<3, 1>() = r_ab_inb_;
  return T_ba;
}

const Eigen::Matrix3d& Transformation::C_ba() const { return C_ba_; }

Eigen::Vector3d Transformation::r_ba_ina() const {
  return -C_ba_.transpose() * r_ab_inb_;
}

const Eigen::Vector3d& Transformation::r_ab_inb() const { return r_ab_inb_; }

Eigen::Matrix<double, 6, 1> Transformation::vec() const {
  return lgmath::se3::tran2vec(C_ba_, r_ab_inb_);
}

Transformation Transformation::inverse() const {
  Transformation temp;
  temp.C_ba_ = C_ba_.transpose();
  // Trigger a conditional reprojection, depending on determinant
  temp.reproject(false);
  temp.r_ab_inb_ = -temp.C_ba_ * r_ab_inb_;
  return temp;
}

Eigen::Matrix<double, 6, 6> Transformation::adjoint() const {
  return lgmath::se3::tranAd(C_ba_, r_ab_inb_);
}

void Transformation::reproject(bool force) {
  // Note that the translation parameter always belongs to SE(3), but the
  // rotation can incur numerical error that accumulates.
  so3::Rotation rotation(C_ba_);
  rotation.reproject(force);
  C_ba_ = rotation.matrix();
}

Transformation& Transformation::operator*=(const Transformation& T_rhs) {
  // Perform operation
  this->r_ab_inb_ += this->C_ba_ * T_rhs.r_ab_inb_;
  this->C_ba_ = this->C_ba_ * T_rhs.C_ba_;

  // Trigger a conditional reprojection, depending on determinant
  this->reproject(false);

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

  // Trigger a conditional reprojection, depending on determinant
  this->reproject(false);

  return *this;
}

Transformation Transformation::operator/(const Transformation& T_rhs) const {
  Transformation temp(*this);
  temp /= T_rhs;
  return temp;
}

Eigen::Vector4d Transformation::operator*(
    const Eigen::Ref<const Eigen::Vector4d>& p_a) const {
  Eigen::Vector4d p_b;
  p_b.head<3>() = C_ba_ * p_a.head<3>() + r_ab_inb_ * p_a[3];
  p_b[3] = p_a[3];
  return p_b;
}

se2::Transformation Transformation::toSE2() const {
  // Check norm of z, roll, pitch components to warn if they are large
  if (this->vec().segment<3>(2).norm() > 1e-3) {
    std::cerr << "Warning: SE(3) has significant z, roll, or pitch component. "
              << "Projecting to SE(2) will discard this information."
              << std::endl;
  }
  return se2::Transformation(C_ba_.block<2, 2>(0, 0), r_ba_ina().head<2>());
}
}  // namespace se3
}  // namespace lgmath

std::ostream& operator<<(std::ostream& out,
                         const lgmath::se3::Transformation& T) {
  out << std::endl << T.matrix() << std::endl;
  return out;
}
