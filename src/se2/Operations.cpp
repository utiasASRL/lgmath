/**
 * \file Operations.cpp
 * \brief Implementation file for the SE2 Lie Group math functions.
 * \details These namespace functions provide implementations of the special
 * Euclidean (SE) in 2D Lie group functions that we commonly use in robotics.
 *
 * \author Daniil Lisus
 */
#include <lgmath/se2/Operations.hpp>

#include <stdio.h>
#include <stdexcept>

#include <Eigen/Dense>

#include <lgmath/so2/Operations.hpp>

namespace lgmath {
namespace se2 {

Eigen::Matrix3d hat(const Eigen::Vector2d& rho, const double angle) {
  Eigen::Matrix3d mat = Eigen::Matrix3d::Zero();
  mat.topLeftCorner<2, 2>() = so2::hat(angle);
  mat.topRightCorner<2, 1>() = rho;
  return mat;
}

Eigen::Matrix3d hat(const Eigen::Matrix<double, 3, 1>& xi) {
  return hat(xi.head<2>(), xi(2));
}

Eigen::Matrix<double, 3, 3> curlyhat(const Eigen::Vector2d& rho,
                                     const double angle) {
  Eigen::Matrix<double, 3, 3> mat = Eigen::Matrix3d::Zero();
  mat.topLeftCorner<2, 2>() = so2::hat(angle);
  Eigen::Matrix2d S;
  S << 0, -1, 1, 0;
  mat.topRightCorner<2, 1>() = -S * rho;
  return mat;
}

Eigen::Matrix<double, 3, 3> curlyhat(const Eigen::Matrix<double, 3, 1>& xi) {
  return curlyhat(xi.head<2>(), xi(2));
}

Eigen::Matrix<double, 3, 3> point2fs(const Eigen::Vector2d& p, double scale) {
  Eigen::Matrix<double, 3, 3> fs = Eigen::Matrix3d::Zero();
  Eigen::Matrix2d S;
  S << 0, -1, 1, 0;
  fs.topLeftCorner<2, 2>() = scale * Eigen::Matrix2d::Identity();
  fs.topRightCorner<2, 1>() = S * p;
  return fs;
}

Eigen::Matrix<double, 3, 3> point2sf(const Eigen::Vector2d& p, double scale) {
  Eigen::Matrix<double, 3, 3> sf = Eigen::Matrix3d::Zero();
  Eigen::Matrix2d S;
  S << 0, -1, 1, 0;
  sf.topRightCorner<2, 1>() = p;
  sf.bottomLeftCorner<1, 2>() = - (S * p).transpose();
  return sf;
}

void vec2tran(const Eigen::Vector3d& xi_ba,
                         Eigen::Matrix2d* out_C_ab,
                         Eigen::Vector2d* out_r_ba_ina) {
  // Check pointers
  if (out_C_ab == NULL) {
    throw std::invalid_argument("Null pointer out_C_ab in vec2tran");
  }
  if (out_r_ba_ina == NULL) {
    throw std::invalid_argument(
        "Null pointer out_r_ba_ina in vec2tran");
  }

  // Normal analytical solution, we don't need to worry about
  // numerical issues with so2 since its all simple sin/cos computations.
  // Form Jacobian to transform translation
  Eigen::Matrix2d Gamma1 = se2::vec2Gamma1(xi_ba(2));

  // Get rotation matrix
  *out_C_ab = so2::vec2rot(xi_ba(2));

  // Convert rho_ba (twist-translation) to r_ba_ina
  *out_r_ba_ina = Gamma1 * xi_ba.head<2>();
}

Eigen::Matrix3d vec2tran(const Eigen::Matrix<double, 3, 1>& xi_ba) {
  // Get rotation and translation
  Eigen::Matrix2d C_ab;
  Eigen::Vector2d r_ba_ina;
  vec2tran(xi_ba, &C_ab, &r_ba_ina);

  // Fill output
  Eigen::Matrix3d T_ab = Eigen::Matrix3d::Identity();
  T_ab.topLeftCorner<2, 2>() = C_ab;
  T_ab.topRightCorner<2, 1>() = r_ba_ina;
  return T_ab;
}

Eigen::Matrix<double, 3, 1> tran2vec(const Eigen::Matrix2d& C_ab,
                                     const Eigen::Vector2d& r_ba_ina) {
  // Init
  Eigen::Matrix<double, 3, 1> xi_ba;

  // Get angle from rotation matrix
  double angle_ba = so2::rot2vec(C_ab);

  // Get twist-translation vector using Jacobian
  Eigen::Vector2d rho_ba = se2::vec2Gamma1(angle_ba).inverse() * r_ba_ina;

  // Return se2 algebra vector
  xi_ba << rho_ba, angle_ba;
  return xi_ba;
}

Eigen::Matrix<double, 3, 1> tran2vec(const Eigen::Matrix3d& T_ab) {
  return tran2vec(T_ab.topLeftCorner<2, 2>(), T_ab.topRightCorner<2, 1>());
}

Eigen::Matrix<double, 3, 3> tranAd(const Eigen::Matrix2d& C_ab,
                                   const Eigen::Vector2d& r_ba_ina) {
  Eigen::Matrix<double, 3, 3> adjoint_T_ab = Eigen::Matrix3d::Identity();
  adjoint_T_ab.topLeftCorner<2, 2>() = C_ab;
  Eigen::Matrix2d S;
  S << 0, -1, 1, 0;
  adjoint_T_ab.topRightCorner<2, 1>() = -S * r_ba_ina;
  return adjoint_T_ab;
}

Eigen::Matrix<double, 3, 3> tranAd(const Eigen::Matrix3d& T_ab) {
  return tranAd(T_ab.topLeftCorner<2, 2>(), T_ab.topRightCorner<2, 1>());
}

Eigen::Matrix2d vec2Gamma1(const double angle) {
  Eigen::Matrix2d Gamma1 = Eigen::Matrix2d::Identity();
  // If angle is very small, Gamma1 is Identity
  if (std::abs(angle) > 1e-12) {
    Eigen::Matrix2d S;
    S << 0, -1, 1, 0;
    Gamma1 = (sin(angle) / angle) * Eigen::Matrix2d::Identity() +
             ((1 - cos(angle)) / angle) * S;
  }
  return Gamma1;
}

Eigen::Matrix2d vec2Gamma1(const Eigen::Matrix<double, 3, 1>& xi_ba) {
  return vec2Gamma1(xi_ba(2));
}

Eigen::Matrix2d vec2Gamma2(const double angle) {
  Eigen::Matrix2d Gamma2 = Eigen::Matrix2d::Zero();
  // As angle approaches 0, Gamma2 approaches 0.5*I
  if (std::abs(angle) < 1e-12) {
    Gamma2 = 0.5 * Eigen::Matrix2d::Identity();
  } else {
    Eigen::Matrix2d S;
    S << 0, -1, 1, 0;
    Gamma2 = ((1 - cos(angle)) / (angle * angle)) * Eigen::Matrix2d::Identity() +
             ((angle - sin(angle)) / (angle * angle)) * S;
  }
  return Gamma2;
}

Eigen::Matrix2d vec2Gamma2(const Eigen::Matrix<double, 3, 1>& xi_ba) {
  return vec2Gamma2(xi_ba(2));
}

Eigen::Matrix<double, 3, 3> vec2jac(const Eigen::Vector2d& rho_ba,
                                    const double angle_ba) {
  // Init
  Eigen::Matrix<double, 3, 3> J_ab = Eigen::Matrix3d::Identity();
  Eigen::Matrix2d Gamma1 = se2::vec2Gamma1(angle_ba);
  Eigen::Matrix2d Gamma2 = se2::vec2Gamma2(angle_ba);
  Eigen::Matrix2d S;
  S << 0, -1, 1, 0;

  // Fill in blocks
  J_ab.topLeftCorner<2, 2>() = Gamma1;
  J_ab.topRightCorner<2, 1>() = - S * Gamma2 * rho_ba;

  return J_ab;
}

Eigen::Matrix<double, 3, 3> vec2jac(const Eigen::Matrix<double, 3, 1>& xi_ba) {
  return vec2jac(xi_ba.head<2>(), xi_ba(2));
}

Eigen::Matrix<double, 3, 3> vec2jacinv(const Eigen::Vector2d& rho_ba,
                                       const double angle_ba) {
  // Init
  Eigen::Matrix<double, 3, 3> J_ab_inv = Eigen::Matrix3d::Identity();
  Eigen::Matrix2d Gamma1_inv = se2::vec2Gamma1(angle_ba).inverse();
  Eigen::Matrix2d Gamma2 = se2::vec2Gamma2(angle_ba);
  Eigen::Matrix2d S;
  S << 0, -1, 1, 0;

  // Fill in blocks
  J_ab_inv.topLeftCorner<2, 2>() = Gamma1_inv;
  J_ab_inv.topRightCorner<2, 1>() = Gamma1_inv * S * Gamma2 * rho_ba;

  return J_ab_inv;
}

Eigen::Matrix<double, 3, 3> vec2jacinv(const Eigen::Matrix<double, 3, 1>& xi_ba) {
  return vec2jacinv(xi_ba.head<2>(), xi_ba(2));
}

}  // namespace se2
}  // namespace lgmath
