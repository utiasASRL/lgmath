/**
 * \file Operations.cpp
 * \brief Implementation file for the SE3 Lie Group math functions.
 * \details These namespace functions provide implementations of the special
 * Euclidean (SE) Lie group functions that we commonly use in robotics.
 *
 * \author Sean Anderson
 */
#include <lgmath/se3/Operations.hpp>

#include <stdio.h>
#include <stdexcept>

#include <Eigen/Dense>

#include <lgmath/so3/Operations.hpp>

namespace lgmath {
namespace se3 {

Eigen::Matrix4d hat(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis) {
  Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
  mat.topLeftCorner<3, 3>() = so3::hat(aaxis);
  mat.topRightCorner<3, 1>() = rho;
  return mat;
}

Eigen::Matrix4d hat(const Eigen::Matrix<double, 6, 1>& xi) {
  return hat(xi.head<3>(), xi.tail<3>());
}

Eigen::Matrix<double, 6, 6> curlyhat(const Eigen::Vector3d& rho,
                                     const Eigen::Vector3d& aaxis) {
  Eigen::Matrix<double, 6, 6> mat = Eigen::Matrix<double, 6, 6>::Zero();
  mat.topLeftCorner<3, 3>() = mat.bottomRightCorner<3, 3>() = so3::hat(aaxis);
  mat.topRightCorner<3, 3>() = so3::hat(rho);
  return mat;
}

Eigen::Matrix<double, 6, 6> curlyhat(const Eigen::Matrix<double, 6, 1>& xi) {
  return curlyhat(xi.head<3>(), xi.tail<3>());
}

Eigen::Matrix<double, 4, 6> point2fs(const Eigen::Vector3d& p, double scale) {
  Eigen::Matrix<double, 4, 6> mat = Eigen::Matrix<double, 4, 6>::Zero();
  mat.topLeftCorner<3, 3>() = scale * Eigen::Matrix3d::Identity();
  mat.topRightCorner<3, 3>() = -so3::hat(p);
  return mat;
}

Eigen::Matrix<double, 6, 4> point2sf(const Eigen::Vector3d& p, double scale) {
  Eigen::Matrix<double, 6, 4> mat = Eigen::Matrix<double, 6, 4>::Zero();
  mat.bottomLeftCorner<3, 3>() = -so3::hat(p);
  mat.topRightCorner<3, 1>() = p;
  return mat;
}

void vec2tran_analytical(const Eigen::Vector3d& rho_ba,
                         const Eigen::Vector3d& aaxis_ba,
                         Eigen::Matrix3d* out_C_ab,
                         Eigen::Vector3d* out_r_ba_ina) {
  // Check pointers
  if (out_C_ab == nullptr) {
    throw std::invalid_argument("Null pointer out_C_ab in vec2tran_analytical");
  }
  if (out_r_ba_ina == nullptr) {
    throw std::invalid_argument(
        "Null pointer out_r_ba_ina in vec2tran_analytical");
  }

  if (aaxis_ba.norm() < 1e-12) {
    // If angle is very small, rotation is Identity
    *out_C_ab = Eigen::Matrix3d::Identity();
    *out_r_ba_ina = rho_ba;
  } else {
    // Normal analytical solution
    Eigen::Matrix3d J_ab;

    // Use rotation identity involving jacobian, as we need it to
    // convert rho_ba to the proper translation
    so3::vec2rot(aaxis_ba, out_C_ab, &J_ab);

    // Convert rho_ba (twist-translation) to r_ba_ina
    *out_r_ba_ina = J_ab * rho_ba;
  }
}

void vec2tran_numerical(const Eigen::Vector3d& rho_ba,
                        const Eigen::Vector3d& aaxis_ba,
                        Eigen::Matrix3d* out_C_ab,
                        Eigen::Vector3d* out_r_ba_ina, unsigned int numTerms) {
  // Check pointers
  if (out_C_ab == nullptr) {
    throw std::invalid_argument("Null pointer out_C_ab in vec2tran_numerical");
  }
  if (out_r_ba_ina == nullptr) {
    throw std::invalid_argument(
        "Null pointer out_r_ba_ina in vec2tran_numerical");
  }

  // Init 4x4 transformation
  Eigen::Matrix4d T_ab = Eigen::Matrix4d::Identity();

  // Incremental variables
  Eigen::Matrix<double, 6, 1> xi_ba;
  xi_ba << rho_ba, aaxis_ba;
  Eigen::Matrix4d x_small = se3::hat(xi_ba);
  Eigen::Matrix4d x_small_n = Eigen::Matrix4d::Identity();

  // Loop over sum up to the specified numTerms
  for (unsigned int n = 1; n <= numTerms; n++) {
    x_small_n = x_small_n * x_small / double(n);
    T_ab += x_small_n;
  }

  // Fill output
  *out_C_ab = T_ab.topLeftCorner<3, 3>();
  *out_r_ba_ina = T_ab.topRightCorner<3, 1>();
}

void vec2tran(const Eigen::Matrix<double, 6, 1>& xi_ba,
              Eigen::Matrix3d* out_C_ab, Eigen::Vector3d* out_r_ba_ina,
              unsigned int numTerms) {
  if (numTerms == 0) {
    // Analytical solution
    vec2tran_analytical(xi_ba.head<3>(), xi_ba.tail<3>(), out_C_ab,
                        out_r_ba_ina);
  } else {
    // Numerical solution (good for testing the analytical solution)
    vec2tran_numerical(xi_ba.head<3>(), xi_ba.tail<3>(), out_C_ab, out_r_ba_ina,
                       numTerms);
  }
}

Eigen::Matrix4d vec2tran(const Eigen::Matrix<double, 6, 1>& xi_ba,
                         unsigned int numTerms) {
  // Get rotation and translation
  Eigen::Matrix3d C_ab;
  Eigen::Vector3d r_ba_ina;
  vec2tran(xi_ba, &C_ab, &r_ba_ina, numTerms);

  // Fill output
  Eigen::Matrix4d T_ab = Eigen::Matrix4d::Identity();
  T_ab.topLeftCorner<3, 3>() = C_ab;
  T_ab.topRightCorner<3, 1>() = r_ba_ina;
  return T_ab;
}

Eigen::Matrix<double, 6, 1> tran2vec(const Eigen::Matrix3d& C_ab,
                                     const Eigen::Vector3d& r_ba_ina) {
  // Init
  Eigen::Matrix<double, 6, 1> xi_ba;

  // Get axis angle from rotation matrix
  Eigen::Vector3d aaxis_ba = so3::rot2vec(C_ab);

  // Get twist-translation vector using Jacobian
  Eigen::Vector3d rho_ba = so3::vec2jacinv(aaxis_ba) * r_ba_ina;

  // Return se3 algebra vector
  xi_ba << rho_ba, aaxis_ba;
  return xi_ba;
}

Eigen::Matrix<double, 6, 1> tran2vec(const Eigen::Matrix4d& T_ab) {
  return tran2vec(T_ab.topLeftCorner<3, 3>(), T_ab.topRightCorner<3, 1>());
}

Eigen::Matrix<double, 6, 6> tranAd(const Eigen::Matrix3d& C_ab,
                                   const Eigen::Vector3d& r_ba_ina) {
  Eigen::Matrix<double, 6, 6> adjoint_T_ab =
      Eigen::Matrix<double, 6, 6>::Zero();
  adjoint_T_ab.topLeftCorner<3, 3>() = adjoint_T_ab.bottomRightCorner<3, 3>() =
      C_ab;
  adjoint_T_ab.topRightCorner<3, 3>() = so3::hat(r_ba_ina) * C_ab;
  return adjoint_T_ab;
}

Eigen::Matrix<double, 6, 6> tranAd(const Eigen::Matrix4d& T_ab) {
  return tranAd(T_ab.topLeftCorner<3, 3>(), T_ab.topRightCorner<3, 1>());
}

Eigen::Matrix3d vec2Q(const Eigen::Vector3d& rho_ba,
                      const Eigen::Vector3d& aaxis_ba) {
  // Construct scalar terms
  const double ang = aaxis_ba.norm();
  const double ang2 = ang * ang;
  const double ang3 = ang2 * ang;
  const double ang4 = ang3 * ang;
  const double ang5 = ang4 * ang;
  const double cang = cos(ang);
  const double sang = sin(ang);
  const double m2 = (ang - sang) / ang3;
  const double m3 = (1.0 - 0.5 * ang2 - cang) / ang4;
  const double m4 = 0.5 * (m3 - 3 * (ang - sang - ang3 / 6) / ang5);

  // Construct matrix terms
  Eigen::Matrix3d rx = so3::hat(rho_ba);
  Eigen::Matrix3d px = so3::hat(aaxis_ba);
  Eigen::Matrix3d pxrx = px * rx;
  Eigen::Matrix3d rxpx = rx * px;
  Eigen::Matrix3d pxrxpx = pxrx * px;

  // Construct Q matrix
  return 0.5 * rx + m2 * (pxrx + rxpx + pxrxpx) -
         m3 * (px * pxrx + rxpx * px - 3 * pxrxpx) -
         m4 * (pxrxpx * px + px * pxrxpx);
}

Eigen::Matrix3d vec2Q(const Eigen::Matrix<double, 6, 1>& xi_ba) {
  return vec2Q(xi_ba.head<3>(), xi_ba.tail<3>());
}

Eigen::Matrix<double, 6, 6> vec2jac(const Eigen::Vector3d& rho_ba,
                                    const Eigen::Vector3d& aaxis_ba) {
  // Init
  Eigen::Matrix<double, 6, 6> J_ab = Eigen::Matrix<double, 6, 6>::Zero();

  if (aaxis_ba.norm() < 1e-12) {
    // If angle is very small, so3 jacobian is Identity
    J_ab.topLeftCorner<3, 3>() = J_ab.bottomRightCorner<3, 3>() =
        Eigen::Matrix3d::Identity();
    J_ab.topRightCorner<3, 3>() = 0.5 * so3::hat(rho_ba);
  } else {
    // General analytical scenario
    J_ab.topLeftCorner<3, 3>() = J_ab.bottomRightCorner<3, 3>() =
        so3::vec2jac(aaxis_ba);
    J_ab.topRightCorner<3, 3>() = se3::vec2Q(rho_ba, aaxis_ba);
  }
  return J_ab;
}

Eigen::Matrix<double, 6, 6> vec2jac(const Eigen::Matrix<double, 6, 1>& xi_ba,
                                    unsigned int numTerms) {
  if (numTerms == 0) {
    // Analytical solution
    return vec2jac(xi_ba.head<3>(), xi_ba.tail<3>());
  } else {
    // Numerical solution (good for testing the analytical solution)
    Eigen::Matrix<double, 6, 6> J_ab = Eigen::Matrix<double, 6, 6>::Identity();

    // Incremental variables
    Eigen::Matrix<double, 6, 6> x_small = se3::curlyhat(xi_ba);
    Eigen::Matrix<double, 6, 6> x_small_n =
        Eigen::Matrix<double, 6, 6>::Identity();

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n * x_small / double(n + 1);
      J_ab += x_small_n;
    }
    return J_ab;
  }
}

Eigen::Matrix<double, 6, 6> vec2jacinv(const Eigen::Vector3d& rho_ba,
                                       const Eigen::Vector3d& aaxis_ba) {
  // Init
  Eigen::Matrix<double, 6, 6> J66_ab_inv = Eigen::Matrix<double, 6, 6>::Zero();

  if (aaxis_ba.norm() < 1e-12) {
    // If angle is very small, so3 jacobian is Identity
    J66_ab_inv.topLeftCorner<3, 3>() = J66_ab_inv.bottomRightCorner<3, 3>() =
        Eigen::Matrix3d::Identity();
    J66_ab_inv.topRightCorner<3, 3>() = -0.5 * so3::hat(rho_ba);
  } else {
    // General analytical scenario
    Eigen::Matrix3d J33_ab_inv = so3::vec2jacinv(aaxis_ba);
    J66_ab_inv.topLeftCorner<3, 3>() = J66_ab_inv.bottomRightCorner<3, 3>() =
        J33_ab_inv;
    J66_ab_inv.topRightCorner<3, 3>() =
        -J33_ab_inv * se3::vec2Q(rho_ba, aaxis_ba) * J33_ab_inv;
  }
  return J66_ab_inv;
}

Eigen::Matrix<double, 6, 6> vec2jacinv(const Eigen::Matrix<double, 6, 1>& xi_ba,
                                       unsigned int numTerms) {
  if (numTerms == 0) {
    // Analytical solution
    return vec2jacinv(xi_ba.head<3>(), xi_ba.tail<3>());
  } else {
    // Logic error
    if (numTerms > 20) {
      throw std::invalid_argument(
          "Numerical vec2jacinv does not support numTerms > 20");
    }

    // Numerical solution (good for testing the analytical solution)
    Eigen::Matrix<double, 6, 6> J_ab = Eigen::Matrix<double, 6, 6>::Identity();

    // Incremental variables
    Eigen::Matrix<double, 6, 6> x_small = se3::curlyhat(xi_ba);
    Eigen::Matrix<double, 6, 6> x_small_n =
        Eigen::Matrix<double, 6, 6>::Identity();

    // Boost has a bernoulli package... but we shouldn't need more than 20
    Eigen::Matrix<double, 21, 1> bernoulli;
    bernoulli << 1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0,
        -1.0 / 30.0, 0.0, 5.0 / 66.0, 0.0, -691.0 / 2730.0, 0.0, 7.0 / 6.0, 0.0,
        -3617.0 / 510.0, 0.0, 43867.0 / 798.0, 0.0, -174611.0 / 330.0;

    // Loop over sum up to the specified numTerms
    for (unsigned int n = 1; n <= numTerms; n++) {
      x_small_n = x_small_n * x_small / double(n);
      J_ab += bernoulli(n) * x_small_n;
    }
    return J_ab;
  }
}

}  // namespace se3
}  // namespace lgmath
