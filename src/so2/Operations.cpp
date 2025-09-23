/**
 * \file Operations.cpp
 * \brief Implementation file for the SO2 Lie Group math functions.
 * \details These namespace functions provide implementations of the special
 * orthogonal (SO) Lie group functions that we commonly use in robotics.
 *
 * \author Daniil Lisus
 */
#include <lgmath/so2/Operations.hpp>

#include <stdio.h>
#include <algorithm>
#include <stdexcept>

#include <Eigen/Dense>

namespace lgmath {
namespace so2 {

Eigen::Matrix2d hat(const double angle) {
  Eigen::Matrix2d mat;
  mat << 0.0, -angle, angle, 0.0;
  return mat;
}

Eigen::Matrix2d vec2rot(const double angle_ba) {
    // Note flipped minus sign on sin term since we're returning C_ab
    // from angle_ba.
    const double sin_ba = sin(angle_ba);
    const double cos_ba = cos(angle_ba);
    Eigen::Matrix2d mat;
    mat << cos_ba, sin_ba, -sin_ba, cos_ba;
    return mat;
}

double rot2vec(const Eigen::Matrix2d& C_ab) {
    // Note minus sign since we're returning angle_ba.
    return -atan2(C_ab(1, 0), C_ab(0, 0));
}

double vec2jac() {
  return 1.0;
}

double vec2jacinv() {
  return 1.0;
}

}  // namespace so2
}  // namespace lgmath
