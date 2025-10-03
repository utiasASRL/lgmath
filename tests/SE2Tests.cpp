//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SE2Tests.cpp
/// \brief Unit tests for the implementation of the SE2 Lie Group math.
/// \details Unit tests for the various Lie Group functions will test both
/// special cases, and randomly generated cases.
///
/// \author Daniil Lisus
//////////////////////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>

#include <math.h>
#include <iomanip>
#include <ios>
#include <iostream>

#include <Eigen/Dense>
#include <lgmath/CommonMath.hpp>
#include <lgmath/se2/Operations.hpp>
#include <lgmath/so2/Operations.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////
///
/// UNIT TESTS OF SE(2) MATH
///
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SE(2) hat function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, TestHatFunction) {
    // Number of random tests
    const unsigned numTests = 20;

    // Add vectors to be tested - random
    std::vector<Eigen::Matrix<double, 3, 1> > trueVecs;
    for (unsigned i = 0; i < numTests; i++) {
        trueVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
    }

    // Setup truth matrices
    std::vector<Eigen::Matrix<double, 3, 3> > trueMats;
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix<double, 3, 3> mat;
        mat << 0.0, -trueVecs.at(i)[2], trueVecs.at(i)[0],
            trueVecs.at(i)[2], 0.0, trueVecs.at(i)[1],
            0.0, 0.0, 0.0;
        trueMats.push_back(mat);
    }

    // Test the function
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix<double, 3, 3> testMat = lgmath::se2::hat(trueVecs.at(i));
        EXPECT_TRUE(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
    }
}

TEST(LGMath, TestHatFunctionInputs) {
    // Number of random tests
    const unsigned numTests = 20;

    // Add vectors to be tested - random
    std::vector<Eigen::Matrix<double, 3, 1> > trueVecs;
    for (unsigned i = 0; i < numTests; i++) {
        trueVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
    }

    // Test the two function inputs
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix<double, 3, 3> testMatFullVec = lgmath::se2::hat(trueVecs.at(i));
        Eigen::Matrix<double, 3, 3> testMatComponents = lgmath::se2::hat(trueVecs.at(i).head<2>(), trueVecs.at(i)[2]);
        EXPECT_EQ(testMatFullVec, testMatComponents);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SE(2) curlyhat function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, TestCurlyHatFunction) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double, 3, 1> > trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double, 3, 3> > trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 3, 3> mat;
    // SE(2) curlyhat: [0 -angle rho2; angle 0 -rho1; 0 0 0]
    // where xi = [rho1, rho2, angle]
    mat << 0.0, -trueVecs.at(i)[2], trueVecs.at(i)[1],
           trueVecs.at(i)[2], 0.0, -trueVecs.at(i)[0],
           0.0, 0.0, 0.0;
    trueMats.push_back(mat);
  }

  // Test the function
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 3, 3> testMat = lgmath::se2::curlyhat(trueVecs.at(i));
    EXPECT_TRUE(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test that the circle dot operator of a homogeneous point p follows the
/// identity: xi^ * p = circledot(p) * xi
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, TestPointToCircleDotOperatorFunction) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double, 3, 1> > xiVecs;
  std::vector<Eigen::Matrix<double, 3, 1> > pVecs;
  for (unsigned i = 0; i < numTests; i++) {
    xiVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
    pVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
  }

  // Test the function
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector2d point = pVecs.at(i).head<2>();
    double scale_i = pVecs.at(i)(2);
    Eigen::Vector3d lhs = lgmath::se2::hat(xiVecs.at(i)) * pVecs.at(i);
    Eigen::Vector3d rhs = lgmath::se2::point2fs(point, scale_i) * xiVecs.at(i);
    
    EXPECT_TRUE(lgmath::common::nearEqual(lhs, rhs, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test that the double-circle operator of a homogeneous point p follows the
/// identity: p^T * xi^ = xi^T * doublecircle(p)
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, TestPointToDoubleCircleOperatorFunction) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double, 3, 1> > xiVecs;
  std::vector<Eigen::Matrix<double, 3, 1> > pVecs;
  for (unsigned i = 0; i < numTests; i++) {
    xiVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
    pVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
  }

  // Test the function
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Vector2d point = pVecs.at(i).head<2>();
    double scale_i = pVecs.at(i)(2);
    Eigen::Vector3d lhs = pVecs.at(i).transpose() * lgmath::se2::hat(xiVecs.at(i));
    Eigen::Vector3d rhs = xiVecs.at(i).transpose() * lgmath::se2::point2sf(point, scale_i);
    
    EXPECT_TRUE(lgmath::common::nearEqual(lhs, rhs, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SE(2) vec2tran function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, TestVec2TranFunction) {
    // Number of random tests
    const unsigned numTests = 20;

    // Add vectors to be tested - random
    std::vector<Eigen::Matrix<double, 3, 1> > trueVecs;
    for (unsigned i = 0; i < numTests; i++) {
        trueVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
    }

    // Setup truth matrices
    std::vector<Eigen::Matrix3d> trueMats;
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix2d C_ab;
        Eigen::Vector2d r_ba_ina;
        lgmath::se2::vec2tran(trueVecs.at(i), &C_ab, &r_ba_ina);
        Eigen::Matrix3d T_ab = Eigen::Matrix3d::Identity();
        T_ab.topLeftCorner<2, 2>() = C_ab;
        T_ab.topRightCorner<2, 1>() = r_ba_ina;
        trueMats.push_back(T_ab);
    }

    // Test the function
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix3d testMat = lgmath::se2::vec2tran(trueVecs.at(i));
        EXPECT_TRUE(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SE(2) tran2vec function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, TestTran2VecFunction) {
    // Number of random tests
    const unsigned numTests = 20;

    // Add vectors to be tested - random
    std::vector<Eigen::Matrix<double, 3, 1> > trueVecs;
    for (unsigned i = 0; i < numTests; i++) {
        trueVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
    }

    // Setup truth matrices
    std::vector<Eigen::Matrix3d> trueMats;
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix2d C_ab;
        Eigen::Vector2d r_ba_ina;
        lgmath::se2::vec2tran(trueVecs.at(i), &C_ab, &r_ba_ina);
        Eigen::Matrix3d T_ab = Eigen::Matrix3d::Identity();
        T_ab.topLeftCorner<2, 2>() = C_ab;
        T_ab.topRightCorner<2, 1>() = r_ba_ina;
        trueMats.push_back(T_ab);
    }

    // Test the function
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix<double, 3, 1> testVec = lgmath::se2::tran2vec(trueMats.at(i));
        EXPECT_TRUE(lgmath::common::nearEqual(trueVecs.at(i), testVec, 1e-6));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential functions: vec2tran and tran2vec
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, Vec2Tran2VecTest) {
    // Add vectors to be tested
    std::vector<Eigen::Matrix<double, 3, 1>> trueVecs;
    Eigen::Matrix<double, 3, 1> temp;
    temp << 0.0, 0.0, 0.0;
    trueVecs.push_back(temp);
    temp << 1.0, 0.0, 0.0;
    trueVecs.push_back(temp);
    temp << 0.0, 1.0, 0.0;
    trueVecs.push_back(temp);
    temp << 0.0, 0.0, lgmath::constants::PI;
    trueVecs.push_back(temp);
    temp << 0.0, 0.0, -lgmath::constants::PI;
    trueVecs.push_back(temp);
    temp << 0.0, 0.0, 0.5 * lgmath::constants::PI;
    trueVecs.push_back(temp);
    const unsigned numRand = 20;
    for (unsigned i = 0; i < numRand; i++) {
        trueVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
    }

    // Get number of tests
    const unsigned numTests = trueVecs.size();

    // Calc matrices
    std::vector<Eigen::Matrix<double, 3, 3> > analyticTrans;
    for (unsigned i = 0; i < numTests; i++) {
        analyticTrans.push_back(lgmath::se2::vec2tran(trueVecs.at(i)));
    }

    // Test rot2vec
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix<double, 3, 1> testVec = lgmath::se2::tran2vec(analyticTrans.at(i));
        EXPECT_TRUE(lgmath::common::nearEqual(trueVecs.at(i), testVec, 1e-6));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential jacobians: vec2jac and vec2jacinv
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, JacAndInvJacTest) {
    // Add vectors to be tested
    std::vector<Eigen::Matrix<double, 3, 1>> trueVecs;
    Eigen::Matrix<double, 3, 1> temp;
    temp << 0.0, 0.0, 0.0;
    trueVecs.push_back(temp);
    temp << 1.0, 0.0, 0.0;
    trueVecs.push_back(temp);
    temp << 0.0, 1.0, 0.0;
    trueVecs.push_back(temp);
    temp << 0.0, 0.0, lgmath::constants::PI;
    trueVecs.push_back(temp);
    temp << 0.0, 0.0, -lgmath::constants::PI;
    trueVecs.push_back(temp);
    temp << 0.0, 0.0, 0.5 * lgmath::constants::PI;
    trueVecs.push_back(temp);
    const unsigned numRand = 20;
    for (unsigned i = 0; i < numRand; i++) {
        trueVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
    }

    // Get number of tests
    const unsigned numTests = trueVecs.size();

    // Compute vec2jac and vec2jacinv
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix<double, 3, 3> Jac = lgmath::se2::vec2jac(trueVecs.at(i));
        Eigen::Matrix<double, 3, 3> JacInv = lgmath::se2::vec2jacinv(trueVecs.at(i));
        // Compare
        EXPECT_TRUE(lgmath::common::nearEqual(Jac * JacInv, Eigen::Matrix3d::Identity(), 1e-6));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of adjoint tranformation identity, Ad(T(v)) = I + curlyhat(v)*J(v)
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, TestIdentityAdTvEqualIPlusHatvTimesJv) {
    // Add vectors to be tested
    std::vector<Eigen::Matrix<double, 3, 1>> trueVecs;
    Eigen::Matrix<double, 3, 1> temp;
    temp << 0.0, 0.0, 0.0;
    trueVecs.push_back(temp);
    temp << 1.0, 0.0, 0.0;
    trueVecs.push_back(temp);
    temp << 0.0, 1.0, 0.0;
    trueVecs.push_back(temp);
    temp << 0.0, 0.0, lgmath::constants::PI;
    trueVecs.push_back(temp);
    temp << 0.0, 0.0, -lgmath::constants::PI;
    trueVecs.push_back(temp);
    temp << 0.0, 0.0, 0.5 * lgmath::constants::PI;
    trueVecs.push_back(temp);
    const unsigned numRand = 20;
    for (unsigned i = 0; i < numRand; i++) {
        trueVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
    }

    // Get number of tests
    const unsigned numTests = trueVecs.size();

    // Check identity
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix<double, 3, 3> lhs = lgmath::se2::tranAd(lgmath::se2::vec2tran(trueVecs.at(i)));
        Eigen::Matrix<double, 3, 3> rhs = Eigen::Matrix3d::Identity() +
                                          lgmath::se2::curlyhat(trueVecs.at(i)) *
                                          lgmath::se2::vec2jac(trueVecs.at(i));
        EXPECT_TRUE(lgmath::common::nearEqual(lhs, rhs, 1e-6));
    }
}