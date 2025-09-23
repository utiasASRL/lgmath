//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SO2Tests.cpp
/// \brief Unit tests for the naive implementation of the SO2 Lie Group math.
/// \details Unit tests for the various Lie Group functions will test both special cases,
///          and randomly generated cases.
///
/// \author Daniil Lisus
//////////////////////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>

#include <math.h>
#include <iostream>
#include <iomanip>
#include <ios>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <Eigen/Dense>
#include <lgmath/CommonMath.hpp>
#include <lgmath/so2/Operations.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////
///
/// UNIT TESTS OF SO(2) MATH
///
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SO(2) hat function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, testHatFunction) {

    // Init
    std::vector<double> trueAngles;
    std::vector<Eigen::Matrix2d> trueMats;

    // Add angles to be tested - we choose a few special cases and random values
    trueAngles.push_back(0.0);
    trueAngles.push_back(1.0);
    trueAngles.push_back(-1.0);
    trueAngles.push_back(M_PI);
    trueAngles.push_back(-M_PI);
    trueAngles.push_back(M_PI/2);
    trueAngles.push_back(-M_PI/2);

    // Add some random angles
    for (int i = 0; i < 5; i++) {
    double randAngle = ((double)rand() / RAND_MAX - 0.5) * 4 * M_PI; // Random angle in [-2π, 2π]
    trueAngles.push_back(randAngle);
    }

    // Get number of tests
    const unsigned numTests = trueAngles.size();

    // Setup truth matrices
    // For SO(2), hat(θ) = [0  -θ]
    //                     [θ   0]
    for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix2d mat;
    mat << 0.0, -trueAngles.at(i),
            trueAngles.at(i), 0.0;
    trueMats.push_back(mat);
    }

    // Test the function
    for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix2d testMat = lgmath::so2::hat(trueAngles.at(i));
    std::cout << "angle: " << trueAngles.at(i) << std::endl;
    std::cout << "true: " << std::endl << trueMats.at(i) << std::endl;
    std::cout << "func: " << std::endl << testMat << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
    }

    // Test identity: hat(θ)^T = -hat(θ) (skew-symmetric property)
    for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix2d testMat = lgmath::so2::hat(trueAngles.at(i));
    EXPECT_TRUE(lgmath::common::nearEqual(testMat.transpose(), -testMat, 1e-6));
    }

    // Test linearity: hat(a*θ) = a*hat(θ)
    for (unsigned i = 0; i < numTests; i++) {
    double scale = 2.5;
    Eigen::Matrix2d hat_scaled_angle = lgmath::so2::hat(scale * trueAngles.at(i));
    Eigen::Matrix2d scaled_hat_angle = scale * lgmath::so2::hat(trueAngles.at(i));
    EXPECT_TRUE(lgmath::common::nearEqual(hat_scaled_angle, scaled_hat_angle, 1e-6));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test special cases of exponential functions: vec2rot and rot2vec
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, testVec2Rot2Vec) {
    // Reminder that we have C_ab = vec2rot(angle_ba)!

    // Init
    std::vector<double> trueAngles;
    std::vector<Eigen::Matrix2d> trueMats;

    // Identity (zero rotation)
    trueAngles.push_back(0.0);
    trueMats.push_back(Eigen::Matrix2d::Identity());

    // Rotation by PI
    trueAngles.push_back(-M_PI);
    Eigen::Matrix2d temp;
    temp << -1.0,  0.0,
            0.0, -1.0;
    trueMats.push_back(temp);

    // Rotation by PI/2 (90 degrees counter-clockwise)
    trueAngles.push_back(-M_PI/2);
    temp <<  0.0, -1.0,
            1.0,  0.0;
    trueMats.push_back(temp);

    // Rotation by PI/4 (45 degrees counter-clockwise)
    trueAngles.push_back(-M_PI/4);
    double sqrt2_2 = sqrt(2.0) / 2.0;
    temp <<  sqrt2_2, -sqrt2_2,
            sqrt2_2,  sqrt2_2;
    trueMats.push_back(temp);

    // Rotation by 3*PI/4 (135 degrees counter-clockwise)
    trueAngles.push_back(-3*M_PI/4);
    temp << -sqrt2_2, -sqrt2_2,
            sqrt2_2, -sqrt2_2;
    trueMats.push_back(temp);

    // Small angle rotation
    double small_angle = 0.0001;
    trueAngles.push_back(small_angle);
    temp <<  cos(small_angle), sin(small_angle),
            -sin(small_angle),  cos(small_angle);
    trueMats.push_back(temp);

    // Get number of tests
    const unsigned numTests = trueAngles.size();

    // Test vec2rot
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix2d testMat = lgmath::so2::vec2rot(trueAngles.at(i));
        std::cout << "angle: " << trueAngles.at(i) << std::endl;
        std::cout << "true: " << std::endl << trueMats.at(i) << std::endl;
        std::cout << "func: " << std::endl << testMat << std::endl;
        EXPECT_TRUE(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
    }

    // Test rot2vec
    for (unsigned i = 0; i < numTests; i++) {
        double testAngle = lgmath::so2::rot2vec(trueMats.at(i));
        std::cout << "true angle: " << trueAngles.at(i) << std::endl;
        std::cout << "func angle: " << testAngle << std::endl;
        
        // For angles near ±π, we need to handle the wraparound
        double expected = trueAngles.at(i);
        double actual = testAngle;
        
        // Normalize both to [-π, π] for comparison
        while (expected > M_PI) expected -= 2.0 * M_PI;
        while (expected < -M_PI) expected += 2.0 * M_PI;
        while (actual > M_PI) actual -= 2.0 * M_PI;
        while (actual < -M_PI) actual += 2.0 * M_PI;
        
        EXPECT_NEAR(expected, actual, 1e-6);
    }

    // Test round-trip consistency: rot2vec(vec2rot(angle)) == angle
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix2d rot_mat = lgmath::so2::vec2rot(trueAngles.at(i));
        double recovered_angle = lgmath::so2::rot2vec(rot_mat);
        
        // Normalize both angles to [-π, π] for comparison
        double expected = trueAngles.at(i);
        double actual = recovered_angle;
        
        while (expected > M_PI) expected -= 2.0 * M_PI;
        while (expected < -M_PI) expected += 2.0 * M_PI;
        while (actual > M_PI) actual -= 2.0 * M_PI;
        while (actual < -M_PI) actual += 2.0 * M_PI;
        
        std::cout << "Round-trip test - original: " << trueAngles.at(i) 
                << ", recovered: " << recovered_angle << std::endl;
        EXPECT_NEAR(expected, actual, 1e-6);
    }

    // Test determinant property: det(R) = 1 for all rotation matrices
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix2d testMat = lgmath::so2::vec2rot(trueAngles.at(i));
        EXPECT_NEAR(testMat.determinant(), 1.0, 1e-6);
    }

    // Test orthogonality property: R^T * R = I
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix2d testMat = lgmath::so2::vec2rot(trueAngles.at(i));
        Eigen::Matrix2d should_be_identity = testMat.transpose() * testMat;
        EXPECT_TRUE(lgmath::common::nearEqual(should_be_identity, Eigen::Matrix2d::Identity(), 1e-6));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential jacobians: vec2jac and vec2jacinv
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, testGroupJacobians) {

    // These should both just be 1.0.
    double J = lgmath::so2::vec2jac();
    EXPECT_EQ(J, 1.0);
    double J_inv = lgmath::so2::vec2jacinv();
    EXPECT_EQ(J_inv, 1.0);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

