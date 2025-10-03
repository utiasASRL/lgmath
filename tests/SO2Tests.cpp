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
#include <lgmath/so2/Rotation.hpp>

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
        // std::cout << "angle: " << trueAngles.at(i) << std::endl;
        // std::cout << "true: " << std::endl << trueMats.at(i) << std::endl;
        // std::cout << "func: " << std::endl << testMat << std::endl;
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
    trueAngles.push_back(M_PI);
    Eigen::Matrix2d temp;
    temp << -1.0,  0.0,
            0.0,  -1.0;
    trueMats.push_back(temp);

    // Rotation by PI/2 (90 degrees counter-clockwise)
    trueAngles.push_back(M_PI/2);
    temp << 0.0, -1.0,
            1.0,  0.0;
    trueMats.push_back(temp);

    // Rotation by PI/4 (45 degrees counter-clockwise)
    trueAngles.push_back(M_PI/4);
    double sqrt2_2 = sqrt(2.0) / 2.0;
    temp << sqrt2_2, -sqrt2_2,
            sqrt2_2,  sqrt2_2;
    trueMats.push_back(temp);

    // Rotation by 3*PI/4 (135 degrees counter-clockwise)
    trueAngles.push_back(3*M_PI/4);
    temp << -sqrt2_2, -sqrt2_2,
            sqrt2_2, -sqrt2_2;
    trueMats.push_back(temp);

    // Small angle rotation
    double small_angle = 0.0001;
    trueAngles.push_back(small_angle);
    temp <<  cos(small_angle), -sin(small_angle),
             sin(small_angle),  cos(small_angle);
    trueMats.push_back(temp);

    // Get number of tests
    const unsigned numTests = trueAngles.size();

    // Test vec2rot
    for (unsigned i = 0; i < numTests; i++) {
        Eigen::Matrix2d testMat = lgmath::so2::vec2rot(trueAngles.at(i));
        // std::cout << "angle: " << trueAngles.at(i) << std::endl;
        // std::cout << "true: " << std::endl << trueMats.at(i) << std::endl;
        // std::cout << "func: " << std::endl << testMat << std::endl;
        EXPECT_TRUE(lgmath::common::nearEqual(trueMats.at(i), testMat, 1e-6));
    }

    // Test rot2vec
    for (unsigned i = 0; i < numTests; i++) {
        double testAngle = lgmath::so2::rot2vec(trueMats.at(i));
        // std::cout << "true angle: " << trueAngles.at(i) << std::endl;
        // std::cout << "func angle: " << testAngle << std::endl;
        
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
        
        // std::cout << "Round-trip test - original: " << trueAngles.at(i) 
        //         << ", recovered: " << recovered_angle << std::endl;
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

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test SO(2) Rotation constructors
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, testSO2RotationDefaultConstructor) {
    lgmath::so2::Rotation rot_default;
    Eigen::Matrix2d expected_identity = Eigen::Matrix2d::Identity();
    EXPECT_TRUE(lgmath::common::nearEqual(rot_default.matrix(), expected_identity, 1e-6));
    EXPECT_NEAR(rot_default.vec(), 0.0, 1e-6);
}

TEST(LGMath, testSO2RotationAngleConstructor) {
    std::vector<double> test_angles = {0.0, M_PI/4, M_PI/2, M_PI, -M_PI/4, -M_PI/2, -M_PI, 2*M_PI, -2*M_PI};
    
    for (double angle : test_angles) {
        lgmath::so2::Rotation rot_from_angle{angle};
        
        // Check that the matrix is correct
        Eigen::Matrix2d expected_matrix = lgmath::so2::vec2rot(angle);
        EXPECT_TRUE(lgmath::common::nearEqual(rot_from_angle.matrix(), expected_matrix, 1e-6));
        
        // Check round-trip: angle -> rotation -> angle
        double recovered_angle = rot_from_angle.vec();
        
        // Normalize angles to [-π, π] for comparison
        double normalized_original = angle;
        while (normalized_original > M_PI) normalized_original -= 2.0 * M_PI;
        while (normalized_original < -M_PI) normalized_original += 2.0 * M_PI;
        
        double normalized_recovered = recovered_angle;
        while (normalized_recovered > M_PI) normalized_recovered -= 2.0 * M_PI;
        while (normalized_recovered < -M_PI) normalized_recovered += 2.0 * M_PI;
        
        EXPECT_NEAR(normalized_original, normalized_recovered, 1e-6);
        
        // Check that it's a valid rotation matrix
        EXPECT_NEAR(rot_from_angle.matrix().determinant(), 1.0, 1e-6);
        Eigen::Matrix2d should_be_identity = rot_from_angle.matrix().transpose() * rot_from_angle.matrix();
        EXPECT_TRUE(lgmath::common::nearEqual(should_be_identity, Eigen::Matrix2d::Identity(), 1e-6));
    }
}

TEST(LGMath, testSO2RotationMatrixConstructor) {
    // Test with various rotation matrices
    std::vector<double> angles = {0.0, M_PI/6, M_PI/4, M_PI/3, M_PI/2, 2*M_PI/3, M_PI};
    
    for (double angle : angles) {
        Eigen::Matrix2d rotation_matrix = lgmath::so2::vec2rot(angle);
        lgmath::so2::Rotation rot_from_matrix(rotation_matrix);
        
        // The constructor should preserve the matrix (possibly with reprojection)
        EXPECT_TRUE(lgmath::common::nearEqual(rot_from_matrix.matrix(), rotation_matrix, 1e-6));
        
        // Check that it's still a valid rotation matrix
        EXPECT_NEAR(rot_from_matrix.matrix().determinant(), 1.0, 1e-6);
        Eigen::Matrix2d should_be_identity = rot_from_matrix.matrix().transpose() * rot_from_matrix.matrix();
        EXPECT_TRUE(lgmath::common::nearEqual(should_be_identity, Eigen::Matrix2d::Identity(), 1e-6));
    }
}

TEST(LGMath, testSO2RotationMatrixReprojection) {
    // Test with a slightly "corrupted" rotation matrix (should be reprojected)
    Eigen::Matrix2d corrupted_matrix;
    corrupted_matrix << 1.001, 0.001,
                       -0.001, 1.001;  // Close to identity but not exactly orthogonal
    
    lgmath::so2::Rotation rot_from_corrupted(corrupted_matrix);
    
    // Should be reprojected to a valid rotation matrix
    EXPECT_NEAR(rot_from_corrupted.matrix().determinant(), 1.0, 1e-6);
    Eigen::Matrix2d should_be_identity = rot_from_corrupted.matrix().transpose() * rot_from_corrupted.matrix();
    EXPECT_TRUE(lgmath::common::nearEqual(should_be_identity, Eigen::Matrix2d::Identity(), 1e-6));
}

TEST(LGMath, testSO2RotationVectorConstructor) {
    std::vector<double> test_angles = {0.0, M_PI/4, M_PI/2, M_PI, -M_PI/4, -M_PI/2};
    
    for (double angle : test_angles) {
        Eigen::VectorXd angle_vector(1);
        angle_vector(0) = angle;
        
        lgmath::so2::Rotation rot_from_vector(angle_vector);
        
        // Should be equivalent to constructor from double
        lgmath::so2::Rotation rot_from_double{angle};
        EXPECT_TRUE(lgmath::common::nearEqual(rot_from_vector.matrix(), rot_from_double.matrix(), 1e-6));
        EXPECT_NEAR(rot_from_vector.vec(), rot_from_double.vec(), 1e-6);
    }
}

TEST(LGMath, testSO2RotationVectorConstructorErrors) {
    // Test error case: VectorXd with wrong dimension
    Eigen::VectorXd wrong_size_vector(2);
    wrong_size_vector << 1.0, 2.0;
    EXPECT_THROW(lgmath::so2::Rotation rot_wrong(wrong_size_vector), std::invalid_argument);
    
    // Test error case: VectorXd with dimension 0
    Eigen::VectorXd empty_vector(0);
    EXPECT_THROW(lgmath::so2::Rotation rot_empty(empty_vector), std::invalid_argument);
}

TEST(LGMath, testSO2RotationCopyAndAssignment) {
    double test_angle = M_PI/3;
    lgmath::so2::Rotation original{test_angle};
    
    // Copy constructor
    lgmath::so2::Rotation copied(original);
    EXPECT_TRUE(lgmath::common::nearEqual(original.matrix(), copied.matrix(), 1e-6));
    EXPECT_NEAR(original.vec(), copied.vec(), 1e-6);
    
    // Assignment operator
    lgmath::so2::Rotation assigned;
    assigned = original;
    EXPECT_TRUE(lgmath::common::nearEqual(original.matrix(), assigned.matrix(), 1e-6));
    EXPECT_NEAR(original.vec(), assigned.vec(), 1e-6);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
