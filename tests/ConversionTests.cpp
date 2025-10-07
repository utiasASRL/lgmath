//////////////////////////////////////////////////////////////////////////////////////////////
/// \file ConversionTests.cpp
/// \brief Unit tests for SE(2) ↔ SE(3) conversion functions
/// \details Unit tests for the conversion functions between SE(2) and SE(3)
/// transformations, both for basic Transformation classes and 
/// TransformationWithCovariance classes.
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
#include <lgmath/se2/Transformation.hpp>
#include <lgmath/se2/TransformationWithCovariance.hpp>
#include <lgmath/se3/Operations.hpp>
#include <lgmath/se3/Transformation.hpp>
#include <lgmath/se3/TransformationWithCovariance.hpp>
#include <lgmath/so2/Operations.hpp>
#include <lgmath/so3/Operations.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////
///
/// UNIT TESTS OF SE(2) ↔ SE(3) CONVERSIONS
///
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test SE(2) → SE(3) → SE(2) round-trip for basic Transformation
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, SE2ToSE3ToSE2RoundTrip) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - specific cases
  std::vector<Eigen::Matrix<double, 3, 1> > se2_vecs;
  Eigen::Matrix<double, 3, 1> temp;
  temp << 0.0, 0.0, 0.0;  // Identity
  se2_vecs.push_back(temp);
  temp << 1.0, 0.0, 0.0;  // Pure x translation
  se2_vecs.push_back(temp);
  temp << 0.0, 1.0, 0.0;  // Pure y translation
  se2_vecs.push_back(temp);
  temp << 0.0, 0.0, M_PI/2;  // Pure rotation
  se2_vecs.push_back(temp);
  temp << 0.0, 0.0, M_PI;    // 180° rotation
  se2_vecs.push_back(temp);
  temp << 0.0, 0.0, -M_PI/2; // Negative rotation
  se2_vecs.push_back(temp);
  temp << 1.0, 1.0, M_PI/4;  // Combined translation and rotation
  se2_vecs.push_back(temp);

  // Add random test cases
  for (unsigned i = 0; i < numTests; i++) {
    se2_vecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
  }

  const unsigned totalTests = se2_vecs.size();

  // Test round-trip conversions
  for (unsigned i = 0; i < totalTests; i++) {
    // Create original SE(2) transformation
    lgmath::se2::Transformation T_se2_original(se2_vecs.at(i));
    
    // Convert to SE(3)
    lgmath::se3::Transformation T_se3 = T_se2_original.toSE3();
    
    // Convert back to SE(2)
    lgmath::se2::Transformation T_se2_recovered = T_se3.toSE2();

    // std::cout << "Test " << i << ":\n";
    // std::cout << "Original SE(2) mat:\n" << T_se2_original.matrix() << "\n";
    // std::cout << "Intermediate SE(3) mat:\n" << T_se3.matrix() << "\n";
    // std::cout << "Recovered SE(2) mat:\n" << T_se2_recovered.matrix() << "\n";
    
    // Check that we recover the original transformation
    EXPECT_TRUE(lgmath::common::nearEqual(
        T_se2_original.matrix(), T_se2_recovered.matrix(), 1e-12))
        << "Round-trip conversion failed for test " << i;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test SE(3) → SE(2) → SE(3) projection properties for basic Transformation
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, SE3ToSE2ProjectionProperties) {
  // Number of random tests
  const unsigned numTests = 20;
  
  // Add vectors to be tested - specific cases
  // Note that we want to keep roll/pitch small to ensure semi-valid SO(2) projection
  std::vector<Eigen::Matrix<double, 6, 1> > se3_vecs;
  Eigen::Matrix<double, 6, 1> temp;
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;  // Identity
  se3_vecs.push_back(temp);
  temp << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;  // Pure x translation
  se3_vecs.push_back(temp);
  temp << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;  // Pure y translation
  se3_vecs.push_back(temp);
  temp << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;  // Pure z translation (should be lost)
  se3_vecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, M_PI/2;  // Pure yaw (should be preserved)
  se3_vecs.push_back(temp);
  temp << 1.0, 1.0, 1.0, 0.0, 0.0, M_PI/4;  // Full 6DOF motion
  se3_vecs.push_back(temp);

  // Add random test cases
  for (unsigned i = 0; i < numTests; i++) {
    se3_vecs.push_back(Eigen::Matrix<double, 6, 1>::Random());
    // Zero out roll and pitch to ensure valid SO(2) projection
    se3_vecs.back()(3) = 0.0;
    se3_vecs.back()(4) = 0.0;
  }

  const unsigned totalTests = se3_vecs.size();

  // Test projection properties
  for (unsigned i = 0; i < totalTests; i++) {
    // Create original SE(3) transformation
    lgmath::se3::Transformation T_se3_original(se3_vecs.at(i));
    
    // Convert to SE(2)
    lgmath::se2::Transformation T_se2 = T_se3_original.toSE2();
    
    // Convert back to SE(3)
    lgmath::se3::Transformation T_se3_recovered = T_se2.toSE3();
    
    // Check that x, y translation and yaw rotation are preserved
    Eigen::Matrix4d T_orig = T_se3_original.matrix();
    Eigen::Matrix4d T_recovered = T_se3_recovered.matrix();

    // std::cout << "Test " << i << ":\n";
    // std::cout << "Original SE(3) mat:\n" << T_se3_original.matrix() << "\n";
    // std::cout << "Intermediate SE(2) mat:\n" << T_se2.matrix() << "\n";
    // std::cout << "Recovered SE(3) mat:\n" << T_se3_recovered.matrix() << "\n";
    
    // Check x, y translation preservation
    EXPECT_NEAR(T_orig(0, 3), T_recovered(0, 3), 1e-12) 
        << "X translation not preserved for test " << i;
    EXPECT_NEAR(T_orig(1, 3), T_recovered(1, 3), 1e-12) 
        << "Y translation not preserved for test " << i;
    
    // Check that z translation is zeroed
    EXPECT_NEAR(T_recovered(2, 3), 0.0, 1e-12) 
        << "Z translation not zeroed for test " << i;
    
    // Check yaw rotation preservation (rotation around z-axis)
    // Extract the 2x2 rotation in xy-plane
    Eigen::Matrix2d R_orig = T_orig.block<2, 2>(0, 0);
    Eigen::Matrix2d R_recovered = T_recovered.block<2, 2>(0, 0);
    EXPECT_TRUE(lgmath::common::nearEqual(R_orig, R_recovered, 1e-12))
        << "XY rotation not preserved for test " << i;
    
    // Check that z-axis direction is preserved (roll and pitch should be zero)
    Eigen::Vector3d z_axis_recovered = T_recovered.block<3, 1>(0, 2);
    Eigen::Vector3d expected_z_axis(0, 0, 1);
    EXPECT_TRUE(lgmath::common::nearEqual(z_axis_recovered, expected_z_axis, 1e-12))
        << "Z-axis not aligned after projection for test " << i;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test SE(2) → SE(3) → SE(2) round-trip for TransformationWithCovariance
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, SE2WithCovToSE3WithCovToSE2WithCovRoundTrip) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - specific cases
  std::vector<Eigen::Matrix<double, 3, 1> > se2_vecs;
  Eigen::Matrix<double, 3, 1> temp;
  temp << 0.0, 0.0, 0.0;  // Identity
  se2_vecs.push_back(temp);
  temp << 1.0, 0.0, 0.0;  // Pure x translation
  se2_vecs.push_back(temp);
  temp << 0.0, 1.0, 0.0;  // Pure y translation
  se2_vecs.push_back(temp);
  temp << 0.0, 0.0, M_PI/2;  // Pure rotation
  se2_vecs.push_back(temp);
  temp << 1.0, 1.0, M_PI/4;  // Combined motion
  se2_vecs.push_back(temp);

  // Add random test cases
  for (unsigned i = 0; i < numTests; i++) {
    se2_vecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
  }

  const unsigned totalTests = se2_vecs.size();

  // Test round-trip conversions with covariance
  for (unsigned i = 0; i < totalTests; i++) {
    // Create random covariance matrix for SE(2)
    Eigen::Matrix<double, 3, 3> cov_se2_original = 
        Eigen::Matrix<double, 3, 3>::Random();
    cov_se2_original = cov_se2_original * cov_se2_original.transpose(); // Make positive definite
    
    // Create original SE(2) transformation with covariance
    lgmath::se2::TransformationWithCovariance T_se2_original(se2_vecs.at(i), cov_se2_original);
    
    // Convert to SE(3) with covariance
    lgmath::se3::TransformationWithCovariance T_se3 = T_se2_original.toSE3();
    
    // Convert back to SE(2) with covariance
    lgmath::se2::TransformationWithCovariance T_se2_recovered = T_se3.toSE2();
    
    // Check that we recover the original transformation
    EXPECT_TRUE(lgmath::common::nearEqual(
        T_se2_original.matrix(), T_se2_recovered.matrix(), 1e-12))
        << "Round-trip transformation failed for test " << i;
    
    // Check that we recover the original covariance
    EXPECT_TRUE(lgmath::common::nearEqual(
        T_se2_original.cov(), T_se2_recovered.cov(), 1e-12))
        << "Round-trip covariance failed for test " << i;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test SE(3) → SE(2) covariance projection for TransformationWithCovariance
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, SE3WithCovToSE2WithCovProjection) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - specific cases
  // Note that we want to keep roll/pitch small to ensure semi-valid SO(2) projection
  std::vector<Eigen::Matrix<double, 6, 1> > se3_vecs;
  Eigen::Matrix<double, 6, 1> temp;
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;  // Identity
  se3_vecs.push_back(temp);
  temp << 1.0, 1.0, 1.0, 0.0, 0.0, M_PI/4;  // Full 6DOF motion
  se3_vecs.push_back(temp);

  // Add random test cases
  for (unsigned i = 0; i < numTests; i++) {
    se3_vecs.push_back(Eigen::Matrix<double, 6, 1>::Random());
    // Zero out roll and pitch to ensure valid SO(2) projection
    se3_vecs.back()(3) = 0.0;
    se3_vecs.back()(4) = 0.0;
  }

  const unsigned totalTests = se3_vecs.size();

  // Test covariance projection properties
  for (unsigned i = 0; i < totalTests; i++) {
    // Create random covariance matrix for SE(3)
    Eigen::Matrix<double, 6, 6> cov_se3_original = 
        Eigen::Matrix<double, 6, 6>::Random();
    cov_se3_original = cov_se3_original * cov_se3_original.transpose(); // Make positive definite
    
    // Create original SE(3) transformation with covariance
    lgmath::se3::TransformationWithCovariance T_se3_original(se3_vecs.at(i), cov_se3_original);
    
    // Convert to SE(2) with covariance
    lgmath::se2::TransformationWithCovariance T_se2 = T_se3_original.toSE2();
    
    // Check that transformation projection works
    lgmath::se3::Transformation T_se3_base = T_se3_original;
    lgmath::se2::Transformation T_se2_expected = T_se3_base.toSE2();
    EXPECT_TRUE(lgmath::common::nearEqual(
        T_se2.matrix(), T_se2_expected.matrix(), 1e-12))
        << "Transformation projection failed for test " << i;
    
    // Check covariance projection - should extract x, y, yaw components
    Eigen::Matrix<double, 3, 3> cov_se2 = T_se2.cov();
    
    // x variance should match
    EXPECT_NEAR(cov_se2(0, 0), cov_se3_original(0, 0), 1e-12)
        << "X variance not preserved for test " << i;
    
    // y variance should match
    EXPECT_NEAR(cov_se2(1, 1), cov_se3_original(1, 1), 1e-12)
        << "Y variance not preserved for test " << i;
    
    // yaw variance should match (yaw is index 5 in SE(3), index 2 in SE(2))
    EXPECT_NEAR(cov_se2(2, 2), cov_se3_original(5, 5), 1e-12)
        << "Yaw variance not preserved for test " << i;
    
    // Cross-correlations should be preserved
    EXPECT_NEAR(cov_se2(0, 1), cov_se3_original(0, 1), 1e-12)
        << "X-Y correlation not preserved for test " << i;
    EXPECT_NEAR(cov_se2(1, 0), cov_se3_original(1, 0), 1e-12)
        << "Y-X correlation not preserved for test " << i;
    EXPECT_NEAR(cov_se2(0, 2), cov_se3_original(0, 5), 1e-12)
        << "X-Yaw correlation not preserved for test " << i;
    EXPECT_NEAR(cov_se2(2, 0), cov_se3_original(5, 0), 1e-12)
        << "Yaw-X correlation not preserved for test " << i;
    EXPECT_NEAR(cov_se2(1, 2), cov_se3_original(1, 5), 1e-12)
        << "Y-Yaw correlation not preserved for test " << i;
    EXPECT_NEAR(cov_se2(2, 1), cov_se3_original(5, 1), 1e-12)
        << "Yaw-Y correlation not preserved for test " << i;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test SE(2) → SE(3) covariance embedding for TransformationWithCovariance
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, SE2WithCovToSE3WithCovEmbedding) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - specific cases
  std::vector<Eigen::Matrix<double, 3, 1> > se2_vecs;
  Eigen::Matrix<double, 3, 1> temp;
  temp << 0.0, 0.0, 0.0;  // Identity
  se2_vecs.push_back(temp);
  temp << 1.0, 1.0, M_PI/4;  // Combined motion
  se2_vecs.push_back(temp);

  // Add random test cases
  for (unsigned i = 0; i < numTests; i++) {
    se2_vecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
  }

  const unsigned totalTests = se2_vecs.size();

  // Test covariance embedding properties
  for (unsigned i = 0; i < totalTests; i++) {
    // Create random covariance matrix for SE(2)
    Eigen::Matrix<double, 3, 3> cov_se2_original = 
        Eigen::Matrix<double, 3, 3>::Random();
    cov_se2_original = cov_se2_original * cov_se2_original.transpose(); // Make positive definite
    
    // Create original SE(2) transformation with covariance
    lgmath::se2::TransformationWithCovariance T_se2_original(se2_vecs.at(i), cov_se2_original);
    
    // Convert to SE(3) with covariance
    lgmath::se3::TransformationWithCovariance T_se3 = T_se2_original.toSE3();
    
    // Check that transformation embedding works
    lgmath::se2::Transformation T_se2_base = T_se2_original;
    lgmath::se3::Transformation T_se3_expected = T_se2_base.toSE3();
    EXPECT_TRUE(lgmath::common::nearEqual(
        T_se3.matrix(), T_se3_expected.matrix(), 1e-12))
        << "Transformation embedding failed for test " << i;
    
    // Check covariance embedding - should place SE(2) components in appropriate positions
    Eigen::Matrix<double, 6, 6> cov_se3 = T_se3.cov();
    
    // Check that SE(2) covariance is embedded in the xy-yaw sub-block
    Eigen::Matrix<double, 3, 3> embedded_cov = Eigen::Matrix3d::Zero();
    embedded_cov(0, 0) = cov_se3(0, 0);
    embedded_cov(0, 1) = cov_se3(0, 1);
    embedded_cov(0, 2) = cov_se3(0, 5);
    embedded_cov(1, 0) = cov_se3(1, 0);
    embedded_cov(1, 1) = cov_se3(1, 1);
    embedded_cov(1, 2) = cov_se3(1, 5);
    embedded_cov(2, 0) = cov_se3(5, 0);
    embedded_cov(2, 1) = cov_se3(5, 1);
    embedded_cov(2, 2) = cov_se3(5, 5);
    EXPECT_TRUE(lgmath::common::nearEqual(cov_se2_original, embedded_cov, 1e-12))
        << "SE(2) covariance not properly embedded for test " << i;
    
    // Check that z, roll, pitch components are zero
    EXPECT_NEAR(cov_se3(2, 2), 1e-6, 1e-6) << "Z variance not near zero for test " << i;
    EXPECT_NEAR(cov_se3(3, 3), 1e-6, 1e-6) << "Roll variance not near zero for test " << i;
    EXPECT_NEAR(cov_se3(4, 4), 1e-6, 1e-6) << "Pitch variance not near zero for test " << i;
    
    // Check cross-correlations with z, roll, pitch are zero
    for (int j = 0; j < 6; j++) {
      if (j != 2) {
        EXPECT_NEAR(cov_se3(2, j), 1e-6, 1e-6) << "Z cross-correlation not near zero";
      }
      if (j != 3) {
        EXPECT_NEAR(cov_se3(3, j), 1e-6, 1e-6) << "Roll cross-correlation not near zero";
      }
      if (j != 4) {
        EXPECT_NEAR(cov_se3(4, j), 1e-6, 1e-6) << "Pitch cross-correlation not near zero";
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test consistency between basic and covariance transformations
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, ConsistencyBetweenBasicAndCovarianceTransformations) {
  // Number of random tests
  const unsigned numTests = 10;

  // Generate random test cases
  for (unsigned i = 0; i < numTests; i++) {
    // Generate random SE(2) transformation
    Eigen::Matrix<double, 3, 1> se2_vec = Eigen::Matrix<double, 3, 1>::Random();
    lgmath::se2::Transformation T_se2_basic(se2_vec);
    lgmath::se2::TransformationWithCovariance T_se2_cov(se2_vec);
    
    // Generate random SE(3) transformation  
    Eigen::Matrix<double, 6, 1> se3_vec = Eigen::Matrix<double, 6, 1>::Random();
    lgmath::se3::Transformation T_se3_basic(se3_vec);
    lgmath::se3::TransformationWithCovariance T_se3_cov(se3_vec);
    
    // Test SE(2) → SE(3) consistency
    lgmath::se3::Transformation T_se3_from_basic = T_se2_basic.toSE3();
    lgmath::se3::TransformationWithCovariance T_se3_from_cov = T_se2_cov.toSE3();
    
    EXPECT_TRUE(lgmath::common::nearEqual(
        T_se3_from_basic.matrix(), T_se3_from_cov.matrix(), 1e-12))
        << "SE(2)→SE(3) conversion inconsistent between basic and covariance versions";
    
    // Test SE(3) → SE(2) consistency
    lgmath::se2::Transformation T_se2_from_basic = T_se3_basic.toSE2();
    lgmath::se2::TransformationWithCovariance T_se2_from_cov = T_se3_cov.toSE2();
    
    EXPECT_TRUE(lgmath::common::nearEqual(
        T_se2_from_basic.matrix(), T_se2_from_cov.matrix(), 1e-12))
        << "SE(3)→SE(2) conversion inconsistent between basic and covariance versions";
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test warning for SE(3) to SE(2) projection with significant z, roll, pitch
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, SE3ToSE2ProjectionWarning) {
  // Capture stderr output
  std::streambuf* original_cerr_buf = std::cerr.rdbuf();
  std::ostringstream captured_cerr;
  std::cerr.rdbuf(captured_cerr.rdbuf());

  // Create SE(3) transformation with significant z, roll, pitch
  Eigen::Matrix<double, 6, 1> se3_vec;
  se3_vec << 1.0, 1.0, 1.0, M_PI/6, M_PI/6, M_PI/4; // Significant z, roll, pitch
  lgmath::se3::Transformation T_se3(se3_vec);

  // Convert to SE(2), should trigger warning
  lgmath::se2::Transformation T_se2 = T_se3.toSE2();

  // Restore original cerr buffer
  std::cerr.rdbuf(original_cerr_buf);

  // Check that warning was issued
  std::string cerr_output = captured_cerr.str();
  EXPECT_NE(cerr_output.find("Warning: SE(3) has significant z, roll, or pitch component"), std::string::npos)
      << "Expected warning not issued for SE(3) to SE(2) projection with significant z, roll, pitch";
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
