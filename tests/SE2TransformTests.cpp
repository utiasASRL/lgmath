//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SE2TransformTests.cpp
/// \brief Unit tests for the implementation of the SE(2) transformation matrix class.
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
#include <lgmath/se2/Transformation.hpp>
#include <lgmath/so2/Operations.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////
///
/// UNIT TESTS OF SE(2) TRANSFORMATION MATRIX
///
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SE(2) transformation constructors
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, SE2TransformationConstructors) {
  // Generate random transform from most basic constructor
  Eigen::Matrix<double, 2, 2> C_ab =
      lgmath::so2::vec2rot(Eigen::Matrix<double, 1, 1>::Random()(0));
  Eigen::Matrix<double, 2, 1> r_ba_ina = Eigen::Matrix<double, 2, 1>::Random();
  lgmath::se2::Transformation rand(C_ab, r_ba_ina);

  // Transformation();
  {
    lgmath::se2::Transformation tmatrix;
    Eigen::Matrix3d test = Eigen::Matrix3d::Identity();
    std::cout << "tmat: " << tmatrix.matrix() << std::endl;
    std::cout << "test: " << test << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(tmatrix.matrix(), test, 1e-6));
  }

  // Transformation(const Transformation& T);
  {
    lgmath::se2::Transformation test(rand);
    std::cout << "tmat: " << rand.matrix() << std::endl;
    std::cout << "test: " << test.matrix() << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(rand.matrix(), test.matrix(), 1e-6));
  }

  // Transformation(const Eigen::Matrix3d& T);
  {
    lgmath::se2::Transformation test(rand.matrix());
    std::cout << "tmat: " << rand.matrix() << std::endl;
    std::cout << "test: " << test.matrix() << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(rand.matrix(), test.matrix(), 1e-6));
  }

  // Transformation& operator=(Transformation T);
  {
    lgmath::se2::Transformation test = rand;
    std::cout << "tmat: " << rand.matrix() << std::endl;
    std::cout << "test: " << test.matrix() << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(rand.matrix(), test.matrix(), 1e-6));
  }

  // Transformation(const Eigen::Matrix<double,3,1>& vec);
  {
    Eigen::Matrix<double, 3, 1> vec = Eigen::Matrix<double, 3, 1>::Random();
    Eigen::Matrix3d tmat = lgmath::se2::vec2tran(vec);
    lgmath::se2::Transformation test(vec);
    std::cout << "tmat: " << tmat << std::endl;
    std::cout << "test: " << test.matrix() << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(tmat, test.matrix(), 1e-6));
  }

  // Transformation(const Eigen::Matrix2d& C_ab,
  //               const Eigen::Vector2d& r_ba_ina);
  {
    lgmath::se2::Transformation tmat(C_ab, r_ba_ina);
    Eigen::Matrix3d test = Eigen::Matrix3d::Identity();
    test.topLeftCorner<2, 2>() = C_ab;
    test.topRightCorner<2, 1>() = r_ba_ina;
    std::cout << "tmat: " << tmat.matrix() << std::endl;
    std::cout << "test: " << test << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(tmat.matrix(), test, 1e-6));
  }

  // Transformation(Transformation&&);
  {
    auto rand2 = rand;
    lgmath::se2::Transformation test(std::move(rand));
    rand = rand2;

    std::cout << "tmat: " << test.matrix() << std::endl;
    std::cout << "test: " << rand.matrix() << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(test.matrix(), rand.matrix(), 1e-6));
  }

  // Transformation = Transformation&&;
  {
    lgmath::se2::Transformation test;
    auto rand2 = rand;
    test = std::move(rand);
    rand = rand2;

    std::cout << "tmat: " << test.matrix() << std::endl;
    std::cout << "test: " << rand.matrix() << std::endl;
    EXPECT_TRUE(lgmath::common::nearEqual(test.matrix(), rand.matrix(), 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test some get methods for SE(2) transformation
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, SE2TransformationGetMethods) {
  // Generate random transform from most basic constructor
  Eigen::Matrix<double, 2, 2> C_ab =
      lgmath::so2::vec2rot(Eigen::Matrix<double, 1, 1>::Random()(0));
  Eigen::Matrix<double, 2, 1> r_ba_ina = Eigen::Matrix<double, 2, 1>::Random();
  lgmath::se2::Transformation T_ab(C_ab, r_ba_ina);

  // Construct simple eigen matrix from random rotation and translation
  Eigen::Matrix3d test = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 2, 1> r_ab_inb = -C_ab.transpose() * r_ba_ina;
  test.topLeftCorner<2, 2>() = C_ab;
  test.topRightCorner<2, 1>() = r_ba_ina;

  // Test matrix()
  std::cout << "T_ab: " << T_ab.matrix() << std::endl;
  std::cout << "test: " << test << std::endl;
  EXPECT_TRUE(lgmath::common::nearEqual(T_ab.matrix(), test, 1e-6));

  // Test C_ab()
  std::cout << "T_ab: " << T_ab.C_ab() << std::endl;
  std::cout << "C_ab: " << C_ab << std::endl;
  EXPECT_TRUE(lgmath::common::nearEqual(T_ab.C_ab(), C_ab, 1e-6));

  // Test r_ba_ina()
  std::cout << "T_ab: " << T_ab.r_ba_ina() << std::endl;
  std::cout << "r_ba_ina: " << r_ba_ina << std::endl;
  EXPECT_TRUE(lgmath::common::nearEqual(T_ab.r_ba_ina(), r_ba_ina, 1e-6));

  // Test r_ab_inb()
  std::cout << "T_ab: " << T_ab.r_ab_inb() << std::endl;
  std::cout << "r_ab_inb: " << r_ab_inb << std::endl;
  EXPECT_TRUE(lgmath::common::nearEqual(T_ab.r_ab_inb(), r_ab_inb, 1e-6));
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test exponential map construction and logarithmic vec() method for SE(2)
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, SE2TransformationToFromSE2Algebra) {
  // Add vectors to be tested
  std::vector<Eigen::Matrix<double, 3, 1> > trueVecs;
  Eigen::Matrix<double, 3, 1> temp;
  temp << 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 1.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 1.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 1.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, -lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.5 * lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << 1.0, 1.0, 0.5 * lgmath::constants::PI;
  trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Calc transformation matrices
  std::vector<Eigen::Matrix3d> transMatrices;
  for (unsigned i = 0; i < numTests; i++) {
    transMatrices.push_back(lgmath::se2::vec2tran(trueVecs.at(i)));
  }

  // Calc transformations
  std::vector<lgmath::se2::Transformation> transformations;
  for (unsigned i = 0; i < numTests; i++) {
    transformations.push_back(lgmath::se2::Transformation(trueVecs.at(i)));
  }

  // Compare matrices
  {
    for (unsigned i = 0; i < numTests; i++) {
      std::cout << "matr: " << transMatrices.at(i) << std::endl;
      std::cout << "tran: " << transformations.at(i).matrix() << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          transMatrices.at(i), transformations.at(i).matrix(), 1e-6));
    }
  }

  // Test logarithmic map
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 3, 1> testVec = transformations.at(i).vec();
      std::cout << "true: " << trueVecs.at(i) << std::endl;
      std::cout << "func: " << testVec << std::endl;
      EXPECT_TRUE(
          lgmath::common::nearEqual(trueVecs.at(i), testVec, 1e-6));
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Test inverse, adjoint and operations for SE(2)
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(LGMath, SE2TransformationInverse) {
  // Add vectors to be tested
  std::vector<Eigen::Matrix<double, 3, 1> > trueVecs;
  Eigen::Matrix<double, 3, 1> temp;
  temp << 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 1.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 1.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 1.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, -lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.5 * lgmath::constants::PI;
  trueVecs.push_back(temp);
  temp << 1.0, 1.0, 0.5 * lgmath::constants::PI;
  trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 3, 1>::Random());
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Add vectors to be tested - random points
  std::vector<Eigen::Matrix<double, 3, 1> > landmarks;
  for (unsigned i = 0; i < numTests; i++) {
    landmarks.push_back(Eigen::Matrix<double, 3, 1>::Random());
  }

  // Calc transformation matrices
  std::vector<Eigen::Matrix3d> transMatrices;
  for (unsigned i = 0; i < numTests; i++) {
    transMatrices.push_back(lgmath::se2::vec2tran(trueVecs.at(i)));
  }

  // Calc transformations
  std::vector<lgmath::se2::Transformation> transformations;
  for (unsigned i = 0; i < numTests; i++) {
    transformations.push_back(lgmath::se2::Transformation(trueVecs.at(i)));
  }

  // Compare inverse to basic matrix inverse
  {
    for (unsigned i = 0; i < numTests; i++) {
      std::cout << "matr: " << transMatrices.at(i).inverse() << std::endl;
      std::cout << "tran: " << transformations.at(i).inverse().matrix()
                << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(
          transMatrices.at(i).inverse(),
          transformations.at(i).inverse().matrix(), 1e-6));
    }
  }

  // Test that product of inverse and self make identity
  {
    for (unsigned i = 0; i < numTests; i++) {
      std::cout << "T*Tinv: "
                << transformations.at(i).matrix() *
                       transformations.at(i).inverse().matrix();
      EXPECT_TRUE(lgmath::common::nearEqual(
          transformations.at(i).matrix() *
              transformations.at(i).inverse().matrix(),
          Eigen::Matrix3d::Identity(), 1e-6));
    }
  }

  // Test adjoint
  {
    for (unsigned i = 0; i < numTests; i++) {
      // Calculate expected adjoint matrix for SE(2)
      // Adjoint(T_ab) = [C_ab   -S*r_ba_ina] where S = [0 -1; 1 0]
      //                 [ 0^T        1     ]
      Eigen::Matrix<double, 3, 3> expectedAdjoint = Eigen::Matrix3d::Identity();
      Eigen::Matrix2d C = transformations.at(i).C_ab();
      Eigen::Vector2d r = transformations.at(i).r_ba_ina();
      Eigen::Matrix2d S;
      S << 0, -1, 1, 0;
      expectedAdjoint.topLeftCorner<2, 2>() = C;
      expectedAdjoint.topRightCorner<2, 1>() = -S * r;
      
      std::cout << "expected: " << expectedAdjoint << std::endl;
      std::cout << "tran: " << transformations.at(i).adjoint() << std::endl;
      EXPECT_TRUE(
          lgmath::common::nearEqual(expectedAdjoint,
                                    transformations.at(i).adjoint(), 1e-6));
    }
  }

  // Test self-product
  {
    for (unsigned i = 0; i < numTests - 1; i++) {
      lgmath::se2::Transformation test = transformations.at(i);
      test *= transformations.at(i + 1);
      Eigen::Matrix3d matrix = transMatrices.at(i) * transMatrices.at(i + 1);
      std::cout << "matr: " << matrix << std::endl;
      std::cout << "tran: " << test.matrix() << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(matrix, test.matrix(), 1e-6));
    }
  }

  // Test product
  {
    for (unsigned i = 0; i < numTests - 1; i++) {
      lgmath::se2::Transformation test =
          transformations.at(i) * transformations.at(i + 1);
      Eigen::Matrix3d matrix = transMatrices.at(i) * transMatrices.at(i + 1);
      std::cout << "matr: " << matrix << std::endl;
      std::cout << "tran: " << test.matrix() << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(matrix, test.matrix(), 1e-6));
    }
  }

  // Test self product with inverse
  {
    for (unsigned i = 0; i < numTests - 1; i++) {
      lgmath::se2::Transformation test = transformations.at(i);
      test /= transformations.at(i + 1);
      Eigen::Matrix3d matrix =
          transMatrices.at(i) * transMatrices.at(i + 1).inverse();
      std::cout << "matr: " << matrix << std::endl;
      std::cout << "tran: " << test.matrix() << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(matrix, test.matrix(), 1e-6));
    }
  }

  // Test product with inverse
  {
    for (unsigned i = 0; i < numTests - 1; i++) {
      lgmath::se2::Transformation test =
          transformations.at(i) / transformations.at(i + 1);
      Eigen::Matrix3d matrix =
          transMatrices.at(i) * transMatrices.at(i + 1).inverse();
      std::cout << "matr: " << matrix << std::endl;
      std::cout << "tran: " << test.matrix() << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(matrix, test.matrix(), 1e-6));
    }
  }

  // Test product with landmark
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 3, 1> mat = transMatrices.at(i) * landmarks.at(i);
      Eigen::Matrix<double, 3, 1> test =
          transformations.at(i) * landmarks.at(i);

      std::cout << "matr: " << mat << std::endl;
      std::cout << "test: " << test << std::endl;
      EXPECT_TRUE(lgmath::common::nearEqual(mat, test, 1e-6));
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
