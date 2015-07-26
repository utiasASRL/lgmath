//////////////////////////////////////////////////////////////////////////////////////////////
/// \file Transformation.cpp
/// \brief Implementation file for a transformation matrix class.
/// \details Light weight transformation class, intended to be fast, and not to provide
///          unnecessary functionality.
///
/// \author Sean Anderson
//////////////////////////////////////////////////////////////////////////////////////////////

#include <lgmath/se3/TransformationWithCovariance.hpp>

#include <stdexcept>

#include <lgmath/so3/Operations.hpp>
#include <lgmath/se3/Operations.hpp>

namespace lgmath {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Default constructor
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance::TransformationWithCovariance() :
        Transformation(), U_(Eigen::Matrix<double,6,6>::Zero()), covarianceSet_(false) {
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Copy constructor
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance::TransformationWithCovariance(const TransformationWithCovariance& T) :
        Transformation(T), U_(T.U_), covarianceSet_(T.covarianceSet_) {

}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Copy constructor from deterministic Transformation
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance::TransformationWithCovariance(const Transformation& T, bool covarianceSet) :
        Transformation(T), U_(Eigen::Matrix<double,6,6>::Zero()), covarianceSet_(covarianceSet) {

//TODO: Decide whether explicitly calling TransformationWithCovariance(Transform) should set covarianceSet_ to TRUE
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Copy constructor from deterministic Transformation, with covariance
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance::TransformationWithCovariance(const Transformation& T, const Eigen::Matrix<double,6,6>& U) :
        Transformation(T), U_(U), covarianceSet_(true) {
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance::TransformationWithCovariance(const Eigen::Matrix4d& T, bool reproj) :
        Transformation(T, reproj), U_(Eigen::Matrix<double,6,6>::Zero()), covarianceSet_(false) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor with covariance
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance::TransformationWithCovariance(const Eigen::Matrix4d& T, const Eigen::Matrix<double,6,6>& U, bool reproj) :
        Transformation(T, reproj), U_(U), covarianceSet_(true) {
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor. The transformation will be T_ba = [C_ba, -C_ba*r_ba_ina; 0 0 0 1]
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance::TransformationWithCovariance(const Eigen::Matrix3d& C_ba, const Eigen::Vector3d& r_ba_ina, bool reproj) :
        Transformation(C_ba, r_ba_ina, reproj), U_(Eigen::Matrix<double,6,6>::Zero()), covarianceSet_(false) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor with covariance. The transformation will be
/// T_ba = [C_ba, -C_ba*r_ba_ina; 0 0 0 1]
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance::TransformationWithCovariance(const Eigen::Matrix3d& C_ba, const Eigen::Vector3d& r_ba_ina,
                             const Eigen::Matrix<double,6,6>& U, bool reproj) :
        Transformation(C_ba, r_ba_ina, reproj), U_(U), covarianceSet_(true) {
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor. The transformation will be T_ba = vec2tran(xi_ab)
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance::TransformationWithCovariance(const Eigen::Matrix<double,6,1>& xi_ab, unsigned int numTerms) :
        Transformation(xi_ab, numTerms), U_(Eigen::Matrix<double,6,6>::Zero()), covarianceSet_(false) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor with covariance. The transformation will be T_ba = vec2tran(xi_ab)
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance::TransformationWithCovariance(const Eigen::Matrix<double,6,1>& xi_ab,
                                                           const Eigen::Matrix<double,6,6>& U, unsigned int numTerms) :
        Transformation(xi_ab, numTerms), U_(U), covarianceSet_(true) {
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor. The transformation will be T_ba = vec2tran(xi_ab), xi_ab must be 6x1
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance::TransformationWithCovariance(const Eigen::VectorXd& xi_ab) :
        Transformation(xi_ab), U_(Eigen::Matrix<double,6,6>::Zero()), covarianceSet_(false) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor. The transformation will be T_ba = vec2tran(xi_ab), xi_ab must be 6x1
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance::TransformationWithCovariance(const Eigen::VectorXd& xi_ab, const Eigen::Matrix<double,6,6>& U) :
        Transformation(xi_ab), U_(U), covarianceSet_(true) {
}



//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Assignment operator. Note pass-by-value is intentional.
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance& TransformationWithCovariance::operator=(TransformationWithCovariance T) {
    // Call the assignment operator on the super class, as the internal members are not acessible here
    Transformation::operator=(T);

    // Swap (this)'s parameters with the temporary object passed by value
    // The temporary object is then destroyed at end of scope
    std::swap( this->U_, T.U_ );
    std::swap( this->covarianceSet_, T.covarianceSet_ );
    return (*this);
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Gets the underlying covariance matrix
//////////////////////////////////////////////////////////////////////////////////////////////
const Eigen::Matrix<double,6,6>& TransformationWithCovariance::U() const {
    if (!covarianceSet_) {
        throw std::logic_error("Covariance accessed before being set.  "
                               "Use setCovariance or initialize with a covariance.");
    }
    return U_;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Sets the underlying covariance matrix
//////////////////////////////////////////////////////////////////////////////////////////////
void TransformationWithCovariance::setCovariance(const Eigen::Matrix<double,6,6>& U) {
    U_ = U;
    covarianceSet_ = true;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Sets the underlying covariance matrix to the 6x6 zero matrix
//////////////////////////////////////////////////////////////////////////////////////////////
void TransformationWithCovariance::zeroCovariance() {
    U_ = Eigen::Matrix<double,6,6>::Zero();
    covarianceSet_ = true;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get the inverse matrix
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance TransformationWithCovariance::inverse() const {
    return TransformationWithCovariance(Transformation::inverse(), U_);
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief In-place right-hand side multiply T_rhs
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance& TransformationWithCovariance::operator*=(const TransformationWithCovariance& T_rhs) {
    Eigen::Matrix<double,6,6> Ad = Transformation::adjoint();
    Transformation::operator*=(T_rhs);
    this->U_ = this->U_ + Ad * T_rhs.U_ * Ad.transpose();
    this->covarianceSet_ = (this->covarianceSet_ && T_rhs.covarianceSet_);
    return *this;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief In-place right-hand side multiply deterministic T_rhs
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance& TransformationWithCovariance::operator*=(const Transformation& T_rhs) {
    Transformation::operator*=(T_rhs);
    return *this;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief In-place right-hand side multiply this matrix by the inverse of T_rhs
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance& TransformationWithCovariance::operator/=(const TransformationWithCovariance& T_rhs) {
    Eigen::Matrix<double,6,6> Ad = Transformation::adjoint();
    Transformation::operator/=(T_rhs);
    this->U_ = this->U_ + Ad * T_rhs.U_ * Ad.transpose();
    this->covarianceSet_ = (this->covarianceSet_ && T_rhs.covarianceSet_);
    return *this;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief In-place right-hand side multiply this matrix by the inverse of deterministic T_rhs
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance& TransformationWithCovariance::operator/=(const Transformation& T_rhs) {
    Transformation::operator/=(T_rhs);
    return *this;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Multiplication of TransformWithCovariance by TransformWithCovariance
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance operator*(TransformationWithCovariance T_lhs, const TransformationWithCovariance& T_rhs) {
    T_lhs *= T_rhs;
    return T_lhs;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Multiplication of TransformWithCovariance by Transform
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance operator*(TransformationWithCovariance T_lhs, const Transformation& T_rhs) {
    T_lhs *= T_rhs;
    return T_lhs;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Multiplication of Transform by TransformWithCovariance
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance operator*(const Transformation& T_lhs, const TransformationWithCovariance& T_rhs) {
    // Convert the Transform to a TransformWithCovariance, with the covarianceSet_ flag as true
    TransformationWithCovariance temp(T_lhs, true);
    temp *= T_rhs;
    return temp;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Multiplication of TransformWithCovariance by inverse TransformWithCovariance
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance operator/(TransformationWithCovariance T_lhs, const TransformationWithCovariance& T_rhs) {
    T_lhs /= T_rhs;
    return T_lhs;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Multiplication of TransformWithCovariance by inverse Transform
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance operator/(TransformationWithCovariance T_lhs, const Transformation& T_rhs) {
    T_lhs /= T_rhs;
    return T_lhs;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Multiplication of Transform by inverse TransformWithCovariance
//////////////////////////////////////////////////////////////////////////////////////////////
TransformationWithCovariance operator/(const Transformation& T_lhs, const TransformationWithCovariance& T_rhs) {
    // Convert the Transform to a TransformWithCovariance, with the covarianceSet_ flag as true
    TransformationWithCovariance temp(T_lhs, true);
    temp /= T_rhs;
    return temp;
}

} // se3
} // lgmath


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief print transformation
//////////////////////////////////////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& out, const lgmath::se3::TransformationWithCovariance& T) {
    out << std::endl << T.matrix() << std::endl;
    return out;
}
