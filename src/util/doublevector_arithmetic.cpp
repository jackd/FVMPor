#include <util/doublevector.h>
#include <math.h>

namespace util {

DoubleVector& DoubleVector::operator=(double d) {
    difference_type sz = size();
    double* ptr = data();
    for (difference_type i = 0; i < sz; ++i) {
        //(*this)[i] = d;
        ptr[i] = d;
    }
    return *this;
}

DoubleVector& DoubleVector::operator+=(double d) {
    difference_type sz = size();
    double* ptr = data();
    for (difference_type i = 0; i < sz; ++i) {
        //(*this)[i] += d;
        ptr[i] += d;
    }
    return *this;
}

DoubleVector& DoubleVector::operator-=(double d) {
    difference_type sz = size();
    double* ptr = data();
    for (difference_type i = 0; i < sz; ++i) {
        //(*this)[i] -= d;
        ptr[i] -= d;
    }
    return *this;
}

DoubleVector& DoubleVector::operator*=(double d) {
    difference_type sz = size();
    double* ptr = data();
    for (difference_type i = 0; i < sz; ++i) {
        //(*this)[i] *= d;
        ptr[i] *= d;
    }
    return *this;
}

DoubleVector& DoubleVector::operator/=(double d) {
    difference_type sz = size();
    double* ptr = data();
    for (difference_type i = 0; i < sz; ++i) {
        //(*this)[i] /= d;
        ptr[i] /= d;
    }
    return *this;
}

DoubleVector& DoubleVector::raise_to(double d) {
    difference_type sz = size();
    double* ptr = data();
    for (difference_type i = 0; i < sz; ++i) {
        ptr[i] = pow(ptr[i], d);
    }
    return *this;
}

DoubleVector& DoubleVector::sqroot() {
    difference_type sz = size();
    double* ptr = data();
    for (difference_type i = 0; i < sz; ++i) {
        ptr[i] = sqrt(ptr[i]);
    }
    return *this;
}

DoubleVector& DoubleVector::by_sqroot(const DoubleVector& v) {
    difference_type sz = size();
    double* ptr = data();
    const double* ptr2 = v.data();
    for (difference_type i = 0; i < sz; ++i) {
        ptr[i] *= sqrt(ptr2[i]);
    }
    return *this;
}

DoubleVector& DoubleVector::equals_prod( const DoubleVector& u, const DoubleVector& v){
    difference_type sz = size();
    double* ptr = data();
    const double* ptr1 = u.data();
    const double* ptr2 = v.data();
    for (difference_type i = 0; i < sz; ++i) {
        ptr[i] = ptr1[i]*ptr2[i];
    }
    return *this;
}

DoubleVector& DoubleVector::plus_equals_prod( const DoubleVector& u, const DoubleVector& v){
    difference_type sz = size();
    double* ptr = data();
    const double* ptr1 = u.data();
    const double* ptr2 = v.data();
    for (difference_type i = 0; i < sz; ++i) {
        ptr[i] += ptr1[i]*ptr2[i];
    }
    return *this;
}

DoubleVector& DoubleVector::operator+=(const DoubleVector& v) {
    #ifdef VECTOR_DEBUG
    if (size() != v.size()) throw std::logic_error("DoubleVector::operator+=(v): size() != v.size()");
    #endif
    double* lhs = data();
    const double* rhs = v.data();
    difference_type sz = size();
    for (difference_type i = 0; i < sz; ++i) {
        //(*this)[i] += v[i];
        lhs[i] += rhs[i];
    }
    return *this;
}

DoubleVector& DoubleVector::operator-=(const DoubleVector& v) {
    #ifdef VECTOR_DEBUG
    if (size() != v.size()) throw std::logic_error("DoubleVector::operator-=(v): size() != v.size()");
    #endif
    double* lhs = data();
    const double* rhs = v.data();
    difference_type sz = size();
    for (difference_type i = 0; i < sz; ++i) {
        //(*this)[i] -= v[i];
        lhs[i] -= rhs[i];
    }
    return *this;
}

DoubleVector& DoubleVector::operator*=(const DoubleVector& v) {
    #ifdef VECTOR_DEBUG
    if (size() != v.size()) throw std::logic_error("DoubleVector::operator*=(v): size() != v.size()");
    #endif
    double* lhs = data();
    const double* rhs = v.data();
    difference_type sz = size();
    for (difference_type i = 0; i < sz; ++i) {
        //(*this)[i] *= v[i];
        lhs[i] *= rhs[i];
    }
    return *this;
}

DoubleVector& DoubleVector::operator/=(const DoubleVector& v) {
    #ifdef VECTOR_DEBUG
    if (size() != v.size()) throw std::logic_error("DoubleVector::operator/=(v): size() != v.size()");
    #endif
    double* lhs = data();
    const double* rhs = v.data();
    difference_type sz = size();
    for (difference_type i = 0; i < sz; ++i) {
        //(*this)[i] /= v[i];
        lhs[i] /= rhs[i];
    }
    return *this;
}

DoubleVector& DoubleVector::raise_to(const DoubleVector& v) {
    #ifdef VECTOR_DEBUG
    if (size() != v.size()) throw std::logic_error("DoubleVector::raise_to(v): size() != v.size()");
    #endif
    difference_type sz = size();
    for (difference_type i = 0; i < sz; ++i) {
        (*this)[i] = pow((*this)[i], v[i]);
    }
    return *this;
}

// this = a*v
void DoubleVector::scal_assign(double a, const DoubleVector &v){
    #ifdef VECTOR_DEBUG
    if (size() != v.size()) throw std::logic_error("DoubleVector::raise_to(v): size() != v.size()");
    #endif
    double* lhs = data();
    const double* rhs = v.data();
    difference_type sz = size();
    for (difference_type i = 0; i < sz; ++i) {
        lhs[i] = a*rhs[i];
    }
}

// this += a*v
void DoubleVector::axpy(double a, const DoubleVector &v){
    #ifdef VECTOR_DEBUG
    if (size() != v.size()) throw std::logic_error("DoubleVector::raise_to(v): size() != v.size()");
    #endif
    double* lhs = data();
    const double* rhs = v.data();
    difference_type sz = size();
    for (difference_type i = 0; i < sz; ++i) {
        lhs[i] = a*rhs[i];
    }
}

// this[0:p.size-1] = v[p] 
void DoubleVector::permute_assign(const DoubleVector &v, const IntVector &p){
    #ifdef VECTOR_DEBUG
    //if (size() != v.size()) throw std::logic_error("DoubleVector::permute_assign(v): size() != v.size()");
    //if (size() != p.size()) throw std::logic_error("DoubleVector::permute_assign(v): size() != p.size()");
    #endif
    double* lhs = data();
    const double* rhs = v.data();
    difference_type sz = p.size();
    for (difference_type i = 0; i < sz; ++i) {
        lhs[i] = rhs[p[i]];
    }
}

// this[p] = v[0:p.size-1] 
void DoubleVector::permute_assign_inverse(const DoubleVector &v, const IntVector &p){
    #ifdef VECTOR_DEBUG
    //if (size() != v.size()) throw std::logic_error("DoubleVector::permute_assign_inverse(v): size() != v.size()");
    //if (size() != p.size()) throw std::logic_error("DoubleVector::permute_assign_inverse(v): size() != p.size()");
    #endif
    double* lhs = data();
    const double* rhs = v.data();
    difference_type sz = p.size();
    for (difference_type i = 0; i < sz; ++i) {
        lhs[p[i]] = rhs[i];
    }
}

void DoubleVector::permute_add_inverse(const DoubleVector &v, const IntVector &p){
    #ifdef VECTOR_DEBUG
    //if (size() != v.size()) throw std::logic_error("DoubleVector::permute_add_inverse(v): size() != v.size()");
    //if (size() != p.size()) throw std::logic_error("DoubleVector::permute_add_inverse(v): size() != p.size()");
    #endif
    double* lhs = data();
    const double* rhs = v.data();
    difference_type sz = p.size();
    for (difference_type i = 0; i < sz; ++i) {
        lhs[p[i]] += rhs[i];
    }
}

void DoubleVector::permute_add_weighted_pq(const DoubleVector &v, const IntVector &lhs_p, const IntVector &rhs_p, const DoubleVector &w){
    #ifdef VECTOR_DEBUG
    //if (size() != v.size()) throw std::logic_error("DoubleVector::permute_add_inverse(v): size() != v.size()");
    //if (size() != p.size()) throw std::logic_error("DoubleVector::permute_add_inverse(v): size() != p.size()");
    #endif
    double* lhs = data();
    const double* rhs = v.data();
    difference_type sz = lhs_p.size();
    for (difference_type i = 0; i < sz; ++i) {
        lhs[lhs_p[i]] += w[i]*rhs[rhs_p[i]];
    }
}

void DoubleVector::permute_add_weighted_pqr(const DoubleVector &v, const IntVector &lhs_p, const IntVector &rhs_p, const IntVector &w_p, const DoubleVector &w){
    #ifdef VECTOR_DEBUG
    //if (size() != v.size()) throw std::logic_error("DoubleVector::permute_add_inverse(v): size() != v.size()");
    //if (size() != p.size()) throw std::logic_error("DoubleVector::permute_add_inverse(v): size() != p.size()");
    #endif
    double* lhs = data();
    const double* rhs = v.data();
    difference_type sz = lhs_p.size();
    for (difference_type i = 0; i < sz; ++i) {
        lhs[lhs_p[i]] += w[w_p[i]]*rhs[rhs_p[i]];
    }
}

void DoubleVector::permute_add_weighted_inverse(const DoubleVector &v, const IntVector &p, const DoubleVector &w){
    #ifdef VECTOR_DEBUG
    //if (size() != v.size()) throw std::logic_error("DoubleVector::permute_add_inverse(v): size() != v.size()");
    //if (size() != p.size()) throw std::logic_error("DoubleVector::permute_add_inverse(v): size() != p.size()");
    #endif
    double* lhs = data();
    const double* rhs = v.data();
    difference_type sz = p.size();
    for (difference_type i = 0; i < sz; ++i) {
        lhs[p[i]] += w[i]*rhs[i];
    }
}

} // end namespace util
