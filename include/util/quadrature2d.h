#ifndef QUADRATURE2D_H
#define QUADRATURE2D_H

#include <util/point.h>

namespace util {

class Quadrature2D {
public:
    Quadrature2D(const Point& p1_, const Point& p2_,
                 const Point& p3_, const Point& p4_)
    {
        x1 = p1_.x; y1 = p1_.y; z1 = p1_.z;
        x2 = p2_.x; y2 = p2_.y; z2 = p2_.z;
        x3 = p3_.x; y3 = p3_.y; z3 = p3_.z;
        x4 = p4_.x; y4 = p4_.y; z4 = p4_.z;
    }
    template<typename F>
    double operator()(F f, int points) const;
private:
    double x(double u, double v) const;
    double y(double u, double v) const;
    double z(double u, double v) const;
    double duxdv(double u, double v) const;
    double x1, y1, z1;
    double x2, y2, z2;
    double x3, y3, z3;
    double x4, y4, z4;
};

inline double Quadrature2D::x(double u, double v) const {
    return (((x1-x4+x3-x2)*v+x3-x1-x4+x2)*u+(x3-x1+x4-x2)*v+x3+x1+x4+x2) / 4.0;
}

inline double Quadrature2D::y(double u, double v) const {
    return (((y1-y4+y3-y2)*v+y3-y1-y4+y2)*u+(y3-y1+y4-y2)*v+y3+y1+y4+y2) / 4.0;
}

inline double Quadrature2D::z(double u, double v) const {
    return (((z1-z4+z3-z2)*v+z3-z1-z4+z2)*u+(z3-z1+z4-z2)*v+z3+z1+z4+z2) / 4.0;
}

inline double Quadrature2D::duxdv(double u, double v) const {

    Point p(
        (-y1*z3+y4*z2-y3*z2+y1*z4-y4*z1+y3*z1+y2*z3-y2*z4)*u
      + (-y3*z1-y2*z4+y1*z3-y1*z2+y2*z1+y3*z4-y4*z3+y4*z2)*v
         -y4*z3+y2*z3-y2*z1-y3*z2+y4*z1+y1*z2-y1*z4+y3*z4,

        (z2*x3-z1*x3+z3*x1-z3*x2+z1*x4-z2*x4-z4*x1+z4*x2)*u
      +(-z3*x1+z1*x3+z4*x2-z4*x3-z1*x2-z2*x4+z2*x1+z3*x4)*v
        +z3*x4-z3*x2-z1*x4+z1*x2-z4*x3+z4*x1+z2*x3-z2*x1,

        (x4*y2-x2*y4+x2*y3-x4*y1-x3*y2+x1*y4-x1*y3+x3*y1)*u
      +(-x1*y2+x3*y4-x3*y1+x1*y3-x2*y4+x2*y1+x4*y2-x4*y3)*v
        +x4*y1+x2*y3-x2*y1-x1*y4+x1*y2-x3*y2+x3*y4-x4*y3);

    return norm(p) / 8.0;
}

template<typename F>
inline double Quadrature2D::operator()(F f, int points) const {

    double result = 0.0;

    assert(points >= 1 && points <= 3);

    if (points == 1) {

        result += 4 * f(x(0.0,0.0), y(0.0,0.0), z(0.0,0.0)) * duxdv(0.0, 0.0);

    } else if (points == 2) {

        double w[] = { 1.000000000000000, 1.000000000000000 };
        double r[] = {-0.577350269189626, 0.577350269189626 };

        for (unsigned i = 0; i < 2; ++i) {
            for (unsigned j = 0; j < 2; ++j) {
                result += w[i]*w[j]
                        * f(x(r[i], r[j]), y(r[i], r[j]), z(r[i], r[j]))
                        * duxdv(r[i], r[j]);
            }
        }

    } else {

        double w[] = { 0.555555555555556, 0.888888888888889, 0.555555555555556 };
        double r[] = {-0.774596669241483, 0.000000000000000, 0.774596669241483 };

        for (unsigned i = 0; i < 3; ++i) {
            for (unsigned j = 0; j < 3; ++j) {
                result += w[i]*w[j]
                        * f(x(r[i], r[j]), y(r[i], r[j]), z(r[i], r[j]))
                        * duxdv(r[i], r[j]);
            }
        }
    }

    return result;

}

} // end namespace util

#endif
