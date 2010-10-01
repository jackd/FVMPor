#ifndef QUADRATURE3D_H
#define QUADRATURE3D_H

#include <cmath>
#include <cassert>

#include <util/point.h>

namespace util {

class Quadrature3D {
public:
    Quadrature3D(const Point& p1_, const Point& p2_,
                 const Point& p3_, const Point& p4_,
                 const Point& p5_, const Point& p6_,
                 const Point& p7_, const Point& p8_)
    {
        x1 = p1_.x; y1 = p1_.y; z1 = p1_.z;
        x2 = p2_.x; y2 = p2_.y; z2 = p2_.z;
        x3 = p3_.x; y3 = p3_.y; z3 = p3_.z;
        x4 = p4_.x; y4 = p4_.y; z4 = p4_.z;
        x5 = p5_.x; y5 = p5_.y; z5 = p5_.z;
        x6 = p6_.x; y6 = p6_.y; z6 = p6_.z;
        x7 = p7_.x; y7 = p7_.y; z7 = p7_.z;
        x8 = p8_.x; y8 = p8_.y; z8 = p8_.z;
    }
    template<typename F>
    double operator()(F f, int points) const;
private:
    double x(double u, double v, double w) const;
    double y(double u, double v, double w) const;
    double z(double u, double v, double w) const;
    double J(double u, double v, double w) const;
    double x1, y1, z1;
    double x2, y2, z2;
    double x3, y3, z3;
    double x4, y4, z4;
    double x5, y5, z5;
    double x6, y6, z6;
    double x7, y7, z7;
    double x8, y8, z8;
};

inline double Quadrature3D::x(double u, double v, double w) const {
    return ((((x4-x3+x7+x2-x1-x6+x5-x8)*w
              +x7-x2-x8+x5-x6+x1-x4+x3)*v
             +(x7+x1+x4-x5+x6-x2-x8-x3)*w
              -x1+x2+x3-x4-x5+x6+x7-x8)*u
           +((-x4-x3+x7+x2+x1-x6-x5+x8)*w
              +x7-x2+x8-x5-x6-x1+x4+x3)*v
             +(x7-x1-x4+x5+x6-x2+x8-x3)*w
              +x1+x2+x3+x4+x5+x6+x7+x8) / 8.0;
}

inline double Quadrature3D::y(double u, double v, double w) const {
    return ((((y4-y3+y7+y2-y1-y6+y5-y8)*w
              +y7-y2-y8+y5-y6+y1-y4+y3)*v
             +(y7+y1+y4-y5+y6-y2-y8-y3)*w
              -y1+y2+y3-y4-y5+y6+y7-y8)*u
           +((-y4-y3+y7+y2+y1-y6-y5+y8)*w
              +y7-y2+y8-y5-y6-y1+y4+y3)*v
             +(y7-y1-y4+y5+y6-y2+y8-y3)*w
              +y1+y2+y3+y4+y5+y6+y7+y8) / 8.0;
}

inline double Quadrature3D::z(double u, double v, double w) const {
    return ((((z4-z3+z7+z2-z1-z6+z5-z8)*w
              +z7-z2-z8+z5-z6+z1-z4+z3)*v
             +(z7+z1+z4-z5+z6-z2-z8-z3)*w
              -z1+z2+z3-z4-z5+z6+z7-z8)*u
           +((-z4-z3+z7+z2+z1-z6-z5+z8)*w
              +z7-z2+z8-z5-z6-z1+z4+z3)*v
             +(z7-z1-z4+z5+z6-z2+z8-z3)*w
              +z1+z2+z3+z4+z5+z6+z7+z8) / 8.0;
}

inline double Quadrature3D::J(double u, double v, double w) const {

    double J11 = ((x4-x8-x3+x5+x7-x1+x2-x6)*w
                  +x7-x2-x8+x5-x6+x1-x4+x3)*v
                 +(x7+x1+x4-x5+x6-x2-x8-x3)*w
                  -x1+x2+x3-x4-x5+x6+x7-x8;

    double J12 = ((x4-x8-x3+x5+x7-x1+x2-x6)*w
                  +x7-x2-x8+x5-x6+x1-x4+x3)*u
                 +(x1+x8-x4-x5-x6+x7-x3+x2)*w
                  -x1-x2+x3+x4-x5-x6+x7+x8;

    double J13 = ((x4-x8-x3+x5+x7-x1+x2-x6)*v
                  +x7+x1+x4-x5+x6-x2-x8-x3)*u
                 +(x1+x8-x4-x5-x6+x7-x3+x2)*v
                  -x3-x4-x2+x6+x7+x8-x1+x5;

    double J21 = ((y7+y4-y8-y1+y2-y3-y6+y5)*w
                  -y8-y2+y1+y7-y6+y5-y4+y3)*v
                +(-y2+y1+y7-y5-y8+y4-y3+y6)*w
                  -y1+y2+y3+y7-y5-y8+y6-y4;

    double J22 = ((y7+y4-y8-y1+y2-y3-y6+y5)*w
                  -y8-y2+y1+y7-y6+y5-y4+y3)*u
                 +(y8-y4+y2+y1+y7-y5-y6-y3)*w
                  -y1+y7-y5-y2+y3+y8-y6+y4;

    double J23 = ((y7+y4-y8-y1+y2-y3-y6+y5)*v
                  -y2+y1+y7-y5-y8+y4-y3+y6)*u
                 +(y8-y4+y2+y1+y7-y5-y6-y3)*v
                  -y1+y6-y2-y3+y8+y7+y5-y4;

    double J31 = ((z4-z8+z2-z6+z7-z3-z1+z5)*w
                  -z8-z6-z4+z1+z3+z7-z2+z5)*v
                 +(z7-z5-z3-z8+z4-z2+z1+z6)*w
                  -z8-z5+z2+z6+z7-z1-z4+z3;

    double J32 = ((z4-z8+z2-z6+z7-z3-z1+z5)*w
                  -z8-z6-z4+z1+z3+z7-z2+z5)*u
                +(-z5-z6+z8-z4-z3+z1+z7+z2)*w
                  -z6+z7-z1-z2+z3+z8-z5+z4;

    double J33 = ((z4-z8+z2-z6+z7-z3-z1+z5)*v
                  +z7-z5-z3-z8+z4-z2+z1+z6)*u
                +(-z5-z6+z8-z4-z3+z1+z7+z2)*v
                  +z6+z7-z1-z2-z3+z8+z5-z4;

    return std::abs(
        J11*J22*J33-J11*J23*J32+J21*J32*J13-J21*J12*J33+J31*J12*J23-J31*J22*J13
    ) / 512.0;
}

template<typename F>
inline double Quadrature3D::operator()(F f, int points) const {

    double result = 0.0;

    assert(points >= 1 && points <= 4);

    if (points == 1) {

        result += 8
                * f(x(0.0,0.0,0.0), y(0.0,0.0,0.0), z(0.0,0.0,0.0))
                * J(0.0, 0.0, 0.0);

    } else if (points == 2) {

        double w[] = { 1.000000000000000, 1.000000000000000 };
        double r[] = {-0.577350269189626, 0.577350269189626 };

        for (unsigned i = 0; i < 2; ++i)
            for (unsigned j = 0; j < 2; ++j)
                for (unsigned k = 0; k < 2; ++k)
                    result += w[i]*w[j]*w[k]
                            * f(
                                x(r[i], r[j], r[k]),
                                y(r[i], r[j], r[k]),
                                z(r[i], r[j], r[k])
                                )
                            * J(r[i], r[j], r[k]);

    } else if (points == 3) {

        double w[] = { 0.555555555555556, 0.888888888888889, 0.555555555555556 };
        double r[] = {-0.774596669241483, 0.000000000000000, 0.774596669241483 };

        for (unsigned i = 0; i < 3; ++i)
            for (unsigned j = 0; j < 3; ++j)
                for (unsigned k = 0; k < 3; ++k)
                    result += w[i]*w[j]*w[k]
                            * f(
                                x(r[i], r[j], r[k]),
                                y(r[i], r[j], r[k]),
                                z(r[i], r[j], r[k])
                               )
                            * J(r[i], r[j], r[k]);

    } else {
        double w[] = { 0.347854845137454, 0.652145154862546,
                       0.652145154862546, 0.347854845137454 };
        double r[] = {-0.861136311594053,-0.339981043584856,
                       0.339981043584856, 0.861136311594053 };

        for (unsigned i = 0; i < 4; ++i)
            for (unsigned j = 0; j < 4; ++j)
                for (unsigned k = 0; k < 4; ++k)
                    result += w[i]*w[j]*w[k]
                            * f(
                                x(r[i], r[j], r[k]),
                                y(r[i], r[j], r[k]),
                                z(r[i], r[j], r[k])
                               )
                            * J(r[i], r[j], r[k]);
    }

    return result;

}

} // end namespace util

#endif
