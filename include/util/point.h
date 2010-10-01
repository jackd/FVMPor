#ifndef POINT_H
#define POINT_H

//#include <ostream>
#include <iostream>
#include <cmath>

namespace util {

struct Point {
    Point() : x(0.0), y(0.0), z(0.0) {}
    Point(double X, double Y, double Z)
        : x(X), y(Y), z(Z) {}
    double x;
    double y;
    double z;
    Point& operator+=(const Point& p);
    Point& operator-=(const Point& p);
    Point& operator*=(double c);
    Point& operator/=(double c);
};

inline Point& Point::operator+=(const Point& p) {
    x += p.x; y += p.y; z += p.z;
    return *this;
}

inline Point& Point::operator-=(const Point& p) {
    x -= p.x; y -= p.y; z -= p.z;
    return *this;
}

inline Point& Point::operator*=(double c) {
    x *= c; y *= c; z *= c;
    return *this;
}

inline Point& Point::operator/=(double c) {
    x /= c; y /= c; z /= c;
    return *this;
}

inline Point operator+(const Point& p1, const Point& p2) {
    return Point(p1.x + p2.x, p1.y + p2.y, p1.z + p2.z);
}

inline Point operator-(const Point& p1, const Point& p2) {
    return Point(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
}

inline Point operator*(double c, const Point& p) {
    return Point(c*p.x, c*p.y, c*p.z);
}

inline Point operator*(const Point& p, double c) {
    return Point(p.x*c, p.y*c, p.z*c);
}

inline Point operator/(const Point& p, double c) {
    return Point(p.x/c, p.y/c, p.z/c);
}

inline Point operator-(const Point p) {
    return Point(-p.x, -p.y, -p.z);
}

inline bool operator==(const Point& p1, const Point& p2) {
    return (p1.x == p2.x && p1.y == p2.y && p1.z == p2.z);
}

inline bool operator<(const Point& p1, const Point& p2) {
    if (p1.x < p2.x) return true;
    if (p1.x > p2.x) return false;
    if (p1.y < p2.y) return true;
    if (p1.y > p2.y) return false;
    return (p1.z < p2.z);
}

inline bool operator!=(const Point& p1, const Point& p2) {
    return !(p1 == p2);
}

inline bool operator>(const Point& p1, const Point& p2) {
    return p2 < p1;
}

inline bool operator<=(const Point& p1, const Point& p2) {
    return !(p1 > p2);
}

inline bool operator>=(const Point& p1, const Point& p2) {
    return !(p1 < p2);
}

inline double dot(const Point& p1, const Point& p2) {
    return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

inline Point cross(const Point& p1, const Point& p2) {
    return Point(p1.y*p2.z-p1.z*p2.y, p1.z*p2.x-p1.x*p2.z, p1.x*p2.y-p1.y*p2.x);
}

inline double norm(const Point& p) {
    return std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

inline double distance(const Point& p1, const Point& p2) {
    return norm(p1-p2);
}

inline std::ostream& operator<<(std::ostream& os, const Point& p) {
    return os << '[' << p.x << ", " << p.y << ", " << p.z << ']';
}

inline std::istream& operator>>(std::istream& is, Point& p) {
    is >> p.x >> p.y >> p.z;
    return is;
}

} // end namespace util

#endif
