#ifndef MESH_QUAD_H
#define MESH_QUAD_H



#include <cassert>
#include <cmath>
#include <ostream>

namespace mesh {

class CVFace_shape {
public:
    CVFace_shape(Point p0, Point p1, Point p2, Point p3) {
        p[0] = p0; p[1] = p1; p[2] = p2; p[3] = p3; points_ = 4;
        // assert coplanar
#ifndef NDEBUG
        using std::abs;
        using std::max;
        double p0x = p[0].x - p[3].x;
        double p0y = p[0].y - p[3].y;
        double p0z = p[0].z - p[3].z;
        double p1x = p[1].x - p[3].x;
        double p1y = p[1].y - p[3].y;
        double p1z = p[1].z - p[3].z;
        double p2x = p[2].x - p[3].x;
        double p2y = p[2].y - p[3].y;
        double p2z = p[2].z - p[3].z;
        double term1 = p0x*p1y*p2z-p0x*p1z*p2y;
        double term2 = p1x*p2y*p0z-p1x*p0y*p2z;
        double term3 = p2x*p0y*p1z-p2x*p1y*p0z;
        double maxterm = max(max(abs(term1), abs(term2)), abs(term3));
        double tol = 1.0e-7 * max(1.0, maxterm);
#endif
        assert(abs(term1+term2+term3) <= tol);
    }
    CVFace_shape(Point p0, Point p1) {
        p[0] = p0;
        p[1] = p1;
        points_ = 2;
    }

    // return the number of points in the face
    int points() const {
        return points_;
    }

    // the "area" of the face (is actually the length of the line segment for 2D meshes)
    double area() const {
        switch( points() ){
            case 2: // line - 2D mesh
                return norm(p[1] - p[0]);
            case 4: // quadrilateral - 3D mesh
                return (norm(cross(p[1] - p[0], p[3] - p[0])) +
                        norm(cross(p[1] - p[2], p[3] - p[2]))) / 2.0;
            default:
                assert(false);
                return 0.0;
        }
    }

    // the centroid of the face
    Point centroid() const {
        switch( points() ){
            default:
                assert(false);
                return Point();
            case 2: // line - 2D mesh
                return (p[0] + p[1])/2;
            case 4: // quad - 3D mesh
                Point centroid1 = (p[0] + p[1] + p[3]) / 3.0;
                Point centroid2 = (p[1] + p[2] + p[3]) / 3.0;
                double area1 = norm(cross(p[1] - p[0], p[3] - p[0])) / 2.0;
                double area2 = norm(cross(p[1] - p[2], p[3] - p[2])) / 2.0;
                return (area1*centroid1 + area2*centroid2) / (area1+area2);
        }
    }

    // calculate the normal to the face
    Point normal() const {
        Point result;
        switch( points() ){
            case 2: // line - 2D mesh
                result = cross( p[1]-p[0], Point(0,0,1) );
                return result / norm(result);
            case 4: // quad - 3D mesh
                result = cross(p[1]-p[0], p[3]-p[0]);
                return result / norm(result);
            default:
                assert(false);
                return Point();
        }
    }
    Point point(int i) const {
        assert(i >= 0 && i < points());
        return p[i];
    }
    void reverse() {
        switch( points() ){
            case 2:
                std::swap(p[0], p[1]);
                break;
            case 4:
                std::swap(p[1], p[3]);
                break;
        }
    }
private:
    Point p[4];
    int points_;
};

inline
std::ostream& operator<<(std::ostream& os, const CVFace_shape& fs) {
    os << "face shape with " << fs.points() << " points : ";
    for( int i=0; i<fs.points(); i++ ){
        os << fs.point(i) << " ";
    }
    os << std::endl;
    os << "normal " << fs.normal() << " and area " << fs.area() << std::endl;
    return os;
}

} // end namespace mesh

#endif
