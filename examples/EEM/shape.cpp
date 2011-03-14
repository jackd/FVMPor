#include "shape.h"
#include <cassert>
#include <iostream>

namespace shape {

using mesh::Point;

double Shape::det3(double a11, double a12, double a13,
                  double a21, double a22, double a23,
                  double a31, double a32, double a33)
{
    return a11*a22*a33 - a11*a23*a32 - a21*a12*a33 + a21*a13*a32 + a31*a12*a23 - a31*a13*a22;
}

double Shape::det2(double a11, double a12,
                  double a21, double a22 )
{
    return a11*a22 - a21*a12;
}

Shape::Shape(const mesh::Element& e)
    : shape_f(e.edges()), shape_g(e.edges())
{

    switch (e.type()) {
        case 2: // triangle
        {
            double s[] = {5./12., 5./12., 1./6.};
            double t[] = {1./6., 5./12., 5./12.};
            double x0 = e.node(0).point().x;
            double x1 = e.node(1).point().x;
            double x2 = e.node(2).point().x;
            double y0 = e.node(0).point().y;
            double y1 = e.node(1).point().y;
            double y2 = e.node(2).point().y;

            double A = 0;
            for( int i=0; i<e.scvs(); i++ ){
                A += e.scv(i).vol();
            }

            double dNdx[] = { (y1-y2)/(2*A), (y2-y0)/(2*A), (y0-y1)/(2*A) };
            double dNdy[] = { (x2-x1)/(2*A), (x0-x2)/(2*A), (x1-x0)/(2*A) };

            // for the linear element the gradient is invariant over the triangle, so
            // compute it outside the edge loop
            shape_gradient_vector g( e.nodes() );
            for( int i=0; i<e.nodes(); i++ ){
                g[i] = Point(dNdx[i], dNdy[i], 0);
            }

            for( int i=0; i<e.edges(); i++ ){
                double N[] = {1-t[i]-s[i], s[i], t[i]};
                shape_function_vector f(N, N + e.nodes());

                shape_f[i] = f;
                shape_g[i] = g;
            }
            break;
        }
        case 3: // quadrilateral
        {
            double r[] = {0,     -1./2., 0,      1./2.};
            double s[] = {1./2., 0,      -1./2., 0};
            for( int i=0; i<e.edges(); i++ ){
                double N[] = { .25*(1+r[i])*(1+s[i]), .25*(1-r[i])*(1+s[i]), .25*(1-r[i])*(1-s[i]), .25*(1+r[i])*(1-s[i])};
                shape_function_vector f(N, N + e.nodes());

                double dNdr[] = {.25*(1+s[i]), -.25*(1+s[i]), -.25*(1-s[i]),  .25*(1-s[i])};
                double dNds[] = {.25*(1+r[i]),  .25*(1-r[i]), -.25*(1-r[i]), -.25*(1+r[i])};
                double J11 = 0; double J12 = 0;
                double J21 = 0; double J22 = 0;
                for( int j=0; j<e.nodes(); j++ ){
                    Point p = e.node(j).point();
                    J11 += dNdr[j]*p.x;
                    J12 += dNdr[j]*p.y;
                    J21 += dNds[j]*p.x;
                    J22 += dNds[j]*p.y;
                }
                double detJ = det2(J11, J12, J21, J22);

                double drdx =  J22/detJ; double drdy = -J21/detJ;
                double dsdx = -J12/detJ; double dsdy =  J11/detJ;

                shape_gradient_vector g(e.nodes());
                for( int j=0; j<e.nodes(); j++ ){
                    double dx = drdx*dNdr[j] + dsdx*dNds[j];
                    double dy = drdy*dNdr[j] + dsdy*dNds[j];
                    g[j] = Point(dx, dy, 0);
                }

                shape_f[i] = f;
                shape_g[i] = g;
            }
            break;
        }
        case 4: // tetrahedron
        {
            double xi[] = {13.0 / 36.0, 13.0 / 36.0,  5.0 / 36.0,  5.0 / 36.0, 13.0 / 36.0,  5.0 / 36.0};
            double et[] = { 5.0 / 36.0, 13.0 / 36.0, 13.0 / 36.0,  5.0 / 36.0,  5.0 / 36.0, 13.0 / 36.0};
            double ze[] = { 5.0 / 36.0,  5.0 / 36.0,  5.0 / 36.0, 13.0 / 36.0, 13.0 / 36.0, 13.0 / 36.0};

            for (int i = 0; i < tet_edges; ++i) {

                double N[] = {1.0 - xi[i] - et[i] - ze[i], xi[i], et[i], ze[i]};
                double Nxi[] = {-1.0, 1.0, 0.0, 0.0};
                double Net[] = {-1.0, 0.0, 1.0, 0.0};
                double Nze[] = {-1.0, 0.0, 0.0, 1.0};

                double x_xi = 0; double x_et = 0; double x_ze = 0;
                double y_xi = 0; double y_et = 0; double y_ze = 0;
                double z_xi = 0; double z_et = 0; double z_ze = 0;

                for (int j = 0; j < tet_nodes; ++j) {
                    Point p = e.node(j).point();
                    x_xi += p.x * Nxi[j];  x_et += p.x * Net[j];  x_ze += p.x * Nze[j];
                    y_xi += p.y * Nxi[j];  y_et += p.y * Net[j];  y_ze += p.y * Nze[j];
                    z_xi += p.z * Nxi[j];  z_et += p.z * Net[j];  z_ze += p.z * Nze[j];
                }

                shape_function_vector f(N, N + tet_nodes);
                shape_gradient_vector g(tet_nodes);
                double del = det3(x_xi, y_xi, z_xi, x_et, y_et, z_et, x_ze, y_ze, z_ze);
                for (int j = 0; j < tet_nodes; ++j) {
                    double dx = det3(Nxi[j], y_xi, z_xi, Net[j], y_et, z_et, Nze[j], y_ze, z_ze);
                    double dy = det3(x_xi, Nxi[j], z_xi, x_et, Net[j], z_et, x_ze, Nze[j], z_ze);
                    double dz = det3(x_xi, y_xi, Nxi[j], x_et, y_et, Net[j], x_ze, y_ze, Nze[j]);
                    g[j] = Point(dx, dy, dz) / del;
                }

                shape_f[i] = f;
                shape_g[i] = g;
            }
            break;
        }
        case 5: // brick / hexahedra
        {
            double xi[] = { 0.0,  0.5,  0.0, -0.5,  0.0,  0.5,  0.0, -0.5, -0.5,  0.5,  0.5, -0.5};
            double et[] = {-0.5,  0.0,  0.5,  0.0, -0.5,  0.0,  0.5,  0.0, -0.5, -0.5,  0.5,  0.5};
            double ze[] = {-0.5, -0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5,  0.0,  0.0,  0.0,  0.0};

            for (int i = 0; i < brick_edges; ++i) {

                double a1 = (1.0 + xi[i]) / 2.0;
                double b1 = (1.0 + et[i]) / 2.0;
                double c1 = (1.0 + ze[i]) / 2.0;

                double a2 = (1.0 - xi[i]) / 2.0;
                double b2 = (1.0 - et[i]) / 2.0;
                double c2 = (1.0 - ze[i]) / 2.0;

                double N[] = {a2*b2*c2, a1*b2*c2, a1*b1*c2, a2*b1*c2, a2*b2*c1, a1*b2*c1, a1*b1*c1, a2*b1*c1};
                double Nxi[] = {-b2*c2,  b2*c2,  b1*c2, -b1*c2, -b2*c1,  b2*c1,  b1*c1, -b1*c1};
                double Net[] = {-a2*c2, -a1*c2,  a1*c2,  a2*c2, -a2*c1, -a1*c1,  a1*c1,  a2*c1};
                double Nze[] = {-a2*b2, -a1*b2, -a1*b1, -a2*b1,  a2*b2,  a1*b2,  a1*b1,  a2*b1};

                double x_xi = 0; double x_et = 0; double x_ze = 0;
                double y_xi = 0; double y_et = 0; double y_ze = 0;
                double z_xi = 0; double z_et = 0; double z_ze = 0;

                for (int j = 0; j < brick_nodes; ++j) {
                    Point p = e.node(j).point();
                    x_xi += p.x * Nxi[j];  x_et += p.x * Net[j];  x_ze += p.x * Nze[j];
                    y_xi += p.y * Nxi[j];  y_et += p.y * Net[j];  y_ze += p.y * Nze[j];
                    z_xi += p.z * Nxi[j];  z_et += p.z * Net[j];  z_ze += p.z * Nze[j];
                }

                shape_function_vector f(N, N + brick_nodes);
                shape_gradient_vector g(brick_nodes);
                double del = det3(x_xi, y_xi, z_xi, x_et, y_et, z_et, x_ze, y_ze, z_ze);
                for (int j = 0; j < brick_nodes; ++j) {
                    double dx = det3(Nxi[j], y_xi, z_xi, Net[j], y_et, z_et, Nze[j], y_ze, z_ze);
                    double dy = det3(x_xi, Nxi[j], z_xi, x_et, Net[j], z_et, x_ze, Nze[j], z_ze);
                    double dz = det3(x_xi, y_xi, Nxi[j], x_et, y_et, Net[j], x_ze, y_ze, Nze[j]);
                    g[j] = Point(dx, dy, dz) / del;
                }

                shape_f[i] = f;
                shape_g[i] = g;
            }
            break;
        }
        default:
            assert(false);

    }

}

} // end namespace shape
