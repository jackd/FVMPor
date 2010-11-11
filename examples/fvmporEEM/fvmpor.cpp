#include "fvmpor.h"
#include "shape.h"

#include <mkl_vml_functions.h>

#include <util/doublevector.h>
#include <fvm/mesh.h>

#include <algorithm>
#include <vector>
#include <limits>
#include <utility>

#include <math.h>

// For debugging
#include <iostream>
#include <ostream>
#include <iterator>

namespace fvmpor {

    double relative_permeability(double h, const PhysicalZone& props)
    {
        if( h>=0. )
            return 1.;

        double alphaVG = props.alphaVG;
        double nVG = props.nVG;
        double mVG = props.mVG;

        double Se = 1./ pow( 1. + pow(alphaVG*fabs(h), nVG), mVG );
        return sqrt(Se) * pow( 1. - pow(1.-pow(Se,1./mVG), mVG) , 2. );
    }

    double density(double h, const Constants& constants)
    {
        double beta = constants.beta();
        double rho_0 = constants.rho_0();
        double g = constants.g();

        return rho_0*(1. + rho_0*g*beta*h);
    }

    double porosity(double h, double por_0, const Constants& constants)
    {
        double alpha = constants.beta();
        double g = constants.g();
        double rho_0 = constants.rho_0();

        return 1.-(1.-por_0)*( 1.+rho_0*g*alpha*h );
    }

    void saturation( double h, const PhysicalZone &props, double &Sw, double &dSw, double &theta )
    {
        if( h>=0. )
        {
            Sw  = 1.;
            dSw = 0.;
            theta = props.phi;
            return;
        }
        double Se;
        double alphaVG = props.alphaVG;
        double nVG = props.nVG;
        double mVG = props.mVG;
        double S_r = props.S_r;
        double phi = props.phi;

        Se = 1./ pow( 1. + pow(alphaVG*fabs(h), nVG), mVG );
        Sw = S_r + (1.-S_r)*Se;
        theta = phi*Sw;
        dSw = pow(alphaVG,nVG) * (1.-S_r) * (nVG-1.) * pow(fabs(h),nVG-1.) / pow(1.+pow(alphaVG*fabs(h),nVG), mVG+1.);
    }

    double moisture_content(double h, const PhysicalZone& props)
    {
        if( h>=0. )
            return props.phi;

        double Se;
        double alphaVG = props.alphaVG;
        double nVG = props.nVG;
        double mVG = props.mVG;
        double S_r = props.S_r;
        double phi = props.phi;

        Se = 1. / pow( 1. + pow(alphaVG*fabs(h), nVG), mVG );
        return phi*(S_r + (1.-S_r)*Se);
    }
} // end namespace fvmpor
