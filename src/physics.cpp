#include <cmath>
#include <vector>

#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
//#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "physics.h"


namespace PHiLiP
{

    template <int dim, int nstate, typename real>
    Physics<dim,nstate,real>* // returns points to base class Physics
    PhysicsFactory<dim,nstate,real>
    ::create_Physics(Parameters::AllParameters::PartialDifferentialEquation pde_type)
    {
        using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;

        if (pde_type == PDE_enum::advection) {
            return new LinearAdvection<dim, nstate, real>;
        } else if (pde_type == PDE_enum::diffusion) {
            return new Diffusion<dim, nstate, real>;
        } else if (pde_type == PDE_enum::convection_diffusion) {
            return new ConvectionDiffusion<dim, nstate, real>;
        }
        std::cout << "Can't create Physics, invalid PDE type: " << pde_type << std::endl;

        return nullptr;
    }

    // Common manufactured solution for advection, diffusion, convection-diffusion
    template <int dim, int nstate, typename real>
    void Physics<dim, nstate, real>
    ::manufactured_solution (const Point<dim,double> pos, real &solution)
    {
        real uexact;
        
        using phys = Physics<dim,nstate,real>;
        const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
        const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;
        if (dim==1) uexact = sin(a*pos[0]+d);
        if (dim==2) uexact = sin(a*pos[0]+d)*sin(b*pos[1]+e);
        if (dim==3) uexact = sin(a*pos[0]+d)*sin(b*pos[1]+e)*sin(c*pos[2]+f);
        solution = uexact;
    }


    // Linear advection functions
    template <int dim, int nstate, typename real>
    Tensor <1, dim, real> LinearAdvection<dim, nstate, real>
    ::advection_speed ()
    {
        Tensor <1, dim, real> advection_speed;
        const double pi = atan(1)*4.0;
        // Works but requires finer grid for optimal convergence.
        //if(dim >= 1) advection_speed[0] = 1.0;
        //if(dim >= 2) advection_speed[1] = -pi; // -pi/2
        //if(dim >= 3) advection_speed[2] = sqrt(2);
        //advection_speed = advection_speed / pi;

        if(dim >= 1) advection_speed[0] = 1.0;
        if(dim >= 2) advection_speed[1] = -pi/4.0;
        if(dim >= 3) advection_speed[2] = sqrt(2);

        return advection_speed;
    }

    template <int dim, int nstate, typename real>
    Tensor <1, dim, real> LinearAdvection<dim, nstate, real>
    ::convective_eigenvalues (const real &/*solution*/)
    {
        return advection_speed();
    }

    template <int dim, int nstate, typename real>
    void LinearAdvection<dim, nstate, real>
    ::convective_flux (const real &solution, Tensor <1, dim, real> &conv_flux)
    {
        // Assert conv_flux dimensions
        const Tensor <1, dim, real> velocity_field = advection_speed();
        conv_flux = velocity_field * solution;
    }

    template <int dim, int nstate, typename real>
    void LinearAdvection<dim, nstate, real>
    ::dissipative_flux (const real &/*solution*/,
                        const Tensor<1,dim,real> &/*solution_gradient*/,
                        Tensor<1,dim,real> &diss_flux)
    {
        // No dissipation
        diss_flux = 0.0;
    }

    template <int dim, int nstate, typename real>
    void LinearAdvection<dim, nstate, real>
    ::source_term (const Point<dim,double> pos, const real &/*solution*/, real &source)
    {
        const Tensor <1, dim, real> velocity_field = advection_speed();
        using phys = Physics<dim,nstate,real>;
        const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
        const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;
        if (dim==1) {
            const real x = pos[0];
            source = velocity_field[0]*a*cos(a*x+d);
        } else if (dim==2) {
            const real x = pos[0], y = pos[1];
            source = velocity_field[0]*a*cos(a*x+d)*sin(b*y+e) +
                     velocity_field[1]*b*sin(a*x+d)*cos(b*y+e);
        } else if (dim==3) {
            const real x = pos[0], y = pos[1], z = pos[2];

            source =  velocity_field[0]*a*cos(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                      velocity_field[1]*b*sin(a*x+d)*cos(b*y+e)*sin(c*z+f) +
                      velocity_field[2]*c*sin(a*x+d)*sin(b*y+e)*cos(c*z+f);
        }
    }

    // Diffusion functions
    template <int dim, int nstate, typename real>
    void Diffusion<dim, nstate, real>
    ::convective_flux (const real &/*solution*/, Tensor <1, dim, real> &/*conv_flux*/) { }

    template <int dim, int nstate, typename real>
    Tensor <1, dim, real> Diffusion<dim, nstate, real>
    ::convective_eigenvalues (const real &/*solution*/)
    {
        Tensor <1, dim, real> eig;
        for (int i=0; i<dim; ++i) {
            eig[i] = 0;
        }
        return eig;
    }

    template <int dim, int nstate, typename real>
    void Diffusion<dim, nstate, real>
    ::dissipative_flux (const real &/*solution*/,
                        const Tensor<1,dim,real> &solution_gradient,
                        Tensor<1,dim,real> &diss_flux)
    { diss_flux = -solution_gradient; }

    template <int dim, int nstate, typename real>
    void Diffusion<dim, nstate, real>
    ::source_term (const Point<dim,double> pos, const real &/*solution*/, real &source)
    {
        using phys = Physics<dim,nstate,real>;
        const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
        const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;
        if (dim==1) {
            const real x = pos[0];
            source = a*a*sin(a*x+d);
        } else if (dim==2) {
            const real x = pos[0], y = pos[1];
            source = a*a*sin(a*x+d)*sin(b*y+e) +
                     b*b*sin(a*x+d)*sin(b*y+e);
        } else if (dim==3) {
            const real x = pos[0], y = pos[1], z = pos[2];

            source =  a*a*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                      b*b*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                      c*c*sin(a*x+d)*sin(b*y+e)*sin(c*z+f);
        }
    }


    template <int dim, int nstate, typename real>
    void ConvectionDiffusion<dim, nstate, real>
    ::convective_flux (const real &solution, Tensor <1, dim, real> &conv_flux)
    {
        const Tensor <1, dim, real> velocity_field = advection_speed();
        
        conv_flux = velocity_field * solution;
        //if(dim >= 1) {
        //    conv_flux[0] = velocity_field[0] * solution;
        //}
        //if(dim >= 2) {
        //    conv_flux[1] = velocity_field[1] * solution;
        //}
        //if(dim >= 3) {
        //    conv_flux[2] = velocity_field[2] * solution;
        //}
    }

    template <int dim, int nstate, typename real>
    Tensor <1, dim, real> ConvectionDiffusion<dim, nstate, real>
    ::advection_speed ()
    {
        Tensor <1, dim, real> advection_speed;
        const double pi = atan(1)*4.0;
        // Works but requires finer grid for optimal convergence.
        //if(dim >= 1) advection_speed[0] = 1.0;
        //if(dim >= 2) advection_speed[1] = -pi; // -pi/2
        //if(dim >= 3) advection_speed[2] = sqrt(2);
        //advection_speed = advection_speed / pi;

        if(dim >= 1) advection_speed[0] = 1.0;
        if(dim >= 2) advection_speed[1] = -pi/4.0;
        if(dim >= 3) advection_speed[2] = sqrt(2);
        return advection_speed;
    }

    template <int dim, int nstate, typename real>
    Tensor <1, dim, real> ConvectionDiffusion<dim, nstate, real>
    ::convective_eigenvalues (const real &/*solution*/)
    {
        return advection_speed();
    }

    template <int dim, int nstate, typename real>
    void ConvectionDiffusion<dim, nstate, real>
    ::dissipative_flux (const real &/*solution*/,
                        const Tensor<1,dim,real> &solution_gradient,
                        Tensor<1,dim,real> &diss_flux)
    {
        diss_flux = -solution_gradient;
    }

    template <int dim, int nstate, typename real>
    void ConvectionDiffusion<dim, nstate, real>
    ::source_term (const Point<dim,double> pos, const real &/*solution*/, real &source)
    {
        const Tensor <1, dim, real> velocity_field = advection_speed();
        using phys = Physics<dim,nstate,real>;
        const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
        const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;
        if (dim==1) {
            const real x = pos[0];
            source = velocity_field[0]*a*cos(a*x+d) +
                     a*a*sin(a*x+d);
        } else if (dim==2) {
            const real x = pos[0], y = pos[1];
            source = velocity_field[0]*a*cos(a*x+d)*sin(b*y+e) +
                     velocity_field[1]*b*sin(a*x+d)*cos(b*y+e) +
                     a*a*sin(a*x+d)*sin(b*y+e) +
                     b*b*sin(a*x+d)*sin(b*y+e);
        } else if (dim==3) {
            const real x = pos[0], y = pos[1], z = pos[2];
            source =   velocity_field[0]*a*cos(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                       velocity_field[1]*b*sin(a*x+d)*cos(b*y+e)*sin(c*z+f) +
                       velocity_field[2]*c*sin(a*x+d)*sin(b*y+e)*cos(c*z+f) +
                       a*a*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                       b*b*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                       c*c*sin(a*x+d)*sin(b*y+e)*sin(c*z+f);
        }
    }
    // Instantiate
    template class Physics < PHILIP_DIM, 1, double >;
    template class PhysicsFactory<PHILIP_DIM, 1, double>;
    template class LinearAdvection < PHILIP_DIM, 1, double >;
    template class Diffusion < PHILIP_DIM, 1, double >;
    template class ConvectionDiffusion < PHILIP_DIM, 1, double >;

    template class Physics < PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
    template class PhysicsFactory<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
    template class LinearAdvection < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;
    template class Diffusion < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;
    template class ConvectionDiffusion < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;


} // end of PHiLiP namespace
